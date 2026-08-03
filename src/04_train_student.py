import sys

import torch
import torch.optim as optim
import timm

from paddy_data_loader import load_train_val_test_data
from fitnet_wrapper import FitNetWrapper
from shufflenet_v2 import ShuffleNetV2
import kd_utils
import helper_utils

if __name__ == "__main__":
    # 1. load configuration from config.yaml
    student_configs = helper_utils.load_config("../configs/student_config.yaml")
    dataset_configs = helper_utils.load_config("../configs/dataset_config.yaml")

    # 2. Access values using the keys
    fitnet = student_configs.fitnet
    hyperparams = student_configs.hyperparameters
    dataset = dataset_configs.dataset
    model = student_configs.models
    
    # 3. Safely extract 'teacher_source'. If missing, defaults to "latest" instead of None
    teacher_source = student_configs.get("teacher_source", "latest")
    
    try:
        teacher_path = helper_utils.load_teacher_path(teacher_source)
        print(f"--> Using Teacher Model from: {teacher_path}")
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)

    device = helper_utils.get_device()

    train_loader, val_loader, test_loader = load_train_val_test_data(batch_size=hyperparams.student_batch_size)

    # Load the pre-trained teacher model (DenseNet-201) using the timm library and load its state dictionary
    densenet_201_teacher = timm.create_model(model.teacher_timm_name, pretrained=False, num_classes=dataset.num_classes,)
    densenet_201_teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    # Initialize the FitNetWrapper with the teacher and student models, and specify the hint and guided layers along with their channel dimensions.
    fitnet_wrapper = FitNetWrapper(teacher=densenet_201_teacher, 
                               student=ShuffleNetV2(n_class=dataset.num_classes, 
                                model_size=model.student_model_size), 
                               hint_layer_name=fitnet.hint_layer_name,
                               guided_layer_name=fitnet.guided_layer_name,
                                student_channels=fitnet.student_channel,
                                teacher_channels=fitnet.teacher_channel,)

    # Perform the hint training loop to train the student model using the hints from the teacher model.
    trained_fitnet_wrapper = kd_utils.hint_training_loop(fitnet_wrapper, 
                                                     train_loader, 
                                                     num_epochs=hyperparams.hint_epochs, 
                                                     device=device)

    # After the hint training, we can proceed with the knowledge distillation training loop to further train the student model using the teacher's outputs as soft targets.
    optimizer = torch.optim.Adam(trained_fitnet_wrapper.student.parameters(), lr=hyperparams.student_lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams.distillation_epochs, eta_min=hyperparams.student_eta_min)

    trained_shufflenet_v2_student, history = kd_utils.student_training_loop(
        teacher=densenet_201_teacher,
        student=trained_fitnet_wrapper.student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        temperature=hyperparams.temperature,
        alpha=hyperparams.alpha,
        num_epochs=hyperparams.distillation_epochs,
        device=device,
        scheduler=lr_scheduler,
    )

    test_acc = helper_utils.evaluate(trained_shufflenet_v2_student, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4%}")
    
    history['test_acc'] = test_acc    
    helper_utils.save_model_and_results(trained_shufflenet_v2_student, student_configs, dataset_configs, history, is_teacher=False, teacher_path=teacher_path)
    