import torch.nn as nn
import torch.optim as optim
import timm

from paddy_data_loader import load_train_val_test_data
import helper_utils


if __name__ == "__main__":
    # 1. load configuration from config.yaml
    teacher_configs = helper_utils.load_config("../configs/teacher_config.yaml")
    dataset_configs = helper_utils.load_config("../configs/dataset_config.yaml")

    # 2. Access values using the keys
    hyperparams = teacher_configs.hyperparameters
    models = teacher_configs.models
    
    dataset = dataset_configs.dataset
    
    device = helper_utils.get_device()

    train_loader, val_loader, test_loader = load_train_val_test_data(batch_size=hyperparams.teacher_batch_size)

    densenet201 = timm.create_model(
        models.teacher_timm_name,
        pretrained=False,
        num_classes=dataset.num_classes,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(densenet201.parameters(), lr=hyperparams.teacher_lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams.teacher_epochs, eta_min=hyperparams.teacher_eta_min)

    trained_densenet201, history = helper_utils.training_loop(
        model=densenet201,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=hyperparams.teacher_epochs,
        device=device,
        scheduler=lr_scheduler,
    )
    
    test_acc = helper_utils.evaluate(trained_densenet201, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4%}")
    
    history['test_acc'] = test_acc    
    helper_utils.save_model_and_results(trained_densenet201, teacher_configs, dataset_configs, history, is_teacher=True)