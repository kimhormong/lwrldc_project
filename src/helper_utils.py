import random
import json
import os
from datetime import datetime

from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import (
    MulticlassAccuracy,
)


def unnormalize(tensor):
    """
    Reverses the normalization of a PyTorch image tensor.

    This function takes a normalized tensor and applies the inverse
    transformation to return the pixel values to the standard [0, 1] range.
    The mean and standard deviation values used for the original
    normalization are hardcoded within this function.

    Args:
        tensor (torch.Tensor): The normalized input tensor with a shape of
                               (C, H, W), where C is the number of channels.

    Returns:
        torch.Tensor: The unnormalized tensor with pixel values clamped to
                      the valid [0, 1] range.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = torch.tensor([0.485, 0.490, 0.451])
    std = torch.tensor([0.214, 0.197, 0.191])
    
    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()
    
    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)
        
    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    # Return the unnormalized tensor.
    return unnormalized_tensor


def training_loop(
    model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None,
):
   
    # Move the model to the specified computation device.
    model.to(device)

    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "train_accuracy": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)

    train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)

    # Set up a single progress bar for the entire training process.
    total_steps = (len(train_loader) + len(val_loader)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    # Begin the main training loop over the specified number of epochs.
    for epoch in range(num_epochs):    
        # --- Training Phase ---
        
        # Reset metric calculators for the new training epoch.
        train_accuracy.reset()
          
        # Set the model to training mode.
        model.train()
        
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)
            # Forward pass: compute predicted outputs by passing inputs to the model.
            outputs = model(inputs)
            # Calculate the loss.
            loss = loss_function(outputs, labels)
            # perform a backward pass
            loss.backward()
            # Update the model weights.
            optimizer.step()
            
            # Update metrics with the current batch's predictions and labels.
            preds = outputs.argmax(dim=1)
            train_accuracy.update(preds, labels)
            
            # Update the count of processed validation samples.
            batch_size = inputs.size(0)
            train_samples_processed += batch_size
            
            # Accumulate the loss, weighted by the batch size.
            running_train_loss += loss.item() * batch_size
            
            # Calculate and display the running accuracy and loss.  
            display_acc = train_accuracy.compute().item()
            display_loss = running_train_loss / train_samples_processed

            pbar.set_postfix(
                train_acc=f"{display_acc:.4%}", 
                train_loss=f"{display_loss:.4f}",
                )
            
            # Update the progress bar for the batch.
            pbar.update(1)

        
        # Compute average accuracy, average loss for the epoch and store them in the history.
        epoch_train_acc = train_accuracy.compute().item()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_accuracy"].append(epoch_train_acc)
        history["train_loss"].append(epoch_train_loss)
       

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        # Initialize variables to accumulate validation loss.
        running_val_loss = 0.0
        val_samples_processed = 0
        
        # Reset metric calculators for the new validation epoch.
        val_accuracy.reset()

        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over the validation data loader.
            for inputs, labels in val_loader:
                # Update the progress bar description for the validation phase.
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Validation]")
                
                # Move input data and labels to the designated device.
                inputs, labels = inputs.to(device), labels.to(device)
               
                # Compute model outputs.
                outputs = model(inputs)
                # Calculate the validation loss.
                loss = loss_function(outputs, labels)
                
                # Update metrics with the current batch's predictions and labels.
                preds = outputs.argmax(dim=1)
                val_accuracy.update(preds, labels)
                
                # Update the count of processed validation samples.
                batch_size = inputs.size(0)
                val_samples_processed += batch_size
                
                # Accumulate the validation loss.
                running_val_loss += loss.item() * batch_size

                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                pbar.set_postfix(
                    val_acc=f"{current_acc:.4%}",
                    val_loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate validation accuracy and average validation loss for the epoch, store them in the history.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute().item()
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Acc: {epoch_train_acc:.4%}, "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4%}"
        )

        # --- SCHEDULER ---
        # Adjust the learning rate based on the scheduler's logic, if one is provided.
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_acc)
            else:
                scheduler.step()
    
                
    # Close the progress bar after the training loop is complete.
    pbar.close()
    
    return model, history



def plot_training_history(history, model_name="Custom DenseNet"):
    """Visualizes the training and validation history of a model.

    This function generates and displays two plots: one for training and
    validation loss, and another for validation accuracy. It also highlights
    the epoch where the highest validation accuracy was achieved.

    Args:
        history (dict): A dictionary containing the model's training history.
                        It must include the keys 'val_accuracy', 'val_loss',
                        and 'train_loss'.
        model_name (str, optional): The name of the model, used for plot
                                    titles and labels. Defaults to "Custom DenseNet".
    """
    # Find the index of the epoch with the highest validation accuracy.
    best_epoch_idx = np.argmax(history['val_accuracy'])
    # Get the best validation accuracy and the corresponding validation loss.
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]

    # Print a summary of the model's performance at the best epoch.
    print("---------- Best Epoch Performance ----------")
    print(f"Model: {model_name}")
    print(f"Epoch: {best_epoch_idx + 1}")
    print(f"Validation Accuracy: {best_val_acc:.2%}")
    print(f"Validation Loss:     {best_val_loss:.4f}")
    print("------------------------------------------\n")

    # Set up the figure and subplots for displaying the history.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Define colors for plot elements to ensure consistency.
    train_color = 'blue'
    val_color = 'red'
    best_epoch_color = 'green'

    # Plot training and validation loss on the first subplot.
    ax1.plot(history['train_loss'], label=f'{model_name} Train Loss', color=train_color, linestyle='-')
    ax1.plot(history['val_loss'], label=f'{model_name} Val Loss', color=val_color, linestyle='--')

    # Highlight the validation loss at the best-accuracy epoch with a marker.
    ax1.plot(best_epoch_idx, best_val_loss, marker='o', color=best_epoch_color, markersize=8, label='Loss When Best Acc Was Achieved')
    # Annotate the marker with its precise value.
    ax1.annotate(f'{best_val_loss:.4f}',
                 xy=(best_epoch_idx, best_val_loss),
                 xytext=(best_epoch_idx, best_val_loss + 0.1),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the loss subplot.
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy on the second subplot.
    ax2.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy', color=val_color)

    # Highlight the best validation accuracy with a marker.
    ax2.plot(best_epoch_idx, best_val_acc, marker='o', color=best_epoch_color, markersize=8, label='Best Accuracy Achieved')
    # Annotate the marker with its value.
    ax2.annotate(f'{best_val_acc:.2%}',
                 xy=(best_epoch_idx, best_val_acc),
                 xytext=(best_epoch_idx, best_val_acc - 0.05),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the accuracy subplot.
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Determine an appropriate interval for x-axis ticks for readability.
    num_epochs = len(history['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Generate tick locations (0-indexed) and corresponding labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom x-axis ticks to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust subplot parameters for a tight layout and display the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def visualize_predictions(model, dataloader, class_names, device):
    """Visualizes model predictions on a sample of images from a dataset.

    This function randomly selects one image from each class in the provided
    dataloader. It then performs inference using the given model and displays
    the images in a grid. Each image is titled with its true and predicted
    labels, colored green for correct predictions and red for incorrect ones.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for inference.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset to visualize.
        class_names (list of str): A list mapping class indices to their names.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform inference.
    """
    # Prepare the model for inference.
    model.to(device)
    model.eval()

    # --- Create a mapping from class index to a list of sample indices for that class ---
    # Initialize a dictionary to hold indices for each class.
    class_to_indices = {i: [] for i in range(len(class_names))}
    # Access the targets and indices from the underlying dataset and subset.
    full_dataset_targets = dataloader.dataset.subset.dataset.targets
    subset_indices = dataloader.dataset.subset.indices
    # Populate the dictionary by mapping each sample's true label to its index within the subset.
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)
    # ---

    # Create a grid of subplots to display the images.
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(18, 8))

    # Disable gradient computations for the inference phase.
    with torch.no_grad():
        # Loop through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # If there are more subplots than classes, turn off the extra ones.
            if i >= len(class_names):
                ax.axis('off')
                continue

            # Randomly select one image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Get the image tensor and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]
            
            # Prepare the image tensor for the model by adding a batch dimension and moving it to the device.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Pass the image through the model to get the output logits.
            outputs = model(image_batch)
            # Determine the predicted class index by finding the index of the maximum logit.
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()
            
            # Set the title color to green for correct predictions and red for incorrect ones.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            # Set the subplot's title with the predicted and true labels.
            ax.set_title(
                f'Predicted: {class_names[predicted_label]}\n(True: {class_names[true_label]})',
                color=title_color
            )
            
            # Reverse the normalization of the image tensor for proper visualization.
            img_to_plot = unnormalize(image_tensor)
            
            # Convert the tensor to a NumPy array and adjust dimensions for displaying.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            # Display the image and hide the axis ticks.
            ax.axis('off')

    # Adjust the layout to prevent titles from overlapping and show the plot.
    plt.tight_layout()
    plt.show()


def evaluate(model, data_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    return acc

def get_device():
    """Determines the best available device for PyTorch computations.

    This function checks for the availability of CUDA (GPU) and MPS (Apple Silicon GPU)
    support in that order. If neither is available, it defaults to using the CPU.

    Returns:
        torch.device: The best available device for PyTorch computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_config(config_path="../configs/config.yaml"):
    """Loads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file. Defaults to "configs/config.yaml".

    Returns:
        dict: A dictionary containing the configuration parameters loaded from the YAML file.
    """

    return OmegaConf.load(config_path)

def _to_dict(cfg):
    """Helper to convert DictConfig objects or raw objects into standard Python dicts."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg if cfg is not None else {}

def _update_latest_pointer(role_dir, run_folder, role_name):
    """
    Saves or updates the pointer file (latest.json) inside runs/teacher/
    recording the absolute path to the most recent teacher run directory and model file.
    """
    pointer_path = os.path.join(role_dir, "latest.json")
    model_filename = f"{role_name}_model.pth"
    
    pointer_data = {
        "latest_run_dir": os.path.abspath(run_folder),
        "latest_model_path": os.path.abspath(os.path.join(run_folder, model_filename)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    os.makedirs(role_dir, exist_ok=True)
    
    with open(pointer_path, "w") as f:
        json.dump(pointer_data, f, indent=4)

def save_model_and_results(
    model, 
    model_config, 
    data_config, 
    history, 
    is_teacher=False, 
    teacher_path=None, 
    base_dir="../runs"
):
    """
    Saves model weights (.pth) and a single results.json file containing 
    merged configs, metrics, and (for students) the exact teacher model path used.
    Only updates latest.json when saving a Teacher model.
    """
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Setup role subfolder (runs/teacher/ or runs/student/)
    role = "teacher" if is_teacher else "student"
    role_dir = os.path.join(base_dir, role)
    run_folder = os.path.join(role_dir, f"run_{file_timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # 2. Save PyTorch Model Weights
    model_filename = f"{role}_model.pth"
    model_save_path = os.path.join(run_folder, model_filename)
    torch.save(model.state_dict(), model_save_path)

    # 3. Merge model & data configurations
    if not is_teacher and teacher_path is not None:
        model_config["teacher_source"] = teacher_path  # Store the exact teacher path used for this student run

    combined_config = {
        "model": _to_dict(model_config),
        "data": _to_dict(data_config)
    }

    # 4. Build JSON payload
    run_data = {
        "timestamp": log_timestamp,
        "model_file": model_filename,
        "config": combined_config,
        "metrics": history,
     }
    

    # 5. Save JSON Results
    json_save_path = os.path.join(run_folder, "results.json")
    with open(json_save_path, "w") as f:
        json.dump(run_data, f, indent=4)

    # 6. Update latest.json pointer ONLY if training a Teacher
    if is_teacher:
        _update_latest_pointer(role_dir, run_folder, role)
        print(f"--> Updated teacher pointer at: {os.path.join(role_dir, 'latest.json')}")

    print(f"--> Saved {role} run and results.json to: {run_folder}")
    
def load_teacher_path(path_or_mode="latest", base_dir="../runs"):
    """
    Loads teacher path:
    - "latest": Scans runs/teacher/ for the newest run_YYYYMMDD_HHMMSS folder containing a valid .pth file.
    - Exact Path: Validates and returns the exact path provided.
    """
    # 1. Handle missing/empty configuration
    if path_or_mode is None or str(path_or_mode).strip() in ("", "None"):
        raise ValueError("[ERROR] 'teacher_source' was not provided in config!")

    teacher_dir = os.path.join(base_dir, "teacher")

    # 2. AUTO-SCANNER MODE
    if str(path_or_mode).lower() == "latest":
        if not os.path.exists(teacher_dir):
            raise FileNotFoundError(f"[ERROR] Teacher directory '{teacher_dir}' does not exist!. run \"python 03_train_teacher.py\" first to train and save a teacher model, or provide an exact path to a teacher")

        # Find all run_ directories
        run_folders = [
            os.path.join(teacher_dir, f) for f in os.listdir(teacher_dir)
            if f.startswith("run_") and os.path.isdir(os.path.join(teacher_dir, f))
        ]

        # Sort alphabetically descending (run_20260730_200000 comes before run_20260730_100000)
        run_folders.sort(reverse=True)

        # Pick the newest folder that actually contains a complete model file
        for folder in run_folders:
            candidate_file = os.path.join(folder, "teacher_model.pth")
            if os.path.exists(candidate_file):
                return os.path.abspath(candidate_file)

        raise FileNotFoundError(f"[ERROR] No valid 'teacher_model.pth' found inside any run folder in '{teacher_dir}'!")

    # 3. EXACT MANUAL PATH MODE
    else:
        exact_path = os.path.abspath(str(path_or_mode))
        if not os.path.exists(exact_path):
            raise FileNotFoundError(f"[ERROR] Specified teacher file does not exist: {exact_path}. run \"python 03_train_teacher.py\" to train and save a teacher model")
        return exact_path