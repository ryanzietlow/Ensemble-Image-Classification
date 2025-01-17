import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet, AlexNet_Weights
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from dataloader import get_cifar100_dataloader


def select_model(model_name):
    """
    Initialize model with proper final layer modifications for CIFAR-100
    """
    if model_name.lower() == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
    elif model_name.lower() == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
    elif model_name.lower() == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
    else:
        raise ValueError("Model name must be 'alexnet', 'vgg16', or 'resnet18'")
    return model


def plot_losses(train_losses, val_losses, model_name, epoch, save_dir='loss_plots'):
    """
    Helper function to plot and save loss curves
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss for {model_name} - Epoch {epoch + 1}")
    plt.legend()
    plt.grid(True)
    # Save plot
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss_plot_.png"))
    plt.close()


def train_model(model, model_name, batch_size, learning_rate, patience, device, max_epochs=500):
    # Modified optimizer for VGG16
    if model_name.lower() == 'vgg16':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[30, 60, 90],
                                                   gamma=0.1)  # Reduce LR at specific epochs
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    criterion = nn.CrossEntropyLoss()

    # Get dataloaders
    train_loader = get_cifar100_dataloader(batch_size=batch_size, train=True)
    test_loader = get_cifar100_dataloader(batch_size=batch_size, train=False)
    model = model.to(device)

    # Training tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epoch = 0

    # Time tracking
    start_time = datetime.now()
    first_5_epochs_time = None

    print(f"\nStarting training at {start_time.strftime('%H:%M:%S')}")

    while epoch < max_epochs:
        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"{model_name} - Epoch {epoch + 1} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"{model_name} - Epoch {epoch + 1} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

        val_loss = val_running_loss / len(test_loader.dataset)
        val_losses.append(val_loss)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Generate and save loss plot after each epoch
        plot_losses(train_losses, val_losses, model_name, epoch)

        # Save model after first 5 epochs
        if epoch == 4:
            # Save model to weights folder
            torch.save(model.state_dict(), os.path.join('weights', f"{model_name}_epoch5.pth"))
            first_5_epochs_time = datetime.now() - start_time
            print(f"First 5 epochs completed in {str(first_5_epochs_time)}")

        # Print progress with timestamp
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"{current_time} - Epoch [{epoch + 1}/{max_epochs}], "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Check for improvement and update early stopping criteria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ## Save model
            torch.save(model.state_dict(), os.path.join('weights', f"{model_name}_best.pth"))
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for {model_name}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        epoch += 1

    # Calculate total training time
    total_training_time = datetime.now() - start_time

    # Print training summary
    print("\nTraining Summary:")
    print(f"Time taken for first 5 epochs: {str(first_5_epochs_time)}")
    print(f"Total training time: {str(total_training_time)}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"Model saved as {model_name}_best.pth")
    print(f"Batch size: {batch_size}")


def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Train models on CIFAR-100 with early stopping and learning rate scheduling.")
    parser.add_argument('-model', type=str, choices=['alexnet', 'vgg16', 'resnet18', 'AlexNet', 'VGG16',
                                                     'ResNet18'], required=True, help="Model to train (case insensitive)")
    parser.add_argument('-b', type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument('-lr', type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument('-p', type=int, default=30, help="Patience for early stopping (default: 30)")
    parser.add_argument('-maxE', type=int, default=500, help="Maximum number of epochs (default: 500)")
    parser.add_argument('-device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to train on: 'cuda' or 'cpu' (default: cuda if available)")
    args = parser.parse_args()

    # Initialize model
    model = select_model(args.model)

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.b}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Early stopping patience: {args.p}")
    print(f"Maximum epochs: {args.maxE}")
    print(f"Device: {args.device}")

    torch.cuda.empty_cache()
    os.makedirs('weights', exist_ok=True)

    # Start training
    train_model(model, args.model, args.b, args.lr, args.p, args.device, args.maxE)
    print(f"Training complete for {args.model}.\n")


if __name__ == "__main__":
    main()
