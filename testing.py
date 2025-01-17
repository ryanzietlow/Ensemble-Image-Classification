import os.path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet, resnet18, vgg16
import argparse
from tqdm import tqdm
from dataloader import get_cifar100_dataloader


def get_model(model_name):
    """
    Initialize the specified model architecture.
    """
    if model_name.lower() == 'alexnet':
        model = alexnet(weights=None, num_classes=100)
    elif model_name.lower() == 'resnet18':
        model = resnet18(weights=None, num_classes=100)
    elif model_name.lower() == 'vgg16':
        model = vgg16(weights=None, num_classes=100)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def load_cifar100(batch_size):
    """
    Load and preprocess CIFAR100 test dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 as per your training
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return test_loader


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """
    Calculate top-k accuracy for given k values.
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def test_model(model, test_loader, device):
    """
    Test the model and return top-1 and top-5 error rates.
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            acc1, acc5 = calculate_accuracy(outputs, targets, topk=(1, 5))

            top1_correct += acc1.item() * targets.size(0) / 100
            top5_correct += acc5.item() * targets.size(0) / 100
            total += targets.size(0)

    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100

    return 100 - top1_accuracy, 100 - top5_accuracy  # Convert to error rates


def main():
    parser = argparse.ArgumentParser(description='Test trained models on CIFAR100')
    parser.add_argument('-model', type=str, choices=['alexnet', 'vgg16', 'resnet18', 'AlexNet', 'VGG16',
                        'ResNet18'], required=True, help="Model to train (case insensitive)")
    parser.add_argument('-b', type=int, default=128,
                        help='Batch size for testing (default: 128)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = get_model(args.model)
    model = model.to(device)

    # Load test data
    test_loader = get_cifar100_dataloader(batch_size=args.b, train=False)

    # Test weights from epoch 5 from weights folder
    weight_path = os.path.join('weights', f"{args.model}_epoch5.pth")

    # weight_path = f"{args.model}_test_epoch5.pth"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print(f"\nTesting {args.model} with weights from epoch 5...")
        top1_error, top5_error = test_model(model, test_loader, device)
        print(f"Results for {args.model} at epoch 5:")
        print(f"Top-1 Error Rate: {top1_error:.2f}%")
        print(f"Top-5 Error Rate: {top5_error:.2f}%")
    except FileNotFoundError:
        print(f"Could not find weights file: {weight_path}")

    # Test final weights
    weight_path = os.path.join('weights', f"{args.model}_best.pth")
    # weight_path = f"{args.model}_final.pth"
    try:
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print(f"\nTesting {args.model} with final weights...")
        top1_error, top5_error = test_model(model, test_loader, device)
        print(f"Results for {args.model} after training completion:")
        print(f"Top-1 Error Rate: {top1_error:.2f}%")
        print(f"Top-5 Error Rate: {top5_error:.2f}%")
    except FileNotFoundError:
        print(f"Could not find weights file: {weight_path}")

    # # Test best weights from weights folder
    # os.path.join('weights', f"{args.model}_best.pth")
    #
    # # weight_path = f"{args.model}_test_best.pth"
    # try:
    #     model.load_state_dict(torch.load(weight_path, weights_only=True))
    #     print(f"\nTesting {args.model} with best validation weights...")
    #     top1_error, top5_error = test_model(model, test_loader, device)
    #     print(f"Results for {args.model} with best validation weights:")
    #     print(f"Top-1 Error Rate: {top1_error:.2f}%")
    #     print(f"Top-5 Error Rate: {top5_error:.2f}%")
    # except FileNotFoundError:
    #     print(f"Could not find weights file: {weight_path}")


if __name__ == '__main__':
    main()