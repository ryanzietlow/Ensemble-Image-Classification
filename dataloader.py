import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar100_dataloader(batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.Resize(224),
        # expected input size for alexnet and vgg, resnet needs multiple of 32, 224 works for this
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # these values represent the mean and std of the pixel values for each colour channel (RGB) in the CIFAR-100 dataset
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if train:
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader

# Usage example:
# train_loader = get_cifar100_dataloader(batch_size=128, train=True)
# test_loader = get_cifar100_dataloader(batch_size=128, train=False)