import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import notebook as tqdm


def get_loaders(bs=128, root="."):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root, train=True, 
                                                        download=True, transform=transform)
    val_dataset = torchvision.datasets.FashionMNIST(root, train=False, 
                                                        download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    return train_loader, val_loader