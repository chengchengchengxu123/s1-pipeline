import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_fashion_mnist_loaders(batch_size=64, data_dir="./data/fashion_mnist"):
    # 1. 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 2. 加载数据集
    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    
    # 3. 构建 DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader