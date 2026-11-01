import torch
from src.dataset import get_fashion_mnist_loaders

def test_dataloader_batch_shape():
    train_loader, _ = get_fashion_mnist_loaders(batch_size=64)
    images, labels = next(iter(train_loader))
    
    # 验证 Batch 维度
    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)
    
    # 验证数据类型
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    
    # 验证标签范围 (0-9)
    assert labels.min() >= 0 and labels.max() <= 9