import torch
from src.models import FashionCNN

def test_fashion_cnn_output_shape():
    model = FashionCNN()
    # 模拟一个 Batch 的输入（64张图，1通道，28x28）
    dummy_input = torch.randn(64, 1, 28, 28)
    
    # 前向传播
    output = model(dummy_input)
    
    # 验证输出形状必须是 [64, 10]
    assert output.shape == (64, 10)
    
    # 验证参数量是否合理（大约在40万左右）
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params < 1_000_000  # 确保模型不要太大