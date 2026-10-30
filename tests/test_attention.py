import torch
import pytest
from src.attention import scaled_dot_product_attention

def test_attention_shape_and_mask():
    seq_len = 5
    batch_size = 2
    d_k = 4
    
    # 构造随机输入
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    # 构造下三角全为1，上三角全为0的因果掩码
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    output, weights = scaled_dot_product_attention(Q, K, V, mask)
    
    # 断言 1: 输出和权重的形状必须正确
    assert output.shape == (batch_size, seq_len, d_k)
    assert weights.shape == (batch_size, seq_len, seq_len)
    
    # 断言 2: 权重的每一行相加必须等于 1 (Softmax的特性)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), "每一行的权重和必须为1"
    
    # 断言 3: 上三角（未来位置）的权重必须为 0
    # 提取对角线以上的元素
    upper_tri = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6), "未来位置的权重必须为0"

if __name__ == "__main__":
    pytest.main(["-v", __file__])