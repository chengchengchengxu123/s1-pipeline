import torch
import sys
import os

# 确保能导入 src 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.attention_rewrite import scaled_dot_product_attention

def test_attention():
    print("\n" + "="*50)
    print("开始验证 Attention 机制 (纯 CPU 运行)")
    print("="*50)

    torch.manual_seed(42) 
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)

    print("Q, K, V 矩阵的形状均为:", q.shape)

    # 场景一：无 Mask
    print("\n--- [场景一] 无 Mask ---")
    out_no_mask, attn_no_mask = scaled_dot_product_attention(q, k, v, use_mask=False)
    print("注意力权重矩阵 (形状 2x3x3):")
    print(attn_no_mask[0].detach().numpy().round(4)) 
    
    row_sums = attn_no_mask.sum(dim=-1)
    print("\n检查每一行的和:")
    print(row_sums.detach().numpy().round(4))
    
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "无 Mask 时，行和不为 1！"
    print("✅ 验证通过：无 Mask 时，每一行的权重之和均为 1.0")

    # 场景二：有 Mask
    print("\n--- [场景二] 有 Causal Mask ---")
    out_has_mask, attn_has_mask = scaled_dot_product_attention(q, k, v, use_mask=True)
    print("加了 Mask 后的权重矩阵 (看第一个 Batch):")
    print(attn_has_mask[0].detach().numpy().round(4))
    
    upper_tri = torch.triu(attn_has_mask, diagonal=1)
    assert torch.all(upper_tri == 0), "Mask 失败，模型偷看了未来的词！"
    print("\n✅ 验证通过：上三角区域已被 Mask 屏蔽，权重全为 0 (模型只能看过去和现在)")

# 👇 关键：确保有这一段，否则直接运行文件时什么都不会发生 👇
if __name__ == "__main__":
    test_attention()