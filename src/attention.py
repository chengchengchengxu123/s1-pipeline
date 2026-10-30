import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: Query 张量, shape: (batch_size, seq_len_q, d_k)
    K: Key 张量, shape: (batch_size, seq_len_k, d_k)
    V: Value 张量, shape: (batch_size, seq_len_v, d_v) 
       注意：在自注意力中，通常 seq_len_q == seq_len_k == seq_len_v
    mask: 因果掩码，shape: (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # 1. 计算 Q 和 K 的点积，并除以 sqrt(d_k)
    # K.transpose(-2, -1) 会把形状变成 (batch_size, d_k, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用 Mask
    if mask is not None:
        # 重点：将 mask 为 0 的位置（即未来位置）填充为负无穷大
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
    # 3. 计算 Softmax，在最后一个维度（Key 的维度）上进行归一化
    weights = F.softmax(scores, dim=-1)
    
    # 4. 乘以 Value
    output = torch.matmul(weights, V)
    
    return output, weights