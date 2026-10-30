import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q,k,v,use_mask=False):
    d_k=q.size(-1)
    scores=torch.matmul(q,k.transpose(-2,-1))/(d_k**0.5)
    if use_mask:
        seq_len=q.size(-2)
        mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool()
        scores=scores.masked_fill(mask,float('-inf'))
    attn_weights=F.softmax(scores,dim=-1)
    output=torch.matmul(attn_weights,v)
    return output,attn_weights