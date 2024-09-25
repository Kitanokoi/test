import torch as th
from torch import nn


class MutiHeadAttention(nn.Module):    
    """
    Import x.shape = [bs, t, d]
    """

    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1) -> None:
        super().__init__()
        
        self.n_head = n_head
        self.d_modle = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        w_qs = nn.Linear(d_model, n_head*d_k, bias=False)
        w_ks = nn.Linear(d_model, n_head*d_k, bias=False)
        w_vs = nn.Linear(d_model, n_head*d_v, bias=False)
        f_C = nn.Linear(n_head*d_v, d_v, bias=False)
        
        
        self.attention = ScaledDotProductAttention(temperature=d_v**0.5)

        self.dropout = nn.Dropout(dropout)
        self.lN = nn.LayerNorm(d_v, eps=1e-6)
        
    def forward(self, x):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        x_bs, t_len= x.size(0), x.size(1)
        
        residual = x
        
        q = self.w_qs(x).view(x_bs, t_len, -1)
        k = self.w_ks(x).view(x_bs, t_len, -1)
        v = self.w_vs(x).view(x_bs, t_len, -1) 
        
        q, attn = self.attention(q, k, v)
        
        q += x

        q = self.lN(q)
        
        return q, attn
        

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperatue) -> None:
        super().__init__()
        
        self. 
            
