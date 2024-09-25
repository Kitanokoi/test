import torch as th
from torch import nn


class SelfAttention(nn.Module):    
    """
    Import x.shape = [bs, t, s]
    """
    
    def __init__(self, d_k, d_v) -> None:
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.lN = nn.LayerNorm(d_v, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q,k,v):
        d_k, d_v= self.d_k, self.d_v
        bs, q_len, k_len, v_len = v.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = v
        
        q = self.w_qs(q).view(bs, q_len, -1)
        k = self.w_ks(k).view(bs, k_len, -1)
        v = self.w_vs(v).view(bs, v_len, -1) 
        
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        q = th.matmul(q,k).view(bs, q_len, -1)
        q = q[-1]//(d_v**0.5)
        q = self.softmax(q)

        q = th.matmul(q,v).view(bs, q_len, -1)
        
        q += residual

        q = self.lN(q)
        
        return q
        
class TrafficFeatureExaction(nn.Module):
    def __init__(self, d_in) -> None:
        super().__init__()

        self.d_in = d_in

        self.fn1 = nn.Linear(d_in, 4*d_in, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fn2 = nn.Linear(4*d_in, d_in, bias=True)

    def forward(self, q):
        q = self.fn1(q)
        q = self.relu(q)
        q = self.dropout(q)
        q = self.fn2(q)

        return q

class LiteFFBlock(nn.Module):
    def __init__(self, d_k, d_v, t_len) -> None:
        super().__init__()

        self.pooling = nn.AvgPool1d(t_len//2)
        self.attention = SelfAttention(d_k, d_v)
        self.featureExac = TrafficFeatureExaction(d_v)

    def forward(self,q,k,v):
        q = q.transpose(1,2)
        q = self.pooling(q)
        q = q.transpose(1,2)

        q = self.attention(q,k,v)
        q = self.featureExac(q)

        return q
    
class NormalBlock(nn.Module):
    def __init__(self, d_k, d_v) -> None:
        super().__init__()

        self.attention = SelfAttention(d_k, d_v)
        self.featureExc = TrafficFeatureExaction(d_v)
    
    def forward(self, q, k, v):
        q = self.attention(q,k,v)
        q = self.featureExc(q)

        return q

class ETFlowFormer(nn.Module):
    def __init__(self, d_k, d_v, t_len, labels) -> None:
        super().__init__()

        self.nb = NormalBlock(d_k,d_v)
        self.liteFF1 = LiteFFBlock(d_k,d_v,t_len)
        self.liteFF2 = LiteFFBlock(d_k,d_v,t_len//2)
        self.liteFF3 = LiteFFBlock(d_k,d_v,t_len//4)
        self.globalavgpooling = nn.AvgPool1d(1)
        self.fn = nn.Linear(3*d_v,labels)
        self.softmax = nn.Softmax(labels)

    def forward(self, x):
        bs, _, _ = x.shape
        x = self.nb(x,x,x)
        x = self.liteFF1(x,x,x)
        feature1 = x

        x = self.liteFF2(x,x,x)
        feature2 = x

        x = self.liteFF3(x,x,x)
        feature3 = x

        feature = th.cat([feature1,feature2,feature3],dim=1)

        feature = feature.transpose(1,2)
        feature = self.globalavgpooling(feature)
        feature = feature.view(bs,-1)
        feature = self.fn(feature)
        output = self.softmax(feature)

        return output



