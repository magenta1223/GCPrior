'''
Set Transformer code from :
https://github.com/juho-lee/set_transformer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        # projection
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        # head split
        dim_split = self.dim_V // self.num_heads
        # channel 방향으로 쪼갠걸 batch 방향으로 붙임.. ? 
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # 일반적인 attention
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        # Query를 resuidual conneciton으로 붙임. 그리고 다시 batch축을 쪼개고 dimension축으로 붙임. 이런짓을 해도되나.. ? 
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # layernorm & FFN & layernorm
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    """
    MAB의 self-attention version
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    """
    Inverse SAB. 
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out)) # target에 대한 leanred embedding 역할
        nn.init.xavier_uniform_(self.I) # uniform분포로 parameter를 초기화
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln) # dim Q, dim K, dim V. mab0에서 K, V가 X임.
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        # mab1에서는 K, V가 H. V는 K를 projection해서 만들기 때문에 달라도 상관없음.
        # 그냥 내부적으로 Inverted Residual Connection처럼 작동한다고 생각

    def forward(self, X):
        # 즉, learned embedding과 입력 사이의 attention
        # 그리고 입력과 mab0의 출력과의 attention. computation graph가 어떻게되나,.. ? 
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    """
    ISAB에서 attention block하나만 있는 버전. 
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
