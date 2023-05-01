import torch.nn as nn

from .set_transformer import ISAB, PMA, SAB


class MLP(nn.Module):
    activation_classes = {
        'relu': nn.ReLU,
    }
    def __init__(self, dims, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = dims[0]
        for dim in dims[1:-1]:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation_classes[activation]())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_attention, n_mlp_layer,
                 n_ind=32, n_head=4, ln=False, activation='relu'):
        super().__init__()

        # leanred embedding에 대해 입력을 attention시키는 모듈
        attention_layers =  [ISAB(in_dim, hidden_dim, n_head, n_ind, ln=ln)]
        # 을 계속 추가. 
        attention_layers += [
            ISAB(hidden_dim, hidden_dim, n_head, n_ind, ln=ln)
            for _ in range(n_attention-1)
        ]
        self.attention = nn.Sequential(*attention_layers)
        # ISAB하고 같은데 block이 하나만 있는거. 근데 이게 어떻게 pooling임?? 
        self.pool = PMA(hidden_dim, n_head, 1, ln=ln)
        # 일반적인 MLP
        self.mlp = MLP([hidden_dim]*n_mlp_layer + [out_dim], activation=activation)
        # 일반적인 transformer는 입력간의 attention을 하는데, 여기서는 내부의 parameter에 대해 attention을 함. 
        # 

    def forward(self, batch_set_x):
        return self.mlp(self.pool(self.attention(batch_set_x)).squeeze(1))
