import numpy as np
import torch
import torch.nn as nn
from ..modules.base import SequentialBuilder, ContextPolicyMixin
from ..utils import *
from ..configs.build.configs import Linear_Config


class MultiModalEncoder(nn.Module):
    """
    1. maze의 경우 multimodal 입력이 들어옴. 이를 기존 single modal 입력과 같은 방식으로처리하기 위함. 
    2. 입력 state를 기준에 따라 분리
    3. 별도의 encoder에 forward
    4. concat
    5. 이 역과정을 decoder에서 수행. 
    """
    
    def __init__(self, config):
        super(MultiModalEncoder, self).__init__()
        
        self.state_dim = 4
        visual_config = {**config}
        visual_config['in_feature'] = 1024 # 32 x 32
        self.vector_enc = SequentialBuilder(Linear_Config(config))
        self.visual_enc = SequentialBuilder(Linear_Config(visual_config))

    def forward(self, x):

        vec_emb = self.vector_enc(x[:, :self.state_dim])
        visual_emb = self.visual_enc(x[:, self.state_dim:])
        
        return torch.cat(( vec_emb, visual_emb), dim = -1)


class MultiModalDecoder(nn.Module):
    def __init__(self, config):
        super(MultiModalDecoder, self).__init__()

        visual_config = {**config}
        visual_config['out_dim'] = 1024 # 32 x 32
        self.vector_dec = SequentialBuilder(Linear_Config(config))
        self.visual_dec = SequentialBuilder(Linear_Config(visual_config))

    def forward(self, x, rollout = False):
        vec_emb, visual_emb = x.chunk(2, -1)
        vec_pred = self.vector_dec(vec_emb)
        visual_pred = torch.sigmoid(self.visual_dec(visual_emb)) # binary image
        
        if rollout:
            visual_pred = torch.where(visual_pred >= 0.5, 1, 0)

        return torch.cat(( vec_pred, visual_pred), dim = -1)







class InverseDynamicsMLP(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None

    # for compatibility with simpl
    def dist_param(self, state):
        only_states = state[:, : self.in_feature]
        out = self(only_states)
        mu, log_std = out.chunk(2, dim = 1)
        return mu, log_std.clamp(-10, 2)

    def dist(self, state, subgoal, tanh = False):        
        id_inputs = torch.cat((state, subgoal), dim = -1)
        dist_params = self(id_inputs)
        dist, dist_detached = get_dist(dist_params, tanh= tanh), get_dist(dist_params.clone().detach(), tanh= tanh)
        
        return dist, dist_detached

class DecoderNetwork(ContextPolicyMixin, SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None
        self.log_sigma = nn.Parameter(-50*torch.ones(self.out_dim)) # for stochastic sampling

    # from simpl. for compatibility with simpl
    def dist(self, batch_state_z):
        if self.state_dim is not None:
            # self.state_dim = 30
            batch_state_z = torch.cat([
                batch_state_z[..., :self.state_dim],
                batch_state_z[..., -self.z_dim:]
            ], dim=-1)

        loc = self(batch_state_z)
        scale = self.log_sigma[None, :].expand(len(loc), -1)
        dist = get_dist(loc, scale)
        return dist

    def act(self, state, ):
        if self.z is None:
            raise RuntimeError('z is not set')
        state = np.concatenate([state, self.z], axis=0)

        if self.explore is None:
            raise RuntimeError('explore is not set')
        
        batch_state = prep_state(state, self.device)


        with torch.no_grad():
            training = self.training
            self.eval()
            dist = self.dist(batch_state)
            self.train(training)

        if self.explore is True:
            batch_action = dist.sample()
        else:
            batch_action = dist.mean
        return batch_action.squeeze(0).cpu().numpy()