import numpy as np
import torch
import torch.nn as nn
from ..modules.base import SequentialBuilder, ContextPolicyMixin
from ..utils import *

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