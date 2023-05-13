import copy

import torch
import torch.distributions as torch_dist
import torch.nn.functional as F

from ...nn import MLP
from ...rl.policy import StochasticNNPolicy

import sys
from .....utils import get_dist

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)
    

class PriorResidualNormalMLPPolicy(StochasticNNPolicy):
    def __init__(self, prior_policy, state_dim, action_dim, hidden_dim, n_hidden,
                 prior_state_dim=None, policy_exclude_dim=None, activation='relu', min_scale=0.001):
        super().__init__()
        self.action_dim = action_dim
        self.mlp = MLP([state_dim] + [hidden_dim]*n_hidden + [2*action_dim], activation)
        
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.prior_state_dim = prior_state_dim
        self.policy_exclude_dim = policy_exclude_dim
        self.min_scale = min_scale


    def dist(self, batch_state):
        self.prior_policy.eval() # 이게 파라미터만 있을텐데.. 그냥 불러오기만 해도 architecture가 되던가.. ? 
        
        batch_prior_state = batch_state
        batch_policy_state = batch_state

        # 음.. 특정 차원을 배제하는 과정. task embedding이 concat된 것을 분리하는 과정. 
        if self.prior_state_dim is not None:
            # 여기서 바꿔줘야 됨
            batch_prior_state = batch_state[..., :self.prior_state_dim]
        if self.policy_exclude_dim is not None:
            batch_policy_state = batch_state[..., self.policy_exclude_dim:]


        # distributions from prior state
        try:
            prior_locs, prior_log_scales = self.prior_policy.dist_param(batch_prior_state)
        except:
            prior_locs, prior_log_scales = self.prior_policy(batch_prior_state).chunk(2, -1)



        prior_pre_scales = inverse_softplus(prior_log_scales.exp())
        
        # distributions from policy state
        res_locs, res_pre_scales = self.mlp(batch_policy_state).chunk(2, dim=-1)

        # 혼합
        locs = prior_locs + res_locs
        scale = self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
        
        if self.prior_policy.tanh:

            return get_dist(locs, scale= scale, tanh = True)

        
        else:
            dist = torch_dist.Normal(
                prior_locs + res_locs,
                self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
            )
            return torch_dist.Independent(dist, 1)