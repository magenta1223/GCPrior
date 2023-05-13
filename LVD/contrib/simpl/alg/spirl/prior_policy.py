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
        
        self.prior_policy = copy.deepcopy(prior_policy) # .requires_grad_(False)
        self.prior_state_dim = prior_state_dim
        self.policy_exclude_dim = policy_exclude_dim
        self.min_scale = min_scale


    def act(self, state):
        if self.explore is None:
            raise RuntimeError('explore is not set')

        batch_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
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

    def dist(self, batch_state):
        self.prior_policy.eval()

        # 음.. 특정 차원을 배제하는 과정. task embedding이 concat된 것을 분리하는 과정. 
            # 여기서 바꿔줘야 됨
        state = batch_state[..., :self.prior_state_dim]
        G = batch_state[..., self.prior_state_dim:self.policy_exclude_dim]
        
        inputs = dict(
            states = state,
            G= G
        )

        prior_skill = self.prior_policy(inputs, "eval")['policy_skill']

        if self.prior_policy.prior_policy.tanh:
            prior_locs = prior_skill._normal.base_dist.loc
            prior_pre_scales = prior_skill._normal.base_dist.scale
        else:
            prior_locs = prior_skill.base_dist.loc
            prior_pre_scales = prior_skill.base_dist.scale


        # distributions from prior state

        # distributions from policy state
        res_locs, res_pre_scales = self.mlp(batch_state).chunk(2, dim=-1)

        # 혼합
        locs = prior_locs + res_locs
        scale = self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
        
        if self.prior_policy.prior_policy.tanh:
            return get_dist(locs, scale= scale, tanh = True)
        
        else:
            dist = torch_dist.Normal(
                prior_locs + res_locs,
                self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
            )
            return torch_dist.Independent(dist, 1)