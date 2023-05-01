import torch
import copy


from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class StateConditioned_Diversity_Prior(BaseModule):
    """
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

        self.methods = {
            "train" : self.__train__,
            "eval" : self.__eval__,
            "rollout" : self.__rollout__
        }

    def forward(self, inputs, mode = "train", *args, **kwargs):
        return self.methods[mode](inputs, *args, **kwargs)

    def __train__(self, inputs):
        """
        State only Conditioned Prior
        inputs : dictionary
            -  states 
        return: state conditioned prior, and detached version for metric
        """
        states = inputs['states']

        if hasattr(self, "state_encoder"):
            with torch.no_grad():
                states = self.state_encoder(states[:, 0])

        prior, prior_detach = self.prior_policy.dist(states, detached = True)

        return dict(
            prior = prior,
            prior_detach = prior_detach
        )

    def __eval__(self, inputs):
        """
        inputs : dictionary
            -  states 
        return: state conditioned prior, and detached version for metric
        """
        states = inputs['states']
        states = states.unsqueeze(1)
        
        if hasattr(self, "state_encoder"):
            with torch.no_grad():
                states = self.state_encoder(states[:, 0])
            
        prior, prior_detach = self.prior_policy.dist(states, detached = True)

        return dict(
            prior = prior,
            prior_detach = prior_detach
        )
    
    def __rollout__(self, inputs):
        self.state_encoder.eval()

        states = inputs['states']
        skill_length = states.shape[1] - 1
        
        N, T, _ = states.shape

        with torch.no_grad():
            hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)
            start, end = hts[:, 0], hts[:, -1]
            
            # GT skill distribution
            prior_dist = self.prior_policy.dist(states)

            # sample skill
            skill = prior_dist.sample()

            # sample interval 마다 skill을 sampling하면서 flat dynamics를 통해 rollout 
            hts_rollout, skills, subgoal = self.rollout(
                start = start,
                skill_length = skill_length,
                skill = skill,
                sample_interval= self.sample_interval,
                buffer = True
            )
        
        result =  {
            "rollout_states" : hts_rollout,
            "rollout_skills" : skills,
            "subgoal_GT" : end,
            "subgoal_rollout" : subgoal
        }

        return result 

    def rollout(self, start, skill_length = 10, skill = None, sample_interval = None, buffer = False):

        hts = []
        zs =  []

        if sample_interval is None:
            sample_interval = skill_length

        if buffer:
            rollout_lengnth = skill_length * 5
        else:
            rollout_lengnth = skill_length
        
        
        # start ! 
        _ht = start.clone().detach()

        for i in range(rollout_lengnth):
            # state, skill 추가 
            hts.append(_ht)
            zs.append(skill)
            
            # execute skill on latent space 
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            _ht = self.flat_dynamics(dynamics_input) 

            # sample next skill
            if 0 < i < skill_length and i % sample_interval == 0:
                skill = self.prior_policy.dist(_ht).sample()
            elif i % skill_length == 0:
                skill = self.prior_policy.dist(_ht).sample()
            
        hts.append(_ht)

        hts = torch.stack(hts, dim = 1)
        zs = torch.stack(zs, dim = 1)

        return hts, zs, hts[:, skill_length]
        

