import torch
import copy

import math

from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class Skimo_Prior(BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        self.step = 0

        self.methods = {
            "train" : self.__train__,
            "eval" : self.__eval__,
            "rollout" : self.__rollout__,
            "finetune" : self.__finetune__,
            "prior" : self.__prior__
        }

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # hard update 
        if self.step % 2 == 0:
            update_moving_average(self.target_state_encoder, self.state_encoder)

    def forward(self, inputs, mode = "train", *args, **kwargs):
        # print(self.methods.keys())
        return self.methods[mode](inputs, *args, **kwargs)
    
    
    def __train__(self, inputs):
        """
        Jointly Optimize 
        - State Encoder / Decoder
        - Inverse Dynamcis
        - Dynamics
        - Subgoal Generator
        - Skill Encoder / Decoder (at upper level)
        """
        if self.training:
            self.step += 1

        states, skill, G = inputs['states'], inputs['skill'], inputs['G']
        N, T, _ = states.shape
        skill_length = T - 1 

        # -------------- State Enc / Dec -------------- #
        # jointly learn
        states_repr = self.state_encoder(states.view(N * T, -1))

        state_emb = states_repr.view(N, T, -1)[:, 0]
        states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda()
        states_hat = self.state_decoder(states_repr).view(N, T, -1)

            # G = self.state_encoder(G)
        hts = states_repr.view(N, T, -1).clone().detach()
        ht = hts[:, 0]

        with torch.no_grad():
            htH = self.target_state_encoder(states[:, -1])

        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.prior_policy.dist(states[:, 0], detached = True)

        # ------------------ Skill Dynamics ------------------- #
        dynamics_input = torch.cat((ht, skill), dim = -1)
        D = self.dynamics(dynamics_input)



        # -------------- High-level policy -------------- #
        if self.gc:
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht.clone().detach(), G), dim = -1))
        else:
            policy_skill = self.highlevel_policy.dist(torch.cat(ht.clone().detach(), dim = -1))


        # -------------- Rollout for metric -------------- #
        # dense execution with loop (for metric)
        with torch.no_grad():
            subgoal_recon_D = self.state_decoder(D)


        result = {
            # states
            "states" : states,
            "states_repr" : state_emb,
            "hts" : hts,
            "states_hat" : states_hat,
            "states_fixed_dist" : states_fixed,
            # state conditioned prior
            "prior" : prior,
            "prior_detach" : prior_detach,

            # Ds
            "D" : D,
            "D_target" : htH, 

            # highlevel policy
            "policy_skill" : policy_skill,

            # for metric
            "z_invD" : skill,
            "subgoal_recon_D" : subgoal_recon_D,


        }


        return result
    
    def __eval__(self, inputs):
        """
        data를 collect할 때 / target Q 계산할 때는 subgoal을 generate하고
        Policy Learning / skill consistency tuning 할 때는 given subgoal을 쓰자. 
        이유 : 원래는 consistency 빼고는 다 generate해서 써야 함. 그러나 
        1) 잘 안되고
        2) 어차피 Q-function은 value estimation임. (s, a)에 대한 가치 추정. action이 구리면 구린대로 가치추정하면 그만인거
        3) 그러나 policy learning은 맞는 state로 학습해야 함. 
        evaluation + subgoal finetune
        """

        state, G = inputs['states'], inputs['G']
        with torch.no_grad():
            ht = self.state_encoder(state)
        
        if self.gc:
            policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1))
        else:
            policy_skill =  self.highlevel_policy.dist(ht)


        return dict(
            policy_skill = policy_skill
        )
    
    @torch.no_grad()
    def __rollout__(self, inputs):
        ht, G = inputs['states'], inputs['G']
        planning_horizon = inputs['planning_horizon']

        # skill sample from high-level policy 
        # 근데 tanh에서 planning하면 numerical unstabilty 때문에 .. 
        # ht = self.state_encoder(states)
        skills = []
        for i in range(planning_horizon):
            if self.gc:
                policy_skill =  self.highlevel_policy.dist(torch.cat((ht, G), dim = -1)).sample()
            else:
                policy_skill =  self.highlevel_policy.dist(ht).sample()
            dynamics_input = torch.cat((ht, policy_skill), dim = -1)
            ht = self.dynamics(dynamics_input)
            skills.append(policy_skill)
        
        return dict(
            policy_skills = torch.stack(skills, dim=1)
        )


    def __finetune__(self, inputs):
        """
        Finetune state encoder, dynamics
        """

        states, next_states, skill = inputs['states'], inputs['next_states'], inputs['actions']

        ht = self.state_encoder(states)
        htH = self.target_state_encoder(next_states)
        htH_hat = self.dynamics(torch.cat((ht, skill), dim = -1))

        
        result = {
            "ht" : ht,
            "subgoal_target" : htH,
            "subgoal" : htH_hat
        }


        return result
    
    
    def __prior__(self, inputs):
        states = inputs['states']

        with torch.no_grad():
            prior = self.prior_policy.dist(states)

        return {
            "prior" : prior
        }