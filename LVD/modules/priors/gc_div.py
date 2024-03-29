import torch
import copy

import math

from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class GoalConditioned_Diversity_Prior(BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_inverse_dynamics = copy.deepcopy(self.inverse_dynamics)
        self.target_dynamics = copy.deepcopy(self.dynamics)
        self.target_flat_dynamics = copy.deepcopy(self.flat_dynamics)

        self.methods = {
            "train" : self.__train__,
            "eval" : self.__eval__,
            "rollout" : self.__rollout__,
            "rollout2" : self.__rollout2__,
            "finetune" : self.__finetune__,
            "prior" : self.__prior__
        }

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # hard update 
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)
        update_moving_average(self.target_flat_dynamics, self.flat_dynamics, 1)

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

        states, G = inputs['states'], inputs['G']  
        N, T, _ = states.shape
        skill_length = T - 1 

        # -------------- State Enc / Dec -------------- #
        if self.joint_learn:
            # jointly learn
            states_repr = self.state_encoder(states.view(N * T, -1))

            state_emb = states_repr.view(N, T, -1)[:, 0]
            states_fixed = torch.randn(512, *state_emb.shape[1:]).cuda()
            states_hat = self.state_decoder(states_repr).view(N, T, -1)

            with torch.no_grad():
                # G = self.state_encoder(G)
                hts = states_repr.view(N, T, -1).clone().detach()
                start, subgoal = hts[:, 0], hts[:, -1]

        else:
            # pretrained 
            self.state_encoder.eval()
            self.state_decoder.eval()
            with torch.no_grad():
                states_repr = self.state_encoder(states.view(N * T, -1))
                states_hat = self.state_decoder(states_repr).view(N, T, -1)

                hts = states_repr.view(N, T, -1)
                start, subgoal = hts[:, 0], hts[:, -1]
                # hts_target = self.target_state_encoder(states.view(N * T, -1)).view(N, T, -1)
                # subgoal_target = hts_target[:, -1]
                
                state_emb = None
                states_fixed = None


        # -------------- State-Conditioned Prior -------------- #
        prior, prior_detach = self.prior_policy.dist(start, detached = True)

        # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = start, subgoal = subgoal, tanh = self.tanh)
        
        # -------------- Dynamics Learning -------------- #
        # skill = inverse_dynamics.rsample()
        skill = inputs['skill'].clone().detach()

        # flat dynamcis for rollout
        flat_dynamics_input = torch.cat((hts[:, :-1], skill.unsqueeze(1).repeat(1, skill_length, 1)), dim=-1)
        diff_flat_D = self.flat_dynamics(flat_dynamics_input)
        flat_D =  hts[:,:-1].clone() + diff_flat_D

        # skill dynamcis for regularization
        dynamics_input = torch.cat((start, skill), dim = -1)
        diff_D = self.dynamics(dynamics_input)
        D = start.clone() + diff_D



        # -------------- Subgoal Generator -------------- #
        sg_input = torch.cat((start,  G), dim = -1)
        diff_subgoal_f = self.subgoal_generator(sg_input)
        subgoal_f = diff_subgoal_f + start.clone()
        invD_sub1, _ = self.target_inverse_dynamics.dist(state = start, subgoal = subgoal_f, tanh = self.tanh)

        skill_sub1 = invD_sub1.rsample()

        dynamics_input = torch.cat((start, skill_sub1), dim = -1)
        diff_subgoal_D = self.target_dynamics(dynamics_input)
        subgoal_D = start.clone() + diff_subgoal_D 



        # -------------- Rollout for metric -------------- #
        _ht = start.clone().detach()
        # dense execution with loop (for metric)
        with torch.no_grad():
            for _ in range(skill_length):
                flat_dynamics_input = torch.cat((_ht, skill), dim=-1)
                diff = self.target_flat_dynamics(flat_dynamics_input) 
                _ht = _ht + diff
            invD_rollout_main, _ = self.target_inverse_dynamics.dist(state = start, subgoal = _ht, tanh = self.tanh)

            subgoal_recon = self.state_decoder(subgoal)
            subgoal_recon_D = self.state_decoder(D)
            subgoal_recon_D_f = self.state_decoder(subgoal_D)
            subgoal_recon_f = self.state_decoder(subgoal_f)

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
            # invD
            "invD" : inverse_dynamics,
            "invD_detach" : inverse_dynamics_detach,
            # Ds

            "D" : D,
            "flat_D" : flat_D,
            "D_target" : subgoal, 
            "flat_D_target" : hts[:, 1:],

            # f
            # "subgoal_D" : subgoal_D.clone().detach(),
            "subgoal_D" : subgoal_D,
            "subgoal_f" : subgoal_f,
            # "subgoal_target" : subgoal,
            "subgoal_target" : subgoal,

            "invD_sub" : invD_sub1,
            "z_sub" : skill_sub1,
            # "invD_sub2" : invD_sub2,



            # for metric
            "z_invD" : skill,
            "invD_rollout_main" : invD_rollout_main,
            "subgoal_rollout" : _ht,
            "subgoal_recon_D" : subgoal_recon_D,
            "subgoal_recon_f" : subgoal_recon_f,
            "subgoal_recon_D_f" : subgoal_recon_D_f,
            "subgoal_recon"  : subgoal_recon

            # "D_metric" : D,
            # "D_target_metric" : subgoal, 
            # "flat_D" : flat_D,
            # "flat_D_target" : hts[:, 1:],


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
        self.state_encoder.eval()
        state, G = inputs['states'], inputs['G']       

        with torch.no_grad():
            # ht, G = self.state_encoder(state), self.state_encoder(G)
            ht = self.state_encoder(state)

        # subgoal 
        sg_input = torch.cat((ht,  G), dim = -1)
        subgoal_f = self.subgoal_generator(sg_input)

        inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f + ht,  tanh = self.tanh)
        # inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)


        skill = inverse_dynamics_hat.rsample() 
        dynamics_input = torch.cat((ht,  skill), dim = -1)
        # subgoal_D = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
        diff = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 


        result = {
            "inverse_D" : inverse_dynamics_hat,
            "subgoal" : diff + ht,
            "subgoal_target" : subgoal_f + ht,
        }


        return result
    

    def __rollout__(self, inputs):
        self.state_encoder.eval()
        self.state_decoder.eval()
        self.prior_policy.eval()
        self.inverse_dynamics.eval()
        self.flat_dynamics.eval()
        self.dynamics.eval()

        states, skill = inputs['states'], inputs['actions']
        N, T, _ = states.shape
        skill_length = T - 1

        hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)            
        hts_rollout = []

        # start ! 
        c = random.sample(range(1, skill_length - 1), 1)[0]
        _ht = hts[:, c].clone()
        skill_sampled_orig = self.prior_policy.dist(_ht).sample()


        skill_sampled = skill_sampled_orig.clone()
        # 1 skill
        for _ in range(c, skill_length):
            # execute skill on latent space and append to the original sub-trajectory 
            dynamics_input = torch.cat((_ht, skill_sampled), dim=-1)
            diff = self.flat_dynamics(dynamics_input) 
            _ht = _ht + diff
            hts_rollout.append(_ht)
        
        invD_rollout, _ = self.inverse_dynamics.dist(state = hts[:, 0], subgoal = _ht,  tanh = self.tanh)
        invD_GT, _ = self.inverse_dynamics.dist(state = hts[:, 0], subgoal = hts[:, -1],  tanh = self.tanh)


        # for f learning, execute 4 skill more
        # (plan_H - skill_length) // skill_length
        for _ in range((self.plan_H - skill_length) // skill_length):
        # for _ in range(9):
            skill = self.prior_policy.dist(_ht).sample()
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            diff = self.dynamics(dynamics_input) 
            _ht = _ht + diff
            hts_rollout.append(_ht)

            
        hts_rollout = torch.stack(hts_rollout, dim = 1)
        N, T, _ = hts_rollout.shape
        states_rollout = self.state_decoder( hts_rollout.view(N * T, -1) ).view(N, T, -1)

        result =  {
            "c" : c,
            "states_rollout" : states_rollout,
            "skill_sampled" : skill_sampled_orig,
            "invD_rollout" : invD_rollout,
            "invD_GT" : invD_GT
        }
        return result 


    def __rollout2__(self, inputs):

        self.state_encoder.eval()
        self.state_decoder.eval()
        self.prior_policy.eval()
        self.inverse_dynamics.eval()
        self.flat_dynamics.eval()
        self.dynamics.eval()

        states = inputs['states']
        skill_length = states.shape[1] - 1
        
        N, T, _ = states.shape
        hts = self.state_encoder(states.view(N * T, -1)).view(N, T, -1)
        start, end = hts[:, 0], hts[:, -1]
    
        # get primitive skill 
        invD_GT, _ = self.inverse_dynamics.dist(state = start, subgoal = end,  tanh = self.tanh)
        skill = invD_GT.sample()
        rollout_lengnth = skill_length * 5

        hts_rollout, zs_rollout = [], []
        
        # start ! 
        _ht = start.clone().detach()

        # flat dynamics for stitching
        for i in range(skill_length * 2):
            # state, skill 추가 
            hts_rollout.append(_ht)
            zs_rollout.append(skill)
            
            # execute skill on latent space 
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            diff = self.flat_dynamics(dynamics_input) 
            _ht = _ht + diff

            # sample next skill
            if (i + 1) % self.sample_interval == 0:
                skill = self.prior_policy.dist(_ht).sample()

        # skill dynamcis for long horizon planning with reduced error accumulation 
        for _ in range((self.plan_H - 2 * skill_length) // skill_length):
        # for i in range(8):
            hts_rollout.append(_ht)
            zs_rollout.append(skill)
            
            # execute skill on latent space 
            dynamics_input = torch.cat((_ht, skill), dim=-1)
            diff = self.dynamics(dynamics_input) 
            _ht = _ht + diff
            skill = self.prior_policy.dist(_ht).sample()

        hts_rollout.append(_ht)

        hts_rollout = torch.stack(hts_rollout, dim = 1)
        zs_rollout = torch.stack(zs_rollout, dim = 1)

        N, T, _ = hts_rollout.shape
        states_rollout = self.state_decoder( hts_rollout.view(N * T, -1) ).view(N, T, -1)


        # 이게.. subgoal 에러누적이 거의 없어서 다른것만 필터링해서 넣어도 될 것 같거든?? 

        result =  {
            "states_rollout" : states_rollout,
            "skills" : zs_rollout,
            "invD_GT" : invD_GT
        }


        return result


    def __finetune__(self, inputs):
        """
        Finetune inverse dynamics and dynamics with the data collected in online.
        """

        states, G, next_states = inputs['states'], inputs['G'], inputs['next_states'] 

        self.state_encoder.eval()

        with torch.no_grad():
            ht = self.state_encoder(states)
            htH = self.state_encoder(next_states)

        # finetune invD, D 
        invD, _ = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.tanh)
        z = invD.rsample()
        
        
        result = {
            "inverse_D" : invD,
            "subgoal_target" : htH
        }


        if self.dynamics is not None:
            dynamics_input = torch.cat((ht,  z), dim = -1)
            diff = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
            result['subgoal'] = ht + diff

        return result
    
    
    def __prior__(self, inputs):
        states = inputs['states']

        with torch.no_grad():
            ht = self.state_encoder(states)
            prior = self.prior_policy.dist(ht)

        return {
            "prior" : prior
        }