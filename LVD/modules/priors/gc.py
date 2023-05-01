import torch
import copy


from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class GoalConditioned_Prior(BaseModule):
    """
    """

    def __init__(self, **submodules):
        self.ema_update = None
        super().__init__(submodules)

        self.target_inverse_dynamics = copy.deepcopy(self.inverse_dynamics)
        self.target_dynamics = copy.deepcopy(self.dynamics)

        self.methods = {
            "train" : self.__train__,
            "eval" : self.__eval__,
            "finetune" : self.__finetune__,
        }

    def soft_update(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        # hard update 
        update_moving_average(self.target_inverse_dynamics, self.inverse_dynamics, 1)
        update_moving_average(self.target_dynamics, self.dynamics, 1)

    def forward(self, inputs, mode = "train", *args, **kwargs):
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

        states, G= inputs['states'], inputs['G']  
        N, T, _ = states.shape

        # -------------- State Enc / Dec -------------- #
        if self.joint_learn:
            # jointly learn
            states_repr = self.state_encoder(states.view(N * T, -1))
            states_hat = self.state_decoder(states_repr).view(N, T, -1)
            with torch.no_grad():
                G = self.state_encoder(G)
                hts = states_repr.view(N, T, -1).clone().detach()
                ht, htH = hts[:, 0], hts[:, -1]

        else:
            # pretrained 
            self.state_encoder.eval()
            self.state_decoder.eval()
            with torch.no_grad():
                states_repr = self.state_encoder(states.view(N * T, -1))
                states_hat = self.state_decoder(states_repr).view(N, T, -1)
                G = self.state_encoder(G)
                hts = states_repr.view(N, T, -1)
                ht, htH = hts[:, 0], hts[:, -1]
                
        skill_length = hts.shape[1] - 1 

        # -------------- State-Conditioned Prior -------------- #
        prior = self.prior_policy.dist(ht)

        # -------------- Inverse Dynamics : Skill Learning -------------- #
        inverse_dynamics, inverse_dynamics_detach  = self.inverse_dynamics.dist(state = ht, subgoal = htH, tanh = self.tanh)
        
        # -------------- Dynamics Learning -------------- #
        skill = inverse_dynamics.rsample()
    
        # skill dynamcis for regularization
        dynamics_input = torch.cat((ht, skill), dim = -1)
        D = self.dynamics(dynamics_input)

        # -------------- Subgoal Generator -------------- #
        sg_input = torch.cat((ht,  G), dim = -1)
        subgoal_f = self.subgoal_generator(sg_input)
        invD_sub, _ = self.target_inverse_dynamics.dist(state = ht, subgoal = subgoal_f, tanh = self.tanh)

        skill_sub = invD_sub.rsample()
        
        dynamics_input = torch.cat((ht, skill_sub), dim = -1)
        subgoal_D = self.target_dynamics(dynamics_input)


        result = {
            # states
            "states" : states,
            "states_repr" : states_repr,
            "hts" : hts,
            "states_hat" : states_hat,
            # state conditioned prior
            "prior" : prior,
            # invD
            "invD" : inverse_dynamics,
            "invD_detach" : inverse_dynamics_detach,
            # Ds
            "D" : D,
            "D_target" : htH, 
            # f
            "subgoal_D" : subgoal_D,
            "subgoal_f" : subgoal_f,
            "subgoal_target" : htH,
            "invD_sub" : invD_sub,
            # for metric
            "z_invD" : skill,
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
            ht, G = self.state_encoder(state), self.state_encoder(G)

        # subgoal 
        sg_input = torch.cat((ht,  G), dim = -1)
        subgoal_f = self.subgoal_generator(sg_input)

        inverse_dynamics_hat, _ = self.inverse_dynamics.dist(state = ht, subgoal = subgoal_f,  tanh = self.tanh)

        skill = inverse_dynamics_hat.rsample() 
        dynamics_input = torch.cat((ht,  skill), dim = -1)
        subgoal_D = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 

        result = {
            "inverse_D" : inverse_dynamics_hat,
            "subgoal" : subgoal_D,
            "subgoal_target" : subgoal_f,
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
            ht = self.dynamics(dynamics_input) # subgoal을 얻고, 그걸로 추론한 action을 통해 진짜 subgoal에 도달해야 한다. 
            result['subgoal'] = ht

        return result