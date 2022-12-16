import numpy as np

import torch
import torch.nn as nn
import torch.distributions as torch_dist

import copy
from easydict import EasyDict as edict

from proposed.modules.base import SequentialBuilder, ContextPolicyMixin, BaseModule
from proposed.utils import *
from proposed.contrib.momentum_encode import update_moving_average




class PriorNetwork(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None

    # for compatibility with simpl
    def dist_param(self, state):
        only_states = state[:, : self.in_feature]
        out = self(only_states)
        mu, log_std = out.chunk(2, dim = 1)
        return mu, log_std.clamp(-10, 2)

    def dist(self, batch_state, detached = False): 
        # print(batch_state.shape, self.in_feature)   
        dist1 = get_dist(self(batch_state[:, : self.in_feature]))
        dist2 = get_dist(self(batch_state[:, : self.in_feature]), detached= True)

        return dist1, dist2



class EncoderNetwork(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None

    def dist_param(self, state):
        # only_states = state[:, : self.in_feature]
        out = self(state.unsqueeze(1)) # AR 생성 필요
        mu, log_std = out.chunk(2, dim = 1)
        return mu, log_std.clamp(-10, 2)



class DecoderNetwork(ContextPolicyMixin, SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None
        self.log_sigma = nn.Parameter(-50*torch.ones(self.out_dim)) # for stochastic sampling

    # from simpl. for compatibility with simpl
    def dist(self, batch_state_z):
        if self.state_dim is not None:
            batch_state_z = torch.cat([
                batch_state_z[..., :self.state_dim],
                batch_state_z[..., -self.z_dim:]
            ], dim=-1)

        loc = self(batch_state_z)
        scale = self.log_sigma[None, :].expand(len(loc), -1)
        dist = get_dist(loc, scale)
        return dist

    def act(self, state):
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




class PriorWrapper(BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        # self.prior_policy = prior_policy
        # for name, module in submodules.items():
        #     setattr(self, name, module)
        self.ema_update = None
        
        super().__init__(submodules)
        
        self.forward_dict = {
            "default" : self.default, # state only conditioned
            "gcid" : self.GCID, # inverse dynamics & dynamics prior
            # TODO : implement 
        }



        if self.ema_update:
            self.target_state_encoder = copy.deepcopy(self.state_encoder)
                    
        print("MODE", self.mode)


        # copy
        if self.mode in ['gcid']:
            self.copy_weights()


    def forward(self, *args):
        return self.forward_dict[self.mode](*args)

    def ma_state_enc(self):
        """
        Exponentially moving averaging the parameters of state encoder 
        """
        update_moving_average(self.target_state_encoder, self.state_encoder)
    
    def copy_weights(self):
        """
        Hard-update the prior policy so that the subgoal generator learns directly from the prior without update
        """

        self.prior_policy_eval.load_state_dict(self.prior_policy.state_dict())
        self.prior_policy_eval.eval()

        
        for p in self.prior_policy_eval.parameters():
            p.requires_grad_(False)

    def default(self, inputs, *args):
        """
        State only Conditioned Prior
        inputs : instance of EasyDict
            -  states 
        return: state conditioned prior, and detached version for metric
        """

        prior, prior_detach = self.prior_policy.dist(inputs.states, detached = True)
        return edict(
            prior = prior,
            prior_detach = prior_detach
        )
    

    def __dhsid__(self, states, h_tH = None):
        """
        Distributional High States & Inverse Dynamics 
        states : states
        h_tH   : hidden states of H-step later. It will be given when inference or RL adaptation. 

        return :
            # state reconstruction
                states = states,
                s_hat = state_hat, 

            # state regularization
                - hsd : distribution of high state
                - hs  : high states sampled from hsd 
                - hsd_target : fixed distribution to guide high state by beta-VAE
            
            # subgoal generation
                - ht : sampled high state at time step t 
                - h_tH_detach : detached & sampled high state at time step t + H 

            # prior distributions 
                - prior : inverse dynamics prior,
                - prior_detach : detached prior
        """

        if h_tH is None: # for prior training
            N, T, D = states.shape
            
            # distribution version
            _h = self.state_encoder(states.view(N * T, -1)) # N, T, 20

            hsd = get_dist(_h) # high state dist N * T, -1
            hs = hsd.rsample().view(N, T, -1)  # high state N, T, 10
            hsd_target = get_fixed_dist(torch.randn(N * T, 20).cuda())

            state_hat = self.state_decoder(hs.view(N * T, -1)).view(N, T, -1)

            # inverse dynamics 
            ht, h_tH = hs[ : , 0], hs[ : , -1]
            prior_input = torch.cat((ht, h_tH), -1)
            prior, prior_detach = self.prior_policy.dist(prior_input, detached = True)


            # 보조 리워드 
            # 원래는 중간골은 task specific 
            # final goal이 taks인데
            # final goal이 중간 goal과 관련이 없다...  ?
            
            # GC skill은 학습해서 만들면 됨
            # goal을 어떻게 생성?
            # final
            return edict(
                # state reconstruction
                    states = states,
                    s_hat = state_hat, 

                # state regularization
                    hsd = hsd, 
                    hs = hs, 
                    hsd_target = hsd_target, 

                # for subgoal generating
                    ht = ht,
                    h_tH_detach = h_tH.clone().detach(),

                # prior distributions 
                    prior = prior,
                    prior_detach = prior_detach
            )
            
        else: # for RL

            if self.state_encdoder.training:
                ht = self.state_encoder.dist(states).rsample()
            else:
                with torch.no_grad():
                    ht = self.state_encoder.dist(states).sample()# single time step
            prior_input = torch.cat((ht, h_tH), -1)
            prior, prior_detach = self.prior_policy.dist(prior_input)  
            return edict(
                prior = prior
            )


    def __hsid__(self, states, result = None):
        """
        High States & Inverse Dynamics 
        states : states
        h_tH   : hidden states of H-step later. It will be given when inference or RL adaptation. 

        return :
            # state reconstruction
                states = states,
                s_hat = state_hat, 
            
            # subgoal generation
                - ht : high state at time step t 
                - h_tH_detach : detached  high state at time step t + H 

            # prior distributions 
                - prior : inverse dynamics prior,
                - prior_detach : detached prior
        """


        if result is None:
            N, T, D = states.shape

            hs = self.state_encoder(states.view(N * T, -1)) # N, T, 20
            state_hat = self.state_decoder(hs).view(N, T, -1)
            
            hs_reshaped = hs.view(N, T, -1)
            # inverse dynamics 
            ht, h_tH = hs_reshaped[ : , 0], hs_reshaped[ : , -1]
            prior_input = torch.cat((ht, h_tH), -1)
            prior, prior_detach = self.prior_policy.dist(prior_input, detached = True)

            return edict(
                # state reconstruction
                    states = states,
                    s_hat = state_hat, 

                # subgoal generating
                    ht = ht,
                    h_tH_detach = h_tH.clone().detach(),
                
                # prior distributions 
                    prior = prior,
                    prior_detach = prior_detach,      

                # for metric
                    hs = hs, # high_states
                    hs_reshaped = hs_reshaped.clone().detach()
            )          

        else :
            # N, T, D = states.shape
            # states : N x D

            # ht = self.state_encoder(states) # N, T, 20
            
            # inverse dynamics 
            prior_input = torch.cat((result.ht, result.h_tH_hat), -1)
            result['prior'], result['prior_detach'] = self.prior_policy.dist(prior_input, detached = True)

            return result

    def __dsg__(self, inputs, result = None):
        """
        Goal-Conditioned Inverse Dynamics 
        """
        states, G = inputs.states, inputs.G
        
        if result is not None:
            # transition probability 
            with torch.no_grad(): 
                h_star = self.target_state_encoder.dist(G).sample() 
                _, result['hsd_tH'] = self.target_state_encoder.dist(states[:,-1], detached = True)
                _, ht_detach = self.target_state_encoder.dist(states[:, 0], detached = True)
                result['ht_detach'] = ht_detach.sample()

            sg_input = torch.cat((result.ht, h_star), dim = -1) # ht, ht_detach
            result['h_tH_hat_dist'] = self.subgoal_generator.dist(sg_input)
            
            if self.direct:
                self.prior_policy_eval.eval()
                prior_input = torch.cat((result.ht_detach, result.h_tH_hat_dist.rsample()), -1)
                result['prior_hat'], _ = self.prior_policy_eval.dist(prior_input)
            else:            
                with torch.no_grad():
                    self.prior_policy_eval.eval()
                    prior_input = torch.cat((result.ht_detach, result.h_tH_hat_dist.sample()), -1)
                    _, result['prior_hat'] = self.prior_policy_eval.dist(prior_input)
            return result
        
        else: # inference
            with torch.no_grad():
                ht = self.state_encoder.dist(states).sample()# single time step
                h_star = self.target_state_encoder.dist(G).sample()
                h_tH_hat = self._generate_subgoal(torch.cat((ht, h_star), dim = -1))
                prior_input = torch.cat((ht, h_tH_hat), -1)
                prior, prior_detach = self.prior_policy.dist(prior_input)  

            result = edict(
                prior = prior,
                ht = ht,
                h_tH_hat = h_tH_hat
            )

            return result 

    def __sg__(self, inputs, result = None):
        states, G = inputs.states, inputs.G
        
        if result is not None:
            # transition probability 
            with torch.no_grad(): 
                h_star = self.target_state_encoder(G)
                result['ht_detach'] = self.target_state_encoder(states[:, 0])
                result['hsd_tH'] = self.target_state_encoder(states[:,-1])


            sg_input = torch.cat((result.ht, h_star), dim = -1) # ht, ht_detach
            result['h_tH_hat'] = self.subgoal_generator(sg_input)
            
            
            if self.direct:
                # for learning 
                self.prior_policy_eval.eval()
                prior_input = torch.cat((result.ht_detach, result.h_tH_hat), -1)
                result['prior_hat'], _ = self.prior_policy_eval.dist(prior_input)
            else:            
                # for metric 
                with torch.no_grad():
                    self.prior_policy_eval.eval()
                    prior_input = torch.cat((result.ht_detach, result.h_tH_hat), -1)
                    _, result['prior_hat'] = self.prior_policy_eval.dist(prior_input)
                        
        else:
            with torch.no_grad():
                ht = self.state_encoder(states)# single time step
                h_star = self.target_state_encoder(G)
                h_tH_hat = self.subgoal_generator(torch.cat((ht, h_star), dim = -1))

            result = edict(
                # prior = prior,
                ht = ht,
                h_tH_hat = h_tH_hat
            )

        return result 


    def GCID(self, inputs, mode = "train"):
        """
        모드가 3개 필요. train, finetune, eval
        
        """
        assert mode in ['train', 'finetune', 'eval'], "Invalid mode. Valid choices are 'train', 'finetune', and 'eval'."
        self.target_state_encoder.eval()
        
        if mode == "train":
            if self.distributional:
                result = self.__dhsid__(inputs.states)
                result = self.__dsg__(inputs, result)
            else:
                result = self.__hsid__(inputs.states)
                result = self.__sg__(inputs, result)
            
        elif mode == "finetune":
            if self.distributional:
                result = self.__dfinetune__(inputs)
            else:
                result = self.__finetune__(inputs)
            
        else:        
            self.eval()
            if self.distributional:
                result = self.__dsg__(inputs, None)
                result = self.__dhsid__(inputs.states, result.h_tH_hat.sample())
            else:
                result = self.__sg__(inputs, None)        
                result = self.__hsid__(inputs.states, result)
        

        return result


   

    def __dfinetune__(self, inputs):
        """
        Fine-tuning method for distributional mode 
        TODO
        1) state encoder, subgoal_generator train mode
        2) calculate ht, hstar, h_tH
        3) calculate h_tH_hat
        4) NLL loss between(h_tH, h_tH_hat)
        """
        
        # 1) set mode : mode control은 외부에서. 여기서는 forwarding만 수행. 
        # self.state_encoder.train()
        # self.subgoal_generator.train()

        self.prior_policy.eval() # 반드시 eval모드. 절대로 업데이트 안함.  
        self.target_state_encoder.eval()

        # 2) get high-states
        ht = self.state_encoder.dist(inputs.states)
        with torch.no_grad():
            hstar = self.target_state_encoder.dist(inputs.G).sample()
            h_tH = self.target_state_encoder.dist(inputs.next_H_states)

        # 3) generate subgoals
        h_tH_hat = self.subgoal_generator.dist(torch.cat((ht.rsample(), hstar), dim= -1))
        
        # 4) prior inputs
        prior_input = torch.cat((ht.sample(), h_tH_hat.rsample()), -1)
        prior_input_GT = torch.cat((ht.sample(), h_tH.sample()), -1)
        
        # not registered in prior optimizer. 뭔짓을해도 절대 업데이트 안된다. 
        prior_hat, _ = self.prior_policy.dist(prior_input)
        prior_GT, _ = self.prior_policy.dist(prior_input_GT)

        # 4) 
        return edict(
            ht = ht,
            h_tH = h_tH,
            h_tH_hat = h_tH_hat,
            prior_hat = prior_hat,
            prior_GT = prior_GT
        )

    def __finetune__(self, inputs):
        """
        Fine-tuning method for non-distributional mode 
        train 시에는 어차피 다 train으로 돌아가 있음. 추가적으로 eval 모드가 필요한 것만 해주면 됨. 
        """
        
        self.prior_policy.eval() # 반드시 eval모드. 절대로 업데이트 안함.  
        self.target_state_encoder.eval()

        # 2) get high-states
        ht = self.state_encoder(inputs.states)
        with torch.no_grad():
            hstar = self.target_state_encoder(inputs.G)
            h_tH = self.target_state_encoder(inputs.next_H_states)

        # 3) generate subgoals
        h_tH_hat = self.subgoal_generator(torch.cat((ht, hstar), dim= -1))
        
        # 4) prior inputs
        prior_input = torch.cat((ht.clone().detach(), h_tH_hat), -1)
        prior_input_GT = torch.cat((ht.clone().detach(), h_tH), -1)
        
        # not registered in prior optimizer. 뭔짓을해도 절대 업데이트 안된다. 
        prior_hat, _ = self.prior_policy.dist(prior_input)
        prior_GT, _ = self.prior_policy.dist(prior_input_GT)

        # 4) 
        return edict(
            ht = ht,
            h_tH = h_tH,
            h_tH_hat = h_tH_hat,
            prior_hat = prior_hat,
            prior_GT = prior_GT
        )


    
class MLPPolicy(ContextPolicyMixin, SequentialBuilder):
    """
    Low Policy for single task SAC
    """
    def __init__(self, config):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
    
    def forward(self, states):
        return super().forward(states)

    def act(self, states):
        dist = self.dist(states)
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, states):

        states = prep_state(states, self.device)
        loc, log_scale = self(states).chunk(2, -1)
        scale = log_scale.clamp(-20, 2).exp()
        return torch_dist.Independent(torch_dist.Normal(loc, scale), 1)

class MLPQF(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, batch_state, batch_action):
        batch_state = prep_state(batch_state, self.device)
        batch_action = prep_state(batch_action, self.device)
        concat = torch.cat([batch_state, batch_action], dim=-1)
        return super().forward(concat).squeeze(-1)


class GCPriorLinear(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.z = None

    # for compatibility with simpl
    def dist_param(self, state):
        only_states = state[:, : self.in_feature]
        out = self(only_states)
        mu, log_std = out.chunk(2, dim = 1)
        return mu, log_std.clamp(-10, 2)

    def dist(self, batch_state):    
        dist = get_dist(self(batch_state[:, : self.in_feature]))
        return dist


class MixedMLPPolicy(ContextPolicyMixin, SequentialBuilder):
    """
    Low Policy for single task SAC
    """
    def __init__(self, config, prior_policy):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.min_scale=0.001
    def forward(self, states):
        return super().forward(states)

    def act(self, states):
        dist = self.dist(states)
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, states):
        self.prior_policy.eval()
        states = prep_state(states, self.device)

        # distributions from prior state
        prior_locs, prior_log_scales = self.prior_policy.dist_param(states)
        prior_pre_scales = inverse_softplus(prior_log_scales.exp())
        
        # distributions from policy state
        res_locs, res_pre_scales = self(states).chunk(2, dim=-1)

        # 혼합
        dist = torch_dist.Normal(
            prior_locs + res_locs,
            self.min_scale + F.softplus(prior_pre_scales + res_pre_scales)
        )
        return torch_dist.Independent(dist, 1)

class MMPH(ContextPolicyMixin, SequentialBuilder):
    """
    Mixed MLP Policy with high state
    """
    def __init__(self, config, prior_policy):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.min_scale=0.001

    def forward(self, states):
        return super().forward(states)

    def act(self, states):

        dist = self.dist(edict(states = states))
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, inputs):
        self.prior_policy.eval()
        states = inputs.states

        states = prep_state(states, self.device)

        # distributions from policy state
        res_locs, res_pre_scales = self(states).chunk(2, dim=-1)

        # 혼합
        dist = torch_dist.Normal(
            res_locs, # + prior_locs,
            self.min_scale + F.softplus(res_pre_scales ) # + prior_pre_scales)
        )
        return torch_dist.Independent(dist, 1)

class MMPHSG(ContextPolicyMixin, SequentialBuilder):
    """
    Mixed MLP Policy with high state
    """
    def __init__(self, config, prior_wrapper):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.prior_policy = copy.deepcopy(prior_wrapper).requires_grad_(False)

        self.min_scale=0.001
    def forward(self, states):
        return super().forward(states)

    def act(self, states, G, mode = "train"):
        dist_inputs = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device)
        )
        dist = self.dist(dist_inputs, mode)
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, inputs, mode = "train"):
        self.prior_policy.eval()

        states, G = inputs.states, inputs.G
        
        # self.prior_policy.eval()
        # states = prep_state(states, self.device)

        # # distributions from prior state
        # prior_locs, prior_log_scales = self.prior_policy.dist_param(states)
        # prior_pre_scales = inverse_softplus(prior_log_scales.exp())

        states = prep_state(states, self.device)
        G = prep_state(G, self.device)

        if states.shape[0] != G.shape[0]:
            # expand
            G = G.repeat(states.shape[0], 1)

        inputs = edict(
            states = states,
            G = G
        )

        # # distributions from prior state
        result = self.prior_policy(inputs, mode)
        prior_dist = result.prior.base_dist
        prior_locs, prior_scales = prior_dist.loc, prior_dist.scale
        prior_pre_scales = inverse_softplus(prior_scales)
        
        # distributions from policy state

        res_locs, res_pre_scales = self(states).chunk(2, dim=-1)

        # 혼합
        dist = torch_dist.Normal(
            res_locs + prior_locs,
            self.min_scale + F.softplus(res_pre_scales + prior_pre_scales)
        )
        return torch_dist.Independent(dist, 1)