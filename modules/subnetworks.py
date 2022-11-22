import torch
import torch.nn as nn
from proposed.utils import get_dist, get_fixed_dist, prep_state, inverse_softplus
import numpy as np
import torch.distributions as torch_dist
from .base import SequentialBuilder, ContextPolicyMixin, BaseModule
from proposed.utils import *
import copy
from easydict import EasyDict as edict
from proposed.contrib.momentum_encode import EMA, update_moving_average


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
        self.ema_updater = None
        
        super().__init__(submodules)
        
        self.forward_dict = {
            "default" : self.default, # state only conditioned
            "inverse" : self.highS_inverseD, # inverse dynamics prior
            "did" : self.DID, # inverse dynamics & dynamics prior
            "gcid" : self.GCID, # inverse dynamics & dynamics prior
            "gc" : self.default, # state에 goal이 relabeling된 경우
            "vic" : self.GCID_VIC,

        }

        if self.ema_updater:
            self.target_state_encoder = copy.deepcopy(self.state_encoder)
            
    def forward(self, *args):
        return self.forward_dict[self.mode](*args)

    def ma_state_enc(self):
        update_moving_average(self.ema_updater, self.target_state_encoder, self.state_encoder)

    def default(self, s0):
        # state only conditioned
        prior, prior_detach = self.prior_policy.dist(s0, detached = True)
        return edict(
            prior = prior,
            prior_detach = prior_detach
        )

    def _hs_id(self, states, h_tH = None):
        """
        High states, inverse dynamics
        states : raw state sequences or single states if hs_ht is not None
        hs_th : high states of t+H time step from other module
        """

        if h_tH is None: # for prior training
            # state AE
            N, T, D = states.shape
            # encoder : variational inference
            hsd = self.state_encoder.dist(states.view(N * T, -1)) # dist
            hs = hsd.rsample().view(N, T, -1) # N * T, -1

            hs_target = get_fixed_dist(torch.randn(N * T, 20).cuda())

            # distribution of time step t+H
            with torch.no_grad():
                self.target_state_encoder.eval()
                _, hsd_tH = self.target_state_encoder.dist(states[:,-1], detached = True)
                # _, hsd_tH = self.state_encoder.dist(states[:,-1], detached = True)


            state_hat = self.state_decoder(hs.view(N * T, -1)).view(N, T, -1)

            # inverse dynamics 
            ht, h_tH = hs[ : , 0], hs[ : , -1]
            prior_input = torch.cat((ht, h_tH), -1)
            prior, prior_detach = self.prior_policy.dist(prior_input, detached = True)


            return edict(
                # shape inform
                N = N,
                T = T,
                # distributions
                hsd = hsd,
                hsd_tH = hsd_tH,
                # N(0, I)
                hs_target = hs_target,
                # for subgoal generating
                ht = ht,
                # state reconstruction
                states = states,
                s_hat = state_hat, 
                # prior distributions 
                prior = prior,
                prior_detach = prior_detach
            )

        else: # for RL
            ht = self.state_encoder.dist(states).sample()# single time step
            prior_input = torch.cat((ht, h_tH), -1)
            prior, prior_detach = self.prior_policy.dist(prior_input)  
            return edict(
                prior = prior
            )

    def _hs_d(self, ht, z):
        dynamics_input = torch.cat((ht, z), dim = -1)
        return self.dynamics(dynamics_input)


    def _generate_subgoal(self, x):
        """
        generating subgoals
        """
        return self.subgoal_generator(x)



    def highS_inverseD(self, s, G, inference = False):

        if inference: # in RL or more
            s = prep_state(s, self.device)
            G = prep_state(G, self.device)

            if s.shape[0] != G.shape[0] : # in RL
                G = G.repeat(s.shape[0], -1)

            h_tH = self._generate_subgoal(torch.cat((s, G), dim = -1)).sample()
            result = self._hs_id(s, h_tH)
        else:
            h_tH = self._generate_subgoal(torch.cat((s, G), dim = -1))
            result = self._hs_id(s)
            result['h_tH_hat'] = h_tH

        return result

    def DID(self, states, G):
        """
        Dynamics, Inverse Dynamics
        deprecated
        """
        # inverse dynamics 
        result = self._hs_id(states)

        # dynamics
        z = result.prior.rsample()
        result['h_tH_dynamics'] = self._hs_d(result.ht, z)

        # subgoal
        with torch.no_grad(): 
            self.target_state_encoder.eval()
            h_star = self.target_state_encoder(G) 
        result['h_tH_pred'] = self._generate_subgoal(torch.cat((result.ht, h_star), dim = -1))
        # prior hat 
        prior_input = torch.cat((result.ht.detach(), result.h_tH_pred), -1)
        result['prior_hat'] = self.prior_policy.dist(prior_input)[1]
        
        
        
        return result

    def GCID(self, inputs, inference = False):

        states, G = inputs.states, inputs.G
        if not inference:
            result = self._hs_id(states)

            # dynamics
            # result['dynamics'] = self.dynamics.dist(torch.cat( (result.ht, result.prior.rsample()) ,dim = -1))

            # transition probability 
            with torch.no_grad(): 
                self.target_state_encoder.eval()
                h_star = self.target_state_encoder.dist(G).sample() 
            sg_input = torch.cat((result.ht, h_star), dim = -1)
            result['h_tH_hat_dist'] = self.subgoal_generator.dist(sg_input)
            
            
            # prior hat 
            with torch.no_grad():
                prior_input = torch.cat((result.ht, result.h_tH_hat_dist.sample()), -1)
                result['prior_hat'] = self.prior_policy.dist(prior_input)[1]

        
            
            return result
        else: # RL

            # get ht
            ht = self.state_encoder.dist(states).sample()

            # get h_star
            with torch.no_grad(): 
                self.target_state_encoder.eval()
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

    def GCID_VIC(self, inputs, inference = False):

        result = self.GCID(inputs, inference)
        if not inference:
            result['pr_proj'] = self.prior_proj(result.prior.rsample())
            # 뺑뺑이
            # with torch.no_grad():
            #     h_th_hat_cycle = self.state_encoder(self.state_decoder(result.h_tH_hat_dist.sample()))
            #     prior_input_cycle = torch.cat((result.ht, h_th_hat_cycle), -1)
            #     pr_proj_cycle, _ = self.prior_policy.dist(prior_input_cycle)  
            #     result['pr_proj_cycle'] = self.prior_proj(pr_proj_cycle.sample())
                

        else:
            result['pr_proj'] = self.prior_proj(result.prior.sample())
            
            # # 뺑뺑이
            # h_th_hat_cycle = self.state_encoder(self.state_decoder(result.h_tH_hat))
            # prior_input_cycle = torch.cat((result.ht, h_th_hat_cycle), -1)
            # result['pr_proj_cycle'], _ = self.prior_policy.dist(prior_input_cycle)  
            

        return result




    
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
        dist = self.dist(states)
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, states):
        self.prior_policy.eval()
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

    def act(self, states, G):
        dist_inputs = edict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device)
        )
        dist = self.dist(dist_inputs)
        # TODO explore 여부에 따라 mu or sample을 결정
        return dist.rsample().detach().cpu().squeeze(0).numpy()

    def dist(self, inputs):
        self.prior_policy.eval()

        states, G = inputs.states, inputs.G
        
        # states = prep_state(states, self.device)
        # G = prep_state(G, self.device)

        # if states.shape[0] != G.shape[0]:
        #     # expand
        #     G = G.repeat(states.shape[0], 1)



        # # distributions from prior state
        # result = self.prior_policy(states, G, True)
        # prior_dist = result.prior.base_dist
        # prior_locs, prior_scales = prior_dist.loc, prior_dist.scale
        # prior_pre_scales = inverse_softplus(prior_scales)
        
        # distributions from policy state

        res_locs, res_pre_scales = self(states).chunk(2, dim=-1)

        # 혼합
        dist = torch_dist.Normal(
            res_locs, # + prior_locs,
            self.min_scale + F.softplus(res_pre_scales )#+ prior_pre_scales)
        )
        return torch_dist.Independent(dist, 1)