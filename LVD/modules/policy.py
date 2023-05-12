
import torch
import torch.nn as nn
import torch.distributions as torch_dist
import copy

from copy import deepcopy
from ..modules.base import SequentialBuilder, ContextPolicyMixin
from ..utils import *
from ..contrib.momentum_encode import update_moving_average
from ..contrib.dists import TanhNormal

class MLPQF(SequentialBuilder):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, batch_state, batch_action):
        batch_state = prep_state(batch_state, self.device)
        batch_action = prep_state(batch_action, self.device)
        concat = torch.cat([batch_state, batch_action], dim=-1)
        return super().forward(concat).squeeze(-1)





class HighPolicy_SC(ContextPolicyMixin, SequentialBuilder):
    """
    MLP Policy for SPiRL, SiMPL, Skimo
    """
    def __init__(self, config, prior_policy, prior_state_dim = None):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.min_scale=0.001
        self.prior_state_dim = prior_state_dim

    def forward(self, states):
        return super().forward(states)

    def act(self, states):
        dist_inputs = dict(
            states = prep_state(states, self.device),
        )
    

        dist = self.dist(dist_inputs, "eval")
        # TODO explore 여부에 따라 mu or sample을 결정
        # return dist.rsample().detach().cpu().squeeze(0).numpy()

        loc, scale = dist.base_dist.loc, dist.base_dist.scale


        # return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        return self.transform_numpy(dist.rsample()), self.transform_numpy(loc), self.transform_numpy(scale)


        # if self.prior_policy.tanh:
        #     z_normal, z = dist.rsample_with_pre_tanh_value()
        #     # to calculate kld analytically 
        #     loc, scale = dist._normal.base_dist.loc, dist._normal.base_dist.scale 

        #     return z_normal.detach().cpu().squeeze(0).numpy(), z.detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        # else:
        #     loc, scale = dist.base_dist.loc, dist.base_dist.scale
        #     return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()

    def dist(self, inputs, mode):
        prior_inputs = {**inputs}
        policy_inputs = {**inputs}

        if self.prior_state_dim is not None:
            prior_inputs['states'] = prior_inputs['states'][:, :self.prior_state_dim]

        with torch.no_grad():
            result = self.prior_policy(prior_inputs, mode)

        prior_dist = result['prior'].base_dist
        prior_locs, prior_scales = prior_dist.loc, prior_dist.scale
        prior_pre_scales = inverse_softplus(prior_scales)

        # distributions from policy state
        states = prep_state(policy_inputs['states'], self.device)
        res_locs, res_pre_scales = self(states).chunk(2, dim=-1)

        # 혼합
        dist = torch_dist.Normal(
            res_locs + prior_locs,
            self.min_scale + F.softplus(res_pre_scales + prior_pre_scales)
        )
        return torch_dist.Independent(dist, 1)



class HighPolicy_GC(ContextPolicyMixin, SequentialBuilder):
    """
    MLP Policy for LVD
    """
    def __init__(self, config, prior_policy, state_dim = None):
        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.min_scale=0.001
        self.prior_policy = deepcopy(prior_policy) # learnable
        del self.layers

    def forward(self, states):
        return super().forward(states)

    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
        dist_inputs = dict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(dist_inputs, "eval")['inverse_D']
        # TODO explore 여부에 따라 mu or sample을 결정
        if self.prior_policy.tanh:
            z_normal, z = dist.rsample_with_pre_tanh_value()
            # to calculate kld analytically 
            loc, scale = dist._normal.base_dist.loc, dist._normal.base_dist.scale 

            return z_normal.detach().cpu().squeeze(0).numpy(), z.detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        else:
            loc, scale = dist.base_dist.loc, dist.base_dist.scale
            return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()

    def dist(self, inputs, prior_mode): # mode = "train"
        states, G = inputs['states'], inputs['G']

        # 
        states = prep_state(states, self.device)
        G = prep_state(G, self.device)

        if states.shape[0] != G.shape[0]:
            # expand
            G = G.repeat(states.shape[0], 1)

        _inputs = dict(
            states = states,
            G = G
        )

        if 'next_states' in inputs.keys():
            _inputs['next_states'] = prep_state(inputs['next_states'], self.device)

        outputs = self.prior_policy(_inputs, prior_mode)

        return outputs

    def set_policy(self, prior_policy):
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
    

    def finetune(self, inputs):
        """
        Finetune inverse D & D
        """
        outputs = self.prior_policy(inputs, "finetune")
        
        return outputs

    def soft_update(self):
        self.prior_policy.soft_update()


class HighPolicy_GC_Naive(ContextPolicyMixin, SequentialBuilder):
    """
    MLP Policy for LVD
    """
    def __init__(self, config, prior_policy, state_dim = None):
        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.min_scale=0.001
        self.prior_policy = deepcopy(prior_policy) # learnable
        del self.layers

    def forward(self, states):
        return super().forward(states)

    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
        dist_inputs = dict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(dist_inputs, "eval")['policy_skill']
        # TODO explore 여부에 따라 mu or sample을 결정
        if self.prior_policy.tanh:
            z_normal, z = dist.rsample_with_pre_tanh_value()
            # to calculate kld analytically 
            loc, scale = dist._normal.base_dist.loc, dist._normal.base_dist.scale 

            return z_normal.detach().cpu().squeeze(0).numpy(), z.detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        else:
            loc, scale = dist.base_dist.loc, dist.base_dist.scale
            return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()

    def dist(self, inputs, prior_mode): # mode = "train"
        states, G = inputs['states'], inputs['G']

        # 
        states = prep_state(states, self.device)
        G = prep_state(G, self.device)

        if states.shape[0] != G.shape[0]:
            # expand
            G = G.repeat(states.shape[0], 1)

        _inputs = dict(
            states = states,
            G = G
        )

        if 'next_states' in inputs.keys():
            _inputs['next_states'] = prep_state(inputs['next_states'], self.device)

        outputs = self.prior_policy(_inputs, prior_mode)

        return outputs

    def set_policy(self, prior_policy):
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
    

    def finetune(self, inputs):
        """
        Finetune inverse D & D
        """
        outputs = self.prior_policy(inputs, "finetune")
        
        return outputs

    def soft_update(self):
        self.prior_policy.soft_update()

class HighPolicy_GC_Dreamer(ContextPolicyMixin, SequentialBuilder):
    """
    MLP Policy for LVD
    """
    def __init__(self, config, prior_policy, state_dim = None):
        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.min_scale=0.001
        self.prior_policy = deepcopy(prior_policy) # learnable
        del self.layers

    def forward(self, states):
        return super().forward(states)

    def act(self, states, G):
        # 환경별로 state를 처리하는 방법이 다름.
        # 여기서 수행하지 말고, collector에서 전처리해서 넣자. 
        dist_inputs = dict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
        )

        dist = self.dist(dist_inputs, "eval")['policy_skill']
        # TODO explore 여부에 따라 mu or sample을 결정
        if self.prior_policy.tanh:
            z_normal, z = dist.rsample_with_pre_tanh_value()
            # to calculate kld analytically 
            loc, scale = dist._normal.base_dist.loc, dist._normal.base_dist.scale 

            return z_normal.detach().cpu().squeeze(0).numpy(), z.detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()
        else:
            loc, scale = dist.base_dist.loc, dist.base_dist.scale
            return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()

    def dist(self, inputs, prior_mode): # mode = "train"
        states, G = inputs['states'], inputs['G']

        # 
        states = prep_state(states, self.device)
        G = prep_state(G, self.device)

        if states.shape[0] != G.shape[0]:
            # expand
            G = G.repeat(states.shape[0], 1)

        _inputs = dict(
            states = states,
            G = G
        )

        if 'next_states' in inputs.keys():
            _inputs['next_states'] = prep_state(inputs['next_states'], self.device)

        outputs = self.prior_policy(_inputs, prior_mode)

        return outputs

    def set_policy(self, prior_policy):
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
    

    def finetune(self, inputs):
        """
        Finetune inverse D & D
        """
        outputs = self.prior_policy(inputs, "finetune")
        
        return outputs

    def soft_update(self):
        self.prior_policy.soft_update()


class HighPolicy_Skimo(ContextPolicyMixin, SequentialBuilder):
    """
    Skimo
    """
    def __init__(self, build_config, skimo_config):

        super().__init__(build_config)
        
        for k, v in skimo_config.items():
            setattr(self, k, v)

        self.prior_policy = copy.deepcopy(self.prior_policy).requires_grad_(False)

        # self._step : episode 길이의 누적합. 
        


    def forward(self, states):
        return super().forward(states)
    
    @torch.no_grad()
    def act(self, states, G, qfs):
        dist_inputs = dict(
            states = prep_state(states, self.device),
            G = prep_state(G, self.device),
            qfs = qfs,
        )

        if self._step < self.warmup_steps:
            skill = self.dist(dist_inputs)['policy_skill'].sample()[0]
        else:
            skill = self.cem_planning(dist_inputs)

        
        self._step += 10 # skill length 

        return skill.detach().cpu().numpy() 

    def dist(self, inputs):
        return self.prior_policy(inputs, "eval")

    @torch.no_grad()
    def estimate_value(self, state, skills, G, horizon, qfs):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            # step을 추가하자
            state_skill = torch.cat((state, skills[:, t]), dim  = -1)
            state = self.prior_policy.dynamics(state_skill)
            reward = self.reward_function(state_skill) 
            value += discount * reward
            discount *= self.rl_discount


        
        # policy_skill = self.prior_policy(policy_inputs, "eval")['policy_skill'].sample()

        policy_skill =  self.prior_policy.highlevel_policy.dist(torch.cat((state.clone().detach(), G), dim = -1)).sample()

        q_values = [  qf( state, policy_skill).unsqueeze(-1)   for qf in qfs]
        value += discount * torch.min(*q_values) # 마지막엔 Q에 넣어서 value를 구함. 
        return value

    @torch.no_grad()
    def cem_planning(self, inputs):
        """
        Cross Entropy Method
        """

        planning_horizon  = int(self._horizon_decay(self._step))

        states, qfs = inputs['states'], inputs['qfs']
        states = self.prior_policy.state_encoder(states)

        # Sample policy trajectories.
        states_policy = states.repeat(self.num_policy_traj, 1) 
        
        rollout_inputs = dict(
            states = states_policy,
            G = inputs['G'].repeat(self.num_policy_traj, 1) ,
            planning_horizon = planning_horizon
        )

        policy_skills =  self.prior_policy(rollout_inputs, "rollout")['policy_skills']

        # CEM optimization.
        state_cem = states.repeat(self.num_policy_traj + self.num_sample_traj, 1)
        
        # zero mean, unit variance
        momentums = torch.zeros(self.num_sample_traj, planning_horizon, self.skill_dim * 2, device= self.device)
        loc, log_scale = momentums.chunk(2, -1)
        dist = get_dist(loc, log_scale= log_scale, tanh = self.tanh)

        for _ in range(self.cem_iter):
            # sampled skill + policy skill
            sampled_skill = dist.sample()
            skills = torch.cat([sampled_skill, policy_skills], dim=0)

            # reward + value 
            imagine_return = self.estimate_value(state_cem, skills, inputs['G'].repeat(skills.shape[0], 1) , planning_horizon, qfs)
                        
            # sort by reward + value
            elite_idxs = imagine_return.sort(dim=0)[1].squeeze(1)[-self.num_elites :]
            elite_value, elite_skills = imagine_return[elite_idxs], skills[elite_idxs]

            # Weighted aggregation of elite plans.
            score = torch.softmax(self.cem_temperature * elite_value, dim = 0).unsqueeze(-1)

            dist = self.score_weighted_skills(loc, score, elite_skills)

        # Sample action for MPC.
        score = score.squeeze().cpu().numpy()
        # score를 기준으로 action을 samplin
        skill = elite_skills[np.random.choice(np.arange(self.num_elites), p=score), 0] # 마구 수행해봤을 때 점수가 제일 높았던 skill 선택. 
        return skill #torch.clamp(skill, -0.999, 0.999)

    def finetune(self, inputs):
        """
        Finetune dynamics & state encoder 
        """
        # finetune state encoder and dynamics 
        outputs = self.prior_policy(inputs, "finetune")

        outputs['rwd_pred'] = self.reward_function(torch.cat((inputs['q_states'], inputs['actions']), dim = -1)) 

        return outputs

    def soft_update(self):
        self.inverse_dynamics.soft_update()

    def _std_decay(self, step):
        # from rolf
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 0.5 * (1-mix) + 0.01 * mix

    def _horizon_decay(self, step):
        # from rolf
        mix = np.clip(step / self.step_interval, 0.0, 1.0)
        return 1 * (1-mix) + self.planning_horizon * mix


    def score_weighted_skills(self, loc, score, skills):
        weighted_loc = (score * skills).sum(dim=0)
        weighted_std = torch.sqrt(torch.sum(score * (skills - weighted_loc.unsqueeze(0)) ** 2, dim=0))
        
        # soft update 
        loc = self.cem_momentum * loc + (1 - self.cem_momentum) * weighted_loc
        log_scale = torch.clamp(weighted_std, self._std_decay(self._step), 2).log() # new_std의 최소값. .. 을 해야돼? 
        dist = get_dist(loc, log_scale, tanh = self.tanh)

        return dist



class HighPolicy_GC_Prompt(HighPolicy_GC):
    """
    MLP Policy for LVD
    """
    def __init__(self, config, inverse_dynamics, state_dim = None):
        super().__init__(config, inverse_dynamics, state_dim)
        self.goal_prompt = nn.Parameter(torch.randn(30))

    def act(self, states, G):
        G = self.goal_prompt
        return super().act(states, G)
    
    def dist(self, inputs, prior_mode): # mode = "train"
        inputs['G'] = self.goal_prompt
        return super().dist(inputs, prior_mode)