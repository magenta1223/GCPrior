
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
    def __init__(self, config, prior_policy, prior_state_dim = None, visual_encoder = None):

        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
        self.min_scale=0.001
        self.prior_state_dim = prior_state_dim
        self.visual_encoder = visual_encoder

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


        return dist.rsample().detach().cpu().squeeze(0).numpy(), loc.detach().cpu().squeeze(0).numpy(), scale.detach().cpu().squeeze(0).numpy()


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
    def __init__(self, config, inverse_dynamics, state_dim = None):
        super().__init__(config)
        self.log_sigma = nn.Parameter(torch.randn(self.out_dim)) # for sampling
        self.min_scale=0.001
        self.inverse_dynamics = deepcopy(inverse_dynamics) # learnable
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
        if self.inverse_dynamics.tanh:
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

        outputs = self.inverse_dynamics(_inputs, prior_mode)

        return outputs

    def set_policy(self, prior_policy):
        self.prior_policy = copy.deepcopy(prior_policy).requires_grad_(False)
    

    def finetune(self, inputs):
        """
        Finetune inverse D & D
        """
        outputs = self.inverse_dynamics(inputs, "finetune")
        
        return outputs

    def soft_update(self):
        self.inverse_dynamics.soft_update()






class HighPolicy_Skimo(ContextPolicyMixin, SequentialBuilder):
    """
    Skimo
    """
    def __init__(self, build_config, others_config):

        super().__init__(build_config)
        
        for k, v in others_config.items():
            setattr(self, k, v)

        self.prior_policy = copy.deepcopy(self.prior_policy).requires_grad_(False)

        # self._step : episode 길이의 누적합. 

    def forward(self, states):
        return super().forward(states)
    
    @torch.no_grad()
    def act(self, states, qfs):
        dist_inputs = dict(
            states = prep_state(states, self.device),
            qfs = qfs,
        )
        # dist = self.dist(dist_inputs, "eval")

        # skill = self.cem_planning(dist_inputs)
        

        dist_inputs['states'] = self.prior_policy.state_encoder(dist_inputs['states'][:, :self.prior_state_dim])

        skill = self.dist(dist_inputs).sample()
        
        self._step += 10 # skill length 
        # 
        return skill.detach().cpu().numpy()[0] 

    def dist(self, inputs, encode = False):
        prior_inputs = {**inputs}
        policy_inputs = {**inputs}

        if self.prior_state_dim is not None:
            prior_inputs['states'] = prior_inputs['states'][:, :self.prior_state_dim]

        with torch.no_grad():
            result = self.prior_policy(prior_inputs, "prior")

        if self.tanh:
            prior_dist = result['prior']._normal.base_dist
        else:
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
        dist =  torch_dist.Independent(dist, 1)

        if self.tanh:
            return TanhNormal(res_locs + prior_locs, self.min_scale + F.softplus(res_pre_scales + prior_pre_scales))
        else:
            return dist

    
    @torch.no_grad()
    def estimate_value(self, state, skills, horizon, qfs):
        """Imagine a trajectory for `horizon` steps, and estimate the value."""
        value, discount = 0, 1
        for t in range(horizon):
            state = self.prior_policy.dynamics(torch.cat((state, skills[:, t]), dim  = -1))
            reward = self.reward_function(torch.cat((state, skills[:, t]), dim= -1)) 
            value += discount * reward
            discount *= self.rl_discount

        _inputs = dict(
            states = state
        )
        values = [  qf( state, self.dist(_inputs).sample()).unsqueeze(1)   for qf in qfs]
        value += discount * torch.min(*values) # 마지막엔 Q에 넣어서 value를 구함. 
        return value

    @torch.no_grad()
    def cem_planning(self, inputs):
        """Plan given an observation `ob`."""

        planning_horizon  = int(self._horizon_decay(self._step))

        state, qfs = inputs['states'][:, :self.prior_state_dim], inputs['qfs']
        state = self.prior_policy.state_encoder(state)

        # Sample policy trajectories.
        hs = state.repeat(self.num_policy_traj, 1) 
            
        # rollout by policy
        policy_skills = []
        for t in range(planning_horizon):
            _inputs = dict(states = hs)
            policy_skills.append(self.dist(_inputs).sample()) 
            hs = self.prior_policy.dynamics(torch.cat((hs, policy_skills[-1]), dim = -1  ))
        policy_skills = torch.stack(policy_skills, dim=1) # N_plan, planning_horizon, skill_dim 

        # CEM optimization.
        hs = state.repeat(self.num_policy_traj + self.num_sample_traj, 1)

        loc = torch.zeros(self.num_sample_traj, planning_horizon, self.skill_dim, device= self.device)
        log_scale = torch.zeros(self.num_sample_traj, planning_horizon, self.skill_dim, device= self.device)
        dist = get_dist(loc, log_scale, tanh = self.tanh)


        for _ in range(self.cem_iter):
            sampled_skill = dist.sample() # n_sample_traj, planning_H, skill_dim
            # 랜덤샘플한 skill과 실제 policy가 취한 skill sequence를 합친다. 
            skills = torch.cat([sampled_skill, policy_skills], dim=0)
        
            # reward function으로 reward 예측함. 
            imagine_return = self.estimate_value(hs, skills, planning_horizon, qfs).squeeze(-1) 
            # 현재 state에서 샘플링한 skill sequence를 넣고 걍 돌림. reward 예측
            
            # reward별로 정렬
            _, idxs = imagine_return.sort(dim=0)
            idxs = idxs[-self.num_elites :]
            # elite set / skill을 선택
            elite_value = imagine_return[idxs]
            elite_skills = skills[idxs]

            # Weighted aggregation of elite plans.
            # softmax with temperature 
            score = torch.exp(self.cem_temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(-1, 1, 1) # 1, N, 1 -> N, 1, 1

            # best mean과 std를 구함. 
            # score가 prob이므로 sum해야 mean이 됨. 
            new_loc = (score * elite_skills).sum(dim=0)
            new_std = torch.sqrt(torch.sum(score * (elite_skills - new_loc.unsqueeze(0)) ** 2, dim=0))
            
            # soft update 
            loc = self.cem_momentum * loc + (1 - self.cem_momentum) * new_loc
            log_scale = torch.clamp(new_std, self._std_decay(self._step), 2).log() # new_std의 최소값. .. 을 해야돼? 
            dist = get_dist(loc, log_scale, tanh = self.tanh)

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

        outputs['rwd_pred'] = self.reward_function(torch.cat((inputs['states'], inputs['actions']), dim = -1)) 

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