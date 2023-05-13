
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..modules.base import BaseModule
from ..contrib.simpl.math import clipped_kl, inverse_softplus
from ..utils import prep_state, kl_annealing, AverageMeter
from ..contrib.momentum_encode import update_moving_average
from ..contrib.dists import *


class SAC(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        
        # Q-functions
        self.qfs = nn.ModuleList(self.qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in self.qfs])
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=self.qf_lr) for qf in self.qfs]

        if self.qf_lr < 1e-12:
            self.optimal_Q = True
        else:
            self.optimal_Q = False

        self.policy_optim = torch.optim.Adam(
            [
                { "params" : self.policy.highlevel_policy.parameters()},  
            ],
            lr = self.policy_lr 
        )


        # finetune : state_encoder, dynamics, reward function 
        self.others_optim = torch.optim.Adam(
            [
                { "params" : self.policy.state_encoder.parameters()}, # 보상최대화하는 subgoal 뽑기. 
                { "params" : self.policy.dynamics.parameters()}, # 보상최대화하는 subgoal 뽑기. 
                { "params" : self.policy.reward_function.parameters()}, # 보상최대화하는 subgoal 뽑기. 
            ],
            lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )


        # Alpha
        if isinstance(self.init_alpha, float):
            self.init_alpha = torch.tensor(self.init_alpha)

        pre_init_alpha = inverse_softplus(self.init_alpha)

        if self.auto_alpha:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=self.alpha_lr)
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.n_step = 0
        self.init_grad_clip_step = 0
        self.init_grad_clip = 5.0

        self.rho = 0.5

        # self.save_prev_module()


        self.sample_time_logger = AverageMeter()


    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def entropy(self, inputs, kl_clip = False):
        inputs = {**inputs}

        with torch.no_grad():
            prior_dist = self.policy.prior_policy(inputs, "prior")['prior']
        if kl_clip:                
            entropy = clipped_kl(inputs['dist'], prior_dist, clip = self.kl_clip)
        else:
            entropy = torch_dist.kl_divergence(inputs['dist'], prior_dist)
        return entropy, prior_dist 
    
    
    def grad_clip(self, optimizer):
        if self.n_step < self.init_grad_clip_step:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.init_grad_clip) 

    def step(self, step_inputs):
        batch = self.buffer.sample(step_inputs['batch_size']).to(self.device)
        self.episode = step_inputs['episode']

        with torch.no_grad():
            states = prep_state(batch.states, self.device)
            next_states = prep_state(batch.next_states, self.device)
            step_inputs['rewards'] = batch.rewards
            step_inputs['dones'] = batch.dones


            step_inputs['states'] = states
            step_inputs['next_states'] = next_states

            step_inputs['q_states'] = self.policy.state_encoder(states) 
            step_inputs['q_next_states'] = self.policy.state_encoder(next_states)
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['batch_size'], 1).cuda()



            step_inputs['raw_states'] = states
            step_inputs['raw_next_states'] = next_states

            step_inputs['done'] = True

            # 요녀석들은 전부 evaluation mode에서 뽑아낸 action들임. 
            step_inputs['actions'] = prep_state(batch.actions, self.device) # high-actions





        self.n_step += 1

        return self._step(step_inputs)

    def _step(self, step_inputs):
        stat = {}

        step_inputs['rollout_high_states']  = self.update_Q_models(step_inputs)

        results = self.update_policy(step_inputs)



        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        # q_results = self.update_qs(step_inputs)
        # subgoal_results = self.update_policy(step_inputs)

        for k, v in results.items():
            stat[k] = v 



        # ------------------- Logging ------------------- # 
        stat['target_kl'] = self.target_kl

        # ------------------- Alpha ------------------- # 
        alpha_results = self.update_alpha(stat['kl'])
        for k, v in alpha_results.items():
            stat[k] = v 

        return stat
    
    @torch.no_grad()
    def compute_target_q(self, step_inputs):
        
        policy_inputs = dict(
            states = step_inputs['next_states'].clone(),
            G = step_inputs['G'],
            # raw_states = step_inputs['raw_next_states']
        )
        
        policy_inputs['dist'] = self.policy.dist(policy_inputs)['policy_skill']
        actions = policy_inputs['dist'].sample()

        # calculate entropy term
        entropy_term, prior_dist = self.entropy(policy_inputs, kl_clip= True) 
        min_qs = torch.min(*[target_qf(step_inputs['q_next_states'], actions) for target_qf in self.target_qfs])

        if self.alpha < 1e-12:
            soft_qs = min_qs
        else:
            soft_qs = min_qs - self.alpha*entropy_term

        rwd_term = step_inputs['rewards'].cuda()
        ent_term = (1 - step_inputs['dones'].cuda())*self.discount*soft_qs

        return rwd_term, ent_term, entropy_term


    def update_alpha(self, kl):
        results = {}
        if self.auto_alpha is True:
            # dual gradient decent 
            alpha_loss = (self.alpha * (self.target_kl - kl.clone().detach())).mean()

            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            results['alpha_loss'] = alpha_loss
            results['alpha'] = self.alpha

        return results
    
    def update_Q_models(self, step_inputs):
        states = step_inputs['states']
        next_states = step_inputs['next_states']
        rewards = step_inputs['rewards']
        dones = step_inputs['dones']
        G = step_inputs['G']

        N, T = states.shape[:2]

        high_states = self.policy.prior_policy.state_encoder(states.view(N * T, -1)).view(N, T, -1)
        high_next_states = self.policy.prior_policy.state_encoder(next_states.view(N * T, -1)).view(N, T, -1)
        skills = step_inputs['actions']
        
        high_state = high_states[:, 0]
        consistency_loss, reward_loss, value_loss = 0, 0, 0
        rollout_high_states = []

        for t in range(T):
            # ---------- value prediction ---------- #
            # Q-value
            q1, q2 = [qf(high_state, step_inputs['actions'][:, t]) for qf in self.qfs]
            target_q_inputs = dict(
                next_states = next_states[:, t],
                G = G,
                q_next_states = high_next_states[:, t],
                rewards = rewards[:, t],
                dones = dones[:, t]
            )
            # target Q
            with torch.no_grad():
                rwd_term, ent_term, entropy_term = self.compute_target_q(target_q_inputs)
                target_qs = rwd_term + ent_term


            # ---------- value function, state consistency ---------- #
            rollout_inputs = dict(
                states = high_state,
                actions = skills[:, t]
            )
            outputs = self.policy.rollout_latent(rollout_inputs)
            # next_state_pred
            high_next_state, rewards_pred = outputs['next_states'], outputs['rewards_pred']
        
            # discounted 
            rho = (self.rho ** t)
            consistency_loss += rho * F.mse_loss(high_next_state, high_next_states[:, t])
            reward_loss += rho * F.mse_loss(rewards_pred, rewards[:, t])
            value_loss += rho * (F.mse_loss(q1, target_qs) + F.mse_loss(q2, target_qs))


            rollout_high_states.append(high_state.clone().detach())
            high_state = high_next_state


        total_loss = (consistency_loss * 2 + reward_loss * 0.5 + value_loss * 0.1) / T

        for qf_optim in self.qf_optims:
            qf_optim.zero_grad()
        self.others_optim.zero_grad()

        total_loss.backward()

        for qf_optim in self.qf_optims:
            qf_optim.step()
        self.others_optim.step()

        # qf_losses = [value_loss]  

        
        
        return rollout_high_states


    def update_policy(self, step_inputs):

        # policy loss 
        rollout_high_states = step_inputs['rollout_high_states']
		# Loss is a weighted sum of Q-values
        policy_loss = 0
        q_values = 0
        entropy_terms = 0
    
        states = step_inputs['states']

        for t, high_state in enumerate(rollout_high_states):
            policy_inputs = dict(
                # states, G
                states = states[:, t],
                high_state = high_state,
                G = step_inputs['G']
            )
        
            policy_inputs['dist'] = self.policy.dist(policy_inputs, latent = True)['policy_skill'] # prior policy mode.
            policy_inputs['policy_actions'] = policy_inputs['dist'].rsample() 
            entropy_term, prior_dist = self.entropy(policy_inputs, kl_clip= False) # policy의 dist로는 gradient 전파함 .
            min_qs = torch.min(*[qf(high_state, policy_inputs['policy_actions']) for qf in self.qfs])

            q_values += min_qs.clone().detach().mean(0).item()
            entropy_terms += entropy_term.clone().detach().mean().item() 


            policy_loss += (- min_qs + self.alpha * entropy_term).mean() * (self.rho ** t)
            
            

        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results = {}
    

        results['policy_loss'] = policy_loss.item()
        results['kl'] = entropy_terms # if self.prior_policy is not None else - entropy_term.mean()
        results['Q-value'] = q_values

        update_moving_average(self.target_qfs, self.qfs, self.tau)
        self.policy.prior_policy.soft_update()
        
        return results
        

    def warmup_Q(self, step_inputs):
        # self.train()
        # self.policy.inverse_dynamics..eval()

        batch = self.buffer.sample(step_inputs['batch_size']).to(self.device)
        self.episode = step_inputs['episode']

        with torch.no_grad():
            states = prep_state(batch.states, self.device)
            next_states = prep_state(batch.next_states, self.device)
            # step_inputs['batch'] = batch
            step_inputs['rewards'] = batch.rewards
            step_inputs['dones'] = batch.dones
            step_inputs['states'] = states
            step_inputs['next_states'] = next_states
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['batch_size'], 1).cuda()
            step_inputs['raw_states'] = states
            step_inputs['raw_next_states'] = next_states

            step_inputs['done'] = True

   
            step_inputs['actions'] = prep_state(batch.actions, self.device) # high-actions


        stat = {}

        for _ in range(self.q_warmup_steps):
            q_results = self.update_Q_models(step_inputs)
                        
        # for k, v in q_results.items():
        #     stat[k] = v 

        return stat