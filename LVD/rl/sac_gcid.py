
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..modules.base import BaseModule
from ..contrib.simpl.math import clipped_kl, inverse_softplus
from ..utils import prep_state, nll_dist, kl_annealing, get_dist, AverageMeter
from ..contrib.momentum_encode import update_moving_average
from ..contrib.dists import *


import datetime

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

        
        # Finetune
        # finetune : subgoal generator
        self.policy_optim = torch.optim.Adam(
            [
                { "params" : self.policy.inverse_dynamics.subgoal_generator.parameters()}, # 보상최대화하는 subgoal 뽑기. 
            ],
            lr = self.policy_lr # 낮추면 잘 안됨. 왜? 
        )

        # finetune : inverse dynamics, dynamics
        param_groups = [
            { "params" : self.policy.inverse_dynamics.inverse_dynamics.parameters()},
            { "params" : self.policy.inverse_dynamics.dynamics.parameters()}

        ]
        
        # if self.policy.inverse_dynamics.dynamics is not None:
        #     param_groups.append({ "params" : self.policy.inverse_dynamics.dynamics.parameters()})
                
        self.consistency_optim = torch.optim.Adam(
            param_groups,
            lr= 1e-8 # 마지막 lr 
        )

        self.consistency_scheduler = ReduceLROnPlateau(
            optimizer = self.consistency_optim,
            factor = 0.5,
            patience = 10, # 4 epsisode
            verbose= True
        )

        self.consistency_meter = AverageMeter()

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

        self.save_prev_module()


        self.sample_time_logger = AverageMeter()

    def save_prev_module(self, target = "all"):
        self.prev_f = copy.deepcopy(self.policy.inverse_dynamics.subgoal_generator).requires_grad_(False)
        self.prev_invD = copy.deepcopy(self.policy.inverse_dynamics.inverse_dynamics).requires_grad_(False)
        self.prev_D = copy.deepcopy(self.policy.inverse_dynamics.dynamics).requires_grad_(False)

    @property
    def target_kl(self):
        return kl_annealing(self.episode, self.target_kl_start, self.target_kl_end, rate = self.kl_decay)

    @property
    def alpha(self):
        return F.softplus(self.pre_alpha)
        
    def entropy(self, inputs, kl_clip = False):
        with torch.no_grad():
            if self.gcprior:
                prior_dists = self.prior_policy(inputs, "eval")['inverse_D']
            else:
                # prior_dists = self.prior_policy(inputs, "eval")['prior']
                prior_dists = self.policy.inverse_dynamics(inputs, "prior")['prior']
                # self.policy.inverse_dynamics.state_encoder()

        if kl_clip:                
            entropy = clipped_kl(inputs['dists'], prior_dists, clip = self.kl_clip)
        else:
            entropy = torch_dist.kl_divergence(inputs['dists'], prior_dists)
        return entropy, prior_dists 
    
    def expand_G(self, step_inputs):
        if step_inputs['states'].shape[0] != step_inputs['G'].shape[0]:
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['states'].shape[0], 1)
        return step_inputs
    
    def grad_clip(self, optimizer):
        if self.n_step < self.init_grad_clip_step:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.init_grad_clip) 

    def step(self, step_inputs):
        self.train()
        self.policy.inverse_dynamics.state_encoder.eval()

        batch = self.buffer.sample(step_inputs['batch_size']).to(self.device)
        self.episode = step_inputs['episode']

        with torch.no_grad():
            states = prep_state(batch.states, self.device)
            next_states = prep_state(batch.next_states, self.device)
            # step_inputs['batch'] = batch
            step_inputs['rewards'] = batch.rewards
            step_inputs['dones'] = batch.dones
            step_inputs['states'] = states #prep_state(batch.states, self.device)
        
            step_inputs['__states__'] = states # prep_state(batch.states, self.device)
            step_inputs['next_states'] = next_states
            step_inputs['__next_states__'] = next_states 

            step_inputs['q_state'] = self.policy.inverse_dynamics.state_encoder(states)
            step_inputs['q_next_state'] = self.policy.inverse_dynamics.state_encoder(next_states)

            step_inputs['done'] = True
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['batch_size'], 1)

            if self.tanh:
                # 요녀석들은 전부 evaluation mode에서 뽑아낸 action들임. 
                step_inputs['actions_normal'] = prep_state(batch.actions[:, :10], self.device) # high-actions
                step_inputs['actions'] = prep_state(batch.actions[:, 10:20], self.device) # high-actions
                step_inputs['loc'] = prep_state(batch.actions[:, 20:30], self.device) # for metric
                step_inputs['scale'] = prep_state(batch.actions[:, 30:], self.device) # for metric


            else:
                step_inputs['actions'] = prep_state(batch.actions[:, :10], self.device) # high-actions
                step_inputs['loc'] = prep_state(batch.actions[:, 10:20], self.device) # for metric
                step_inputs['scale'] = prep_state(batch.actions[:, 20:30], self.device) # for metric
                step_inputs['actions_normal'] = None


        self.n_step += 1

        return self._step(step_inputs)

    def _step(self, step_inputs):
        stat = {}

        # ------------------- Inverse Dynamics & Dynamics ------------------- # 
        policy_finetune_results = self.policy_finetune(step_inputs)

        for k, v in policy_finetune_results.items():
            stat[k] = v 
        
        # ------------------- SAC ------------------- # 
        # ------------------- Q-functions ------------------- #
        # if step_inputs['episode'] < self.q_warmup: 
        #     for _ in range(self.q_weight):
        #         q_results = self.update_qs(step_inputs)
        #         subgoal_results = self.update_subgoal_generator(step_inputs)

        # else:
        #     q_results = self.update_qs(step_inputs)
        #     subgoal_results = self.update_subgoal_generator(steㅟp_inputs)
            
        q_results = self.update_qs(step_inputs)
        subgoal_results = self.update_subgoal_generator(step_inputs)

        for k, v in q_results.items():
            stat[k] = v 

        for k, v in subgoal_results.items():
            stat[k] = v 

        # ------------------- Logging ------------------- # 
        stat['target_kl'] = self.target_kl

        # ------------------- Alpha ------------------- # 
        alpha_results = self.update_alpha(stat['kl'])
        for k, v in alpha_results.items():
            stat[k] = v 

        # prior 도 바뀌어야 함. 
        # 하긴 해야되는데 한 100번에 한번쯤? 
        # update_moving_average(self.prior_policy, self.policy.inverse_dynamics, 1e-5) # hard update 
            
        # Q-function update ratio 
        # stat['delta_Q'] = self.calc_update_ratio('Q')
        stat['delta_f'] = self.calc_update_ratio('f')
        stat['delta_invD'] = self.calc_update_ratio('invD')
        stat['delta_D'] = self.calc_update_ratio('D')

        # if self.n_step % 256 == 0:
        #     self.save_prev_module()
        return stat

    def compute_target_q(self, step_inputs):
        policy_inputs = dict(
            states = step_inputs['__next_states__'].clone(),
            G = step_inputs['G']
        )
        
        policy_inputs['dists'] = self.policy.dist(policy_inputs, "eval")['inverse_D']
        actions = policy_inputs['dists'].sample()

        # calculate entropy term
        entropy_term, prior_dists = self.entropy(policy_inputs, kl_clip= True) 
        if self.use_hidden:
            min_qs = torch.min(*[target_qf(step_inputs['q_next_state'], actions) for target_qf in self.target_qfs])
        else:
            min_qs = torch.min(*[target_qf(step_inputs['next_states'], actions) for target_qf in self.target_qfs])



        if self.alpha < 1e-12:
            soft_qs = min_qs
        else:
            soft_qs = min_qs - self.alpha*entropy_term

        rwd_term = step_inputs['rewards'].cuda()
        ent_term = (1 - step_inputs['dones'].cuda())*self.discount*soft_qs

        return rwd_term, ent_term, entropy_term




    def policy_finetune(self, step_inputs):
        outputs = self.policy.finetune(step_inputs)
        results = {}

        # finetune inverse dynamics
        if self.tanh:
            iD_loss = nll_dist(
                step_inputs['actions'],
                outputs['inverse_D'], # distributions to optimize
                step_inputs['actions_normal'],
                tanh = self.tanh
            ).mean() 
        else:
            iD_loss = nll_dist(
                step_inputs['actions'],
                outputs['inverse_D'], # distributions to optimize
                step_inputs['actions_normal'],
                tanh = self.tanh
            ).mean() 
    
        # finetune dynamics
        D_loss = F.mse_loss(
            outputs['subgoal'],
            outputs['subgoal_target'], # distributions to optimize
        ).mean()

        results['D'] = D_loss.item()
        loss = iD_loss + D_loss



        # if finetune, update network
        # if not, just logging the results
        if self.finetune:
            self.consistency_optim.zero_grad()
            loss.backward()
            self.grad_clip(self.policy_optim)
            self.consistency_optim.step()
            self.policy.soft_update()

        # metrics
        with torch.no_grad():
            dist = get_dist(step_inputs['loc'], step_inputs['scale'].log(), tanh = self.tanh)
            iD_kld = torch_dist.kl_divergence(dist, outputs['inverse_D']).mean() # KL (post || prior)
        self.consistency_meter.update(iD_kld.item(), step_inputs['batch_size'])

        # scheduler step
        if (self.n_step + 1) % 256 == 0:
            self.consistency_scheduler.step(self.consistency_meter.avg)
            self.consistency_meter.reset()
        
        # log 
        results['invD'] = iD_loss.item()
        results['invD_KL'] = iD_kld.item()
        
        return results


    def update_qs(self, step_inputs):
        with torch.no_grad():
            rwd_term, ent_term, entropy_term = self.compute_target_q(step_inputs)
            target_qs = rwd_term + ent_term

        # Q-function의 parameter 업데이트 양을 조사해보자. 
        # 퍼센티지로? ㄴㄴ 크기도 아닌데 
        
        

        qf_losses = []  
        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            if self.use_hidden:
                qs = qf(step_inputs['q_state'], step_inputs['actions'])
            else:
                qs = qf(step_inputs['states'], step_inputs['actions'])

            qf_loss = (qs - target_qs).pow(2).mean()


            if not self.optimal_Q:
                qf_optim.zero_grad()
                qf_loss.backward()
                qf_optim.step()

            qf_losses.append(qf_loss)
        
        if not self.optimal_Q:
            update_moving_average(self.target_qfs, self.qfs, self.tau)

        results = {}
        results['qf_loss'] = torch.stack(qf_losses).mean()
        results['target_Q'] = target_qs.mean()
        results['rwd_term'] = rwd_term.mean()
        results['entropy_term'] = ent_term.mean()

        return results


    def update_subgoal_generator(self, step_inputs):
        results = {}
        dist_out = self.policy.dist(step_inputs, "eval")
        step_inputs['dists'] = dist_out['inverse_D'] # 
        step_inputs['policy_actions'] = step_inputs['dists'].rsample() 

        entropy_term, prior_dists = self.entropy(step_inputs, kl_clip= False) # policy의 dist로는 gradient 전파함 .
        if self.use_hidden:
            min_qs = torch.min(*[qf(step_inputs['q_state'], step_inputs['policy_actions']) for qf in self.qfs])
        else:
            min_qs = torch.min(*[qf(step_inputs['states'], step_inputs['policy_actions']) for qf in self.qfs])


        if self.alpha < 1e-12:
            policy_loss = - min_qs.mean()
        else:
            policy_loss = (- min_qs + self.alpha * entropy_term).mean()

        if self.policy.inverse_dynamics.dynamics is not None:
            # f는 make sense한 subgoal을 줘야 한다. 이건데 이게 GT subgoal이 아니긴 하거든
            dynamics_loss = F.mse_loss(dist_out['subgoal'], dist_out['subgoal_target'])
            policy_loss += dynamics_loss
            results['state_consistency_f'] = dynamics_loss.item() 


        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.grad_clip(self.policy_optim)
        self.policy_optim.step()

        results['policy_loss'] = policy_loss.item()
        results['kl'] = entropy_term.mean().item() # if self.prior_policy is not None else - entropy_term.mean()
        if self.tanh:
            results['mean_policy_scale'] = step_inputs['dists']._normal.base_dist.scale.abs().mean().item() 
            results['mean_prior_scale'] = prior_dists._normal.base_dist.scale.abs().mean().item()
        else:
            results['mean_policy_scale'] = step_inputs['dists'].base_dist.scale.abs().mean().item() 
            results['mean_prior_scale'] = prior_dists.base_dist.scale.abs().mean().item()

        results['Q-value'] = min_qs.mean(0).item()

        return results

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

    def calc_update_ratio(self, module_name):

        if module_name == "Q":
            prev_module = self.prev_qfs
            module = self.qfs
        elif module_name == "f":
            prev_module = self.prev_f
            module = self.policy.inverse_dynamics.subgoal_generator
        elif module_name == "invD":
            prev_module = self.prev_invD
            module = self.policy.inverse_dynamics.inverse_dynamics
        else:
            prev_module = self.prev_D
            module =  self.policy.inverse_dynamics.dynamics

        with torch.no_grad():
            update_rate = []
            for (prev_p, p) in zip(prev_module.parameters(), module.parameters()):

                delta = (p - prev_p).norm()
                prev_norm =  prev_p.norm()
                
                # delta = (p - prev_p).abs()
                # prev_norm = prev_p.abs()

                if prev_norm.item() != 0:
                    update_rate.append((delta / prev_norm).item())

        return np.mean(update_rate)
    

    def warmup_Q(self, step_inputs):
        self.train()
        self.policy.inverse_dynamics.state_encoder.eval()

        batch = self.buffer.sample(step_inputs['batch_size']).to(self.device)
        self.episode = step_inputs['episode']

        with torch.no_grad():
            states = prep_state(batch.states, self.device)
            next_states = prep_state(batch.next_states, self.device)
            # step_inputs['batch'] = batch
            step_inputs['rewards'] = batch.rewards
            step_inputs['dones'] = batch.dones
            step_inputs['states'] = states #prep_state(batch.states, self.device)
        
            step_inputs['__states__'] = states # prep_state(batch.states, self.device)
            step_inputs['next_states'] = next_states
            step_inputs['__next_states__'] = next_states 

            step_inputs['q_state'] = self.policy.inverse_dynamics.state_encoder(states)
            step_inputs['q_next_state'] = self.policy.inverse_dynamics.state_encoder(next_states)

            step_inputs['done'] = True
            step_inputs['G'] = step_inputs['G'].repeat(step_inputs['batch_size'], 1)

            if self.tanh:
                # 요녀석들은 전부 evaluation mode에서 뽑아낸 action들임. 
                step_inputs['actions_normal'] = prep_state(batch.actions[:, :10], self.device) # high-actions
                step_inputs['actions'] = prep_state(batch.actions[:, 10:20], self.device) # high-actions
                step_inputs['loc'] = prep_state(batch.actions[:, 20:30], self.device) # for metric
                step_inputs['scale'] = prep_state(batch.actions[:, 30:], self.device) # for metric


            else:
                step_inputs['actions'] = prep_state(batch.actions[:, :10], self.device) # high-actions
                step_inputs['loc'] = prep_state(batch.actions[:, 10:20], self.device) # for metric
                step_inputs['scale'] = prep_state(batch.actions[:, 20:30], self.device) # for metric
                step_inputs['actions_normal'] = None

        stat = {}

        for _ in range(self.q_warmup_steps):
            q_results = self.update_qs(step_inputs)
                        
        for k, v in q_results.items():
            stat[k] = v 

        return stat