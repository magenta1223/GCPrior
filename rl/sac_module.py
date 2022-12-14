
import copy
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from proposed.modules.base import BaseModule
from proposed.contrib.simpl.math import clipped_kl, inverse_softplus
from proposed.utils import prep_state, nll_dist, get_fixed_dist
from easydict import EasyDict as edict

class SAC(BaseModule):
    """
    Do not edit
    """
    def __init__(self, policy, prior_policy,

                qfs, buffer,
                discount=0.99, tau=0.005, policy_lr=3e-4, qf_lr=3e-4,
                auto_alpha=True, init_alpha=0.1, alpha_lr=3e-4, target_kl=1,
                kl_clip=20, increasing_alpha=False):
        super().__init__(None)
        
        self.policy = policy
        self.prior_policy = prior_policy

        self.qfs = nn.ModuleList(qfs)
        self.target_qfs = nn.ModuleList([copy.deepcopy(qf) for qf in qfs])
        self.buffer = buffer

        self.discount = discount
        self.tau = tau

        self.policy_optim = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.qf_optims = [torch.optim.Adam(qf.parameters(), lr=qf_lr) for qf in qfs]

        # additional         
        # self.prior_optim = torch.optim.Adam(
        #     [
        #         {"params" : self.prior_policy.state_encoder.parameters()},
        #         {"params" : self.prior_policy.subgoal_generator.parameters()}
        #     ],
        #     lr = 1e-5 
        # )

        
        self.auto_alpha = auto_alpha
        pre_init_alpha = inverse_softplus(init_alpha)
        if auto_alpha is True:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.pre_alpha], lr=alpha_lr)
            self.target_kl = target_kl
        else:
            self.pre_alpha = torch.tensor(pre_init_alpha, dtype=torch.float32)

        self.kl_clip = kl_clip
        self.increasing_alpha = increasing_alpha
        self.n_step = 0
        self.init_grad_clip_step = 1000
        self.init_grad_clip = 0.01

        
    @property
    def alpha(self):
        # return F.softplus(self.pre_alpha)
        return self.pre_alpha.exp()
    
    def to(self, device):
        self.policy.to(device)
        return super().to(device)
    
    def clip_grad(self, param):
        if self.n_step < self.init_grad_clip_step:
            torch.nn.utils.clip_grad_norm_(param, self.init_grad_clip) 
        

    # def entropy(self, dists, states, actions, G):
    def entropy(self, inputs):

        # if self.prior_policy is not None:
        # ???????????? ??????
        # ----- TODO ----- #
        # ?????? ?????? ?????????.
        # states = prep_state(states, "cuda:0")
        # G = prep_state(G, "cuda:0")
        # if states.shape[0] != G.shape[0]:
            # G = G.repeat(states.shape[0], 1)

        # with torch.no_grad():
        # self.prior_policy.finetune()

        self.prior_policy.eval()
        
        prior_dists = self.prior_policy(inputs, "eval").prior # ????????? ?????? ??????????????? no_grad ???????????? ?????????. 
        
        # ---------------- #

        return torch_dist.kl_divergence(inputs.dists, prior_dists).mean() # calculate kl divergence analytically
        # else:
            # return dists.log_prob(actions)
    
    @staticmethod
    def expand_G(step_inputs):
        # G?????? ?????? ????????? 
        # if step_inputs.get("G", False): #gc
        if hasattr(step_inputs, "G"):
            if step_inputs.states.shape[0] != step_inputs.G.shape[0]:
                step_inputs['G'] = step_inputs.G.repeat(step_inputs.states.shape[0], 1)
        return step_inputs

    @staticmethod
    def copy_inputs(dict_inputs):
        """
        Deepcopy method for tensors with gradients.
        """
        # deepcopy_methods = lambda x: x.clone() if isinstance(x, torch.Tensor) 

        def __deepcopy__(x):
            if isinstance(x, torch.Tensor):
                result = x.clone()
            elif isinstance(x, np.ndarray):
                result = x.copy()
            else:
                # dist. ????????? ?????? 
                result = x
            return result


        copied = edict({
            k : __deepcopy__(v) for k, v in dict_inputs.items()
            # tensor > clone
            # ndarray > copy
            # else (torch.Distributions and etc.) > shallow copy. not used for entropy term 
        })        


        return copied



    def step(self, step_inputs):
        stat = {}

        # -----------------prep----------------- # 
        batch = self.buffer.sample(step_inputs.batch_size).to(self.device)

        step_inputs['batch'] = batch

        step_inputs['rewards'] = batch.rewards
        step_inputs['dones'] = batch.dones
        step_inputs['states'] = prep_state(batch.states, self.device)
        step_inputs['__states__'] = prep_state(batch.states, self.device)
        step_inputs['__next_states__'] = prep_state(batch.next_states, self.device)
        # if gc
        step_inputs = self.expand_G(step_inputs)

        # -------------------------------------- # 
        # self.finetune_networks(step_inputs)


        # qfs
        with torch.no_grad():
            target_qs = self.compute_target_q(step_inputs)


        # policy
        step_inputs['dists'] = self.policy.dist(step_inputs, "eval")

        qf_losses = []  
        step_inputs['actions'] = step_inputs['dists'].sample()

        for qf, qf_optim in zip(self.qfs, self.qf_optims):
            qs = qf(step_inputs.__states__, step_inputs.actions)
            qf_loss = (qs - target_qs).pow(2).mean()
            qf_optim.zero_grad()
            qf_loss.backward()
            self.clip_grad(qf.parameters())
            qf_optim.step()
            qf_losses.append(qf_loss)
        self.update_target_qfs()
        
        stat['qf_loss'] = torch.stack(qf_losses).mean()

        # 
        # policy_actions = step_inputs['dists'].rsample() #.clamp(-1,1)
        step_inputs['policy_actions'] = step_inputs['dists'].rsample().clamp(-1,1)


        entropy_term = self.entropy(step_inputs)  # action ???  

        min_qs = torch.min(*[qf(step_inputs.__states__, step_inputs.policy_actions) for qf in self.qfs])
        
        # policy loss & update
        policy_loss = (- min_qs + self.alpha * entropy_term).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.clip_grad(self.policy.parameters())
        self.policy_optim.step()

        # prior policy finetuning 
        # dynamics 
        # state recontruction
        # ? 
        
        
        stat['policy_loss'] = policy_loss
        stat['kl'] = entropy_term.mean() # if self.prior_policy is not None else - entropy_term.mean()
        # stat['mean_policy_scale'] = dists.base_dist.scale.abs().mean()
        stat['mean_policy_scale'] = step_inputs.dists.base_dist.scale.abs().mean() # for tanhnormal


        # alpha
        if self.auto_alpha is True:
            alpha_loss = - (self.pre_alpha * (self.target_kl + entropy_term.detach())).mean()
            if self.increasing_alpha is True:
                alpha_loss = alpha_loss.clamp(-np.inf, 0)
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.clip_grad(self.pre_alpha)

            self.alpha_optim.step()
            
            stat['alpha_loss'] = alpha_loss
            stat['alpha'] = self.alpha
    
        return stat

    def compute_target_q(self, step_inputs):
        """
        step_inputs 
          states : states
          next_states : next_states
          G : G
        """


        _inputs = self.copy_inputs(step_inputs) # deepcopy to prevent reference error

        # get action distributions
        _inputs['states'] = _inputs.__next_states__.clone() # overwrite
        

        _inputs['dists'] = self.policy.dist(_inputs, "eval")
        
        # to tensor 
        _inputs['actions'] = _inputs['dists'].sample().clamp(-1,1)

        # calculate entropy term
        entropy_term = self.entropy(_inputs) 

        min_qs = torch.min(*[target_qf(_inputs.states, _inputs.actions) for target_qf in self.target_qfs])
        soft_qs = min_qs - self.alpha*entropy_term

        return _inputs.rewards.cuda() + (1 - _inputs.dones.cuda())*self.discount*soft_qs

    def update_target_qfs(self):
        """
        TODO : replace with momentum updater 
        """
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            for param, target_param in zip(qf.parameters(), target_qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    def finetune_networks(self, step_inputs):
        """
        SKiMO Dynamics finetuning
        
        - buffer?????? state sequence??? ???????????? 
        - ?????? ?????? T, ???????????? H
        - ?????? T- H + 1 ??? ????????? s_t, s_{t+H} ??? ?????? ??????
        - State Consistency loss 
            - ht : train????????? ?????????
            - h_star : eval????????? ?????????
            - \hat{h}_{t+H} : train????????? ????????? ??????
        
        state, G, H-step ????????? state

        G??? ???????????? ???. state, next_H_state??? buffer?????? ???????????? ???. ?????? ???????????? ????????? ?????????

        """
        H = 10
        
        batch_Hstep = self.buffer.sample_Hstep(step_inputs.batch_size).to(self.device)

        inputs = edict(
            states = batch_Hstep.states,
            G = step_inputs.G,
            next_H_states = batch_Hstep.next_H_states
        )
        

        # learning

        self.prior_policy.train()
        # zero grad
        self.prior_optim.zero_grad()


        result = self.prior_policy(inputs, mode = "finetune")

        # state consistency 
        # latent_consistency_loss = nll_dist(result.h_tH, result.h_tH_hat)
        loss = nll_dist(result.prior_GT, result.prior_hat)
        loss.backward()



        self.prior_optim.step()

        self.prior_policy.ma_state_enc()
        self.prior_policy.eval()
        
        

        
        