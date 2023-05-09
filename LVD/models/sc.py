import torch
import torch.nn as nn
from torch.optim import *
import cv2
import gym
import d4rl


from ..modules.priors import *
from ..configs.build import *
from ..modules.base import *
from ..modules.subnetworks import *
from ..utils import *
from ..envs.kitchen import KitchenEnv_GC
from ..contrib.simpl.env.kitchen import KitchenTask


# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 

class StateConditioned_Model(BaseModule):
    """
    """
    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = True
        self.step = 0
        self.Hsteps = 10
        norm_cls = nn.LayerNorm
        act_cls = nn.Mish
        bias = True
        dropout = 0

        self.joint_learn = True


        state_encoder_config = edict(
            n_blocks = self.n_Layers,
            in_feature = self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.latent_state_dim * 2, 
            # hidden_dim = self.hidden_dim, 
            out_dim = self.latent_state_dim, # when variational inference
            norm_cls =  norm_cls,
            # norm_cls =  None,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            bias = bias,
            dropout = dropout
        )

        # state decoder
        state_decoder_config = edict(
            n_blocks = self.n_Layers,#self.n_processing_layers,
            in_feature = self.latent_state_dim, # state_dim + latent_dim 
            hidden_dim = self.latent_state_dim * 2, 
            # hidden_dim = self.hidden_dim, 
            out_dim = self.state_dim,
            norm_cls = norm_cls,
            # norm_cls =  None,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            bias = bias,
            dropout = dropout
        )




        ## skill prior module

        prior_config = edict(
            n_blocks = self.n_Layers, #self.n_processing_layers,
            in_feature =  self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2,
            norm_cls = norm_cls,
            act_cls = act_cls,
            block_cls = LinearBlock,
            true = True,
            tanh = self.tanh,
            bias = bias,
            dropout = dropout 
        )

        encoder_config = edict(
            in_feature = self.action_dim + self.state_dim,
            # in_feature = self.action_dim + self.latent_state_dim,
            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim * 2,
            n_blocks = 1,
            bias = False,
            batch_first = True,
            dropout = 0,
            linear_cls = LinearBlock,
            rnn_cls = nn.LSTM,
            act_cls = act_cls,
            # norm_cls = norm_cls,
            true = True,
            false = False
        )

        decoder_config = edict(
            n_blocks = self.n_Layers, #self.n_processing_layers,
            # state_dim = self.latent_state_dim,
            z_dim = self.latent_dim, 
            state_dim = self.state_dim,
            in_feature = self.latent_dim + self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.action_dim,
            norm_cls = nn.BatchNorm1d,
            act_cls = act_cls,
            block_cls = LinearBlock,
            bias = bias,
            dropout = dropout           
        )

        highlevel_policy_config = edict(
            n_blocks =  self.n_Layers,# 
            in_feature = self.state_dim + self.n_goal, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2,
            norm_cls = norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
            tanh = False,
            bias = bias,
            dropout = dropout    
        )




        prior = SequentialBuilder(Linear_Config(prior_config))
        highlevel_policy = SequentialBuilder(Linear_Config(highlevel_policy_config))
        
        
        self.skill_prior = PRIOR_WRAPPERS['sc'](
            prior_policy = prior,
            highlevel_policy = highlevel_policy,
        )

        ## skill encoder
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))

        ## closed-loop skill decoder
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))

        # optimizer
        # self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.skill_prior.parameters(), lr = self.lr ),
                "metric" : "Prior_S"
            }, 
            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                # "metric" : "Rec_skill"
                "metric" : "skill_metric"
            }
        }


        # Losses
        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
            'prior_metric' : ["kld", torch_dist.kl_divergence],
        }


        self.outputs = {}
        self.loss_dict = {}

        self.step = 0


    @staticmethod
    def dec_input(states, z, steps, detach = False):
        if detach:
            z = z.clone().detach()
        return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def loss_fn(self, key, index = 1):
        """
        Calcalate loss by loss name 
        """
        return self.loss_fns[key][index]
    
    def grad_clip(self, optimizer):
        if self.step < self.init_grad_clip_step:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.init_grad_clip) 

    
    def get_metrics(self):
        """
        Metrics
        """
        # ----------- Metrics ----------- #
        with torch.no_grad():
            # KL (post || state-conditioned prior)
            self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean().item()
            self.loss_dict['Policy_loss']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['policy_skill']).mean().item()

            # dummy metric 
            self.loss_dict['metric'] = self.loss_dict['Prior_S']
            

    def forward(self, states, actions, G):

        

        inputs = dict(
            states = states,
            G = G
        )

        # skill prior
        self.outputs =  self.skill_prior(inputs)

        # skill Encoder 
        enc_inputs = torch.cat( (actions, states.clone()[:,:-1]), dim = -1)
        q = self.skill_encoder(enc_inputs)[:, -1]
        q_clone = q.clone().detach()
        q_clone.requires_grad = False
        
        post = get_dist(q, tanh = self.tanh)
        post_detach = get_dist(q_clone, tanh = self.tanh)
        fixed_dist = get_fixed_dist(q_clone, tanh = self.tanh)

        if self.tanh:
            z_normal, z = post.rsample_with_pre_tanh_value()
            self.outputs['z'] = z.clone().detach()
            self.outputs['z_normal'] = z_normal.clone().detach()
        else:
            z = post.rsample()
            self.outputs['z'] = z.clone().detach()
        
        # Skill Decoder 
        decode_inputs = self.dec_input(states.clone(), z, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        
        # Outputs
        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['skill_hat'] = skill_hat
        self.outputs['skill'] = actions


    def compute_loss(self, skill):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(self.outputs['skill_hat'], skill)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        
        if self.tanh:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

            policy_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['policy_skill'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

        else:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], 
                tanh = self.tanh
            ).mean()

            policy_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['policy_skill'], # distributions to optimize
                tanh = self.tanh
            ).mean()


        # with torch.no_grad()
        #     prior_kld  = self.get_loss_fn('reg')(outputs['post_detach'], outputs['prior_detach']).mean() 

        #     non_branching_indices = state_labels[:, 0] == 1
        #     non_branching_indices = non_branching_indices.cpu().numpy()

        #     prior_mean = outputs['prior_detach'].base_dist.loc.cpu().numpy()#[non_branching_indices].mean()
        #     prior_std = outputs['prior_detach'].base_dist.scale.cpu().numpy()#[non_branching_indices].mean()
        #     prior_mean = prior_mean[non_branching_indices].mean()
        #     prior_std = prior_std[non_branching_indices].mean()



        # ----------- Add -------------- # 
        loss = recon + reg * self.reg_beta  + prior + policy_loss

        if "states_hat" in self.outputs.keys():
            recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 

            z_tilde = self.outputs['states_repr']
            z = self.outputs['states_fixed_dist']
            mmd_loss = compute_mmd(z_tilde, z)

            loss = loss + recon_state + mmd_loss


        self.loss_dict = {           
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Prior" : prior.item(),
            "Policy_loss" : policy_loss.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
            "recon_state" : recon_state.item() if "states_hat" in self.outputs.keys() else 0,
            "mmd_loss" : mmd_loss.item() if "states_hat" in self.outputs.keys() else 0
        }       


        return loss
    
    def __main_network__(self, states, actions, G, validate = False):
        self(states, actions, G)
        loss = self.compute_loss(actions)

        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()



    def optimize(self, batch, e):
        # inputs & targets       

        states, actions, G = batch.values()
        states, actions, G = states.float().cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G)

        with torch.no_grad():
            self.get_metrics()

        return self.loss_dict
    
    def validate(self, batch, e):
        # inputs & targets       

        states, actions, G = batch.values()
        states, actions, G = states.float().cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G, validate= True)

        with torch.no_grad():
            self.get_metrics()

        return self.loss_dict