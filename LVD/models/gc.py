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


class GoalConditioned_Model(BaseModule):
    """
    """
    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = False
        self.tanh = True
        self.step = 0
        self.Hsteps = 10
        self.KL_threshold = 0
        bias = True
        dropout = 0


        norm_cls = nn.LayerNorm
        act_cls = nn.ReLU

        self.env = gym.make("simpl-kitchen-v0")

        self.qpo_dim = self.env.robot.n_jnt + self.env.robot.n_obj
        self.qv = self.env.init_qvel[:].copy()
        self.joint_learn = self.fe_path == ''


        if not self.joint_learn:
            ae = torch.load(self.fe_path)['model'].eval()
    
            state_encoder = ae.state_encoder.eval()
            state_decoder = ae.state_decoder.eval()

            state_encoder.requires_grad_(False)
            state_decoder.requires_grad_(False)

        else:
            # state encoder
            state_encoder_config = edict(
                n_blocks = self.n_processing_layers,
                in_feature = self.state_dim, # state_dim + latent_dim 
                hidden_dim = self.hidden_dim, 
                out_dim = self.latent_state_dim, # when variational inference
                norm_cls =  norm_cls,
                act_cls = act_cls, #nn.LeakyReLU,
                block_cls = LinearBlock,
                bias = bias,
                dropout = dropout
            )

            # state decoder
            state_decoder_config = edict(
                n_blocks = self.n_processing_layers,#self.n_processing_layers,
                in_feature = self.latent_state_dim, # state_dim + latent_dim 
                hidden_dim = self.hidden_dim, 
                out_dim = self.state_dim,
                norm_cls = norm_cls,
                act_cls = act_cls, #nn.LeakyReLU,
                block_cls = LinearBlock,
                bias = bias,
                dropout = dropout
            )

            state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
            state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))


        # ----------------- SUBMODULES ----------------- #

        ## ----------------- Configurations ----------------- ##

        ### ----------------- prior modules ----------------- ###

        prior_config = edict(
            n_blocks = self.n_processing_layers, #self.n_processing_layers,
            in_feature =  self.latent_state_dim, # state_dim + latent_dim 
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
        
        inverse_dynamics_config = edict(
            n_blocks = self.n_processing_layers, 
            in_feature = self.latent_state_dim * 2 , # state dim 
            hidden_dim = self.hidden_dim, # 128
            out_dim = self.latent_dim * 2, # * 2 when variational inference
            norm_cls = norm_cls,
            act_cls = act_cls,
            block_cls = LinearBlock,
            bias = bias,
            dropout = dropout
        )


        dynamics_config = edict(
            n_blocks =  self.n_processing_layers,# 
            in_feature =  self.latent_state_dim + self.latent_dim,  
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_state_dim,
            norm_cls = norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
            tanh = False,
            bias = bias,
            dropout = dropout    
        )

        subgoal_generator_config = edict(
            n_blocks =  3, #self.n_processing_layers,# 
            in_feature = self.latent_state_dim + self.n_env, # state_dim + latent_dim 
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_state_dim,
            norm_cls = norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
            tanh = False,
            bias = bias,
            dropout = dropout    
        )

        ### ----------------- posterior modules ----------------- ###
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
        )

        decoder_config = edict(
            n_blocks = self.n_processing_layers, #self.n_processing_layers,
            state_dim = self.state_dim,
            # state_dim = self.latent_state_dim,
            z_dim = self.latent_dim, 
            in_feature = self.latent_dim + self.state_dim, # state_dim + latent_dim 
            # in_feature = self.latent_state_dim + self.latent_dim,
            hidden_dim = self.hidden_dim, 
            out_dim = self.action_dim,
            norm_cls = nn.BatchNorm1d,
            act_cls = act_cls,
            block_cls = LinearBlock,
            bias = bias,
            dropout = dropout           
        )

        ## ----------------- Builds ----------------- ##

        ### ----------------- Skill Prior Modules ----------------- ###


        inverse_dynamics = InverseDynamicsMLP(Linear_Config(inverse_dynamics_config))
        subgoal_generator = SequentialBuilder(Linear_Config(subgoal_generator_config))
        prior = SequentialBuilder(Linear_Config(prior_config))
        dynamics = SequentialBuilder(Linear_Config(dynamics_config))
        
        prior_wrapper_cls = PRIOR_WRAPPERS['gc']

        self.inverse_dynamics_policy = prior_wrapper_cls(
            # components  
            prior_policy = prior,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            subgoal_generator = subgoal_generator,
            dynamics = dynamics,
            # architecture parameters
            ema_update = True,
            tanh = self.tanh,
            scale = self.scale,
            joint_learn = self.joint_learn,
            sample_interval = self.sample_interval
        )

        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.inverse_dynamics_policy.prior_policy.parameters(), lr = self.lr ),
                # "metric" : "Prior_S"
                "metric" : "Prior_GC"                
            }, 
            "skill_enc_dec" : {
                "optimizer" : RAdam( [
                    {"params" : self.skill_encoder.parameters()},
                    {"params" : self.skill_decoder.parameters()},
                ], lr = self.lr ),
                "metric" : "Rec_skill"
                # "metric" : "skill_metric"
            }, 
            "invD" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.inverse_dynamics.parameters()},
                ], lr = self.lr ),
                # "metric" : "Prior_GC"
                "metric" : "Prior_GC"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.dynamics.parameters()},
                ], lr = self.lr ),
                # "metric" : "D"
                "metric" : "Prior_GC"

            }, 
            "f" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.subgoal_generator.parameters()},
                ], lr = self.lr ),
                # "metric" : "F_skill_GT"
                "metric" : "Prior_GC"
            }, 
        }

        if self.joint_learn:
            state_param_groups = [
                {'params' : self.inverse_dynamics_policy.state_encoder.parameters()},
                {'params' : self.inverse_dynamics_policy.state_decoder.parameters()},
            ]

            self.optimizers['state'] = {
                "optimizer" : RAdam(
                    state_param_groups,            
                    lr = model_config.lr
                ),
                "metric" : "state_recon"
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
            self.loss_dict['F_skill_GT'] = self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']).mean()
            # KL (post || state-conditioned prior)
            self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean()
            
            # KL (post || goal-conditioned policy)
            self.loss_dict['Prior_GC']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']).mean()

            self.loss_dict['Prior_GC_S']  = self.loss_fn('reg')(self.outputs['invD'], self.outputs['prior']).mean()


                
            if "subgoal_div" in self.outputs:
                pass

        self.loss_dict['metric'] = self.loss_dict['Prior_GC']

    def forward(self, states, actions, G):
        
        N, T, _ = states.shape

        inputs = dict(
            states = states,
            G = G
        )

        # skill prior
        self.outputs =  self.inverse_dynamics_policy(inputs, "train")

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
        # decode_inputs = self.dec_input(self.outputs['hts'], z, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        
        # Outputs
        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['skill_hat'] = skill_hat
        self.outputs['skill'] = actions

    def compute_loss(self, skill):

        # ----------- Skill Recon & Regularization -------------- # 
        recon = self.loss_fn('recon')(self.outputs['skill_hat'], skill)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        

        # ----------- State/Goal Conditioned Prior -------------- # 
        if self.tanh:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], # distributions to optimize
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

            invD_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['invD'], 
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean() 
            # prior = nll_dist(
            #     self.outputs['prior'], # distributions to optimize
            #     self.outputs['z'],
            #     self.outputs['z_normal'],
            #     tanh = self.tanh
            # ).mean()

            # invD_loss = nll_dist2(
            #     self.outputs['invD'], 
            #     self.outputs['z'],
            #     self.outputs['z_normal'],
            #     tanh = self.tanh
            # ).mean() 


        else:
            prior = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior'], 
                tanh = self.tanh
            ).mean()

            # inverse dynamics policy
            invD_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['invD'], 
                tanh = self.tanh
            ).mean()          

        # ----------- Dynamics -------------- # 
        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target']
        )         
        

        # ----------- subgoal generator -------------- # 
        # intrinsic reward for subgoal-reaching 
        r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_target'])
        r_int_D = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_target'])
        r_int = r_int_f + r_int_D

        reg_term = self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']).mean()

        F_loss = r_int + reg_term 


        loss = recon + reg * self.reg_beta + prior + invD_loss + D_loss + F_loss

        if self.joint_learn:
            recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 
            # z_tilde = self.outputs['states_repr']
            # z = self.outputs['states_fixed_dist'].sample()
            # mmd_loss = compute_mmd(z_tilde, z)
            loss = loss + recon_state #+ mmd_loss

            
        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "invD" : invD_loss.item(),
            "D" : D_loss.item(),
            "F" : F_loss.item(),
            "F_state" : r_int.item(),
            "r_int_f" : r_int_f.item(),
            "r_int_D" : r_int_D.item(),
            "F_skill_kld" : reg_term.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
        }       

        if self.joint_learn:
            self.loss_dict['recon_state'] = recon_state.item()

        return loss

    def __main_network__(self, states, actions, G, validate = False):

        self(states, actions, G)
        loss = self.compute_loss(actions)


        self.loss_dict['KL_threshold'] = self.KL_threshold


        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()

            
    def optimize(self, batch, e):
        # inputs & targets       
        states, actions, G = batch.values()
        states, actions, G = states.cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G)

        with torch.no_grad():
            self.get_metrics()
            self.inverse_dynamics_policy.soft_update()
        
        return self.loss_dict
    
    
    def validate(self, batch, e):
        # inputs & targets          
        states, actions, G = batch.values()
        states, actions, G = states.cuda(), actions.cuda(), G.cuda()

        self.__main_network__(states, actions, G, validate= True)
        self.get_metrics()
        self.step += 1

        return self.loss_dict