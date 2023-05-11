from ..configs.build import *
from ..modules.base import *
from ..modules.subnetworks import *
import torch
import torch.nn as nn
from torch.optim import *
from LVD.utils import *

from torch.cuda.amp import GradScaler

import cv2
import gym
import d4rl
from ..envs.kitchen import KitchenEnv_GC
from ..contrib.simpl.env.kitchen import KitchenTask
from ..modules.priors import *

from ..envs import ENV_TASK
from ..env_vis import RENDER_FUNCS


class GoalConditioned_Diversity_Joint_Model(BaseModule):
    """
    """
    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = False
        # self.tanh = True
        self.step = 0
        self.Hsteps = 10
        self.KL_threshold = 0
        bias = True
        dropout = 0

        norm_cls = nn.LayerNorm
        act_cls = nn.Mish


        env_cls = ENV_TASK[self.env_name]['env_cls']
        configure = ENV_TASK[self.env_name]['cfg']

        if configure is not None:
            self.env = env_cls(**configure)
        else:
            self.env = env_cls()


        self.render_funcs = RENDER_FUNCS[self.env_name]


        self.joint_learn = True


        


        # state encoder
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

        if self.env_name == "maze":
            self.render_period = 32
            # state_encoder = MultiModalEncoder(state_encoder_config)
            # state_decoder = MultiModalDecoder(state_decoder_config)
            
            
            # output_res = 1
            # last_layer = 32
            # self.visual_encoder = nn.Sequential(
            #     nn.Conv2d(1, 8, 3, 2, 1),
            #     nn.BatchNorm2d(8),
            #     act_cls(),
            #     nn.Conv2d(8, 16, 3, 2, 1),
            #     nn.BatchNorm2d(16),
            #     act_cls(),
            #     nn.Conv2d(16, last_layer, 3, 2, 1),                        
            #     nn.AdaptiveAvgPool2d(output_res),
            #     Flatten()
            # )

            # self.enc_state_dim = self.state_dim + self.action_dim + last_layer * (output_res ** 2)
            # self.dec_state_dim = last_layer * (output_res ** 2)
            self.enc_state_dim = self.dec_state_dim = self.state_dim 


        else:
            self.render_period = 4
            self.enc_state_dim = self.dec_state_dim = self.state_dim # + self.action_dim

            # state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
            # state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))

        state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))

        # ----------------- SUBMODULES ----------------- #

        ## ----------------- Configurations ----------------- ##

        ### ----------------- prior modules ----------------- ###

        prior_config = edict(
            # n_blocks = self.n_processing_layers, #self.n_processing_layers,
            n_blocks = self.n_Layers,
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
            n_blocks = self.n_Layers, 
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
            n_blocks =  self.n_Layers,# 
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
            n_blocks =  self.n_Layers,# 
            # in_feature = self.latent_state_dim * 2,  
            # in_feature = self.latent_state_dim + self.n_env, # state_dim + latent_dim 
            in_feature = self.latent_state_dim + self.n_goal, # state_dim + latent_dim 
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
            # in_feature = self.action_dim + self.state_dim,
            in_feature = self.enc_state_dim + self.action_dim,
            # in_feature = self.action_dim + 36 if self.env_name == "maze" else self.action_dim + self.state_dim,

            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim * 2,
            n_blocks = 1,
            bias = True,
            batch_first = True,
            dropout = 0,
            linear_cls = LinearBlock,
            rnn_cls = nn.LSTM,
            act_cls = act_cls,
            norm_cls = norm_cls,
            true = True,
            false = False           
        )

        decoder_config = edict(
            n_blocks = self.n_Layers, #self.n_processing_layers,
            z_dim = self.latent_dim, 
            state_dim = self.dec_state_dim,
            # in_feature =  self.latent_env_dim + self.latent_dim if self.env_name == "maze" else self.state_dim + self.latent_dim, # state_dim + latent_dim 
            # in_feature =  32 * 32 + self.latent_dim if self.env_name == "maze" else self.state_dim + self.latent_dim, # state_dim + latent_dim  latent_state_dim
            # in_feature =  self.latent_state_dim + self.latent_dim if self.env_name == "maze" else self.state_dim + self.latent_dim, # state_dim + latent_dim  latent_state_dim
            # in_feature =  self.state_dim + self.latent_dim, # state_dim + latent_dim  latent_state_dim
            # in_feature =  self.state_dim + self.latent_dim -4 if self.env_name == "maze" else self.state_dim + self.latent_dim, # exclude position for decoder 
            # in_feature = self.latent_dim + self.state_dim -4 if self.env_name == "maze" else self.state_dim + self.latent_dim,
            in_feature = self.latent_dim + self.dec_state_dim,


            hidden_dim = self.hidden_dim, 
            out_dim = self.action_dim,
            norm_cls = nn.BatchNorm1d,
            act_cls = act_cls,
            block_cls = LinearBlock,
            bias = True,
            dropout = 0,
            env_name = self.env_name
        )

        ## ----------------- Builds ----------------- ##

        ### ----------------- Skill Prior Modules ----------------- ###


        inverse_dynamics = InverseDynamicsMLP(Linear_Config(inverse_dynamics_config))
        subgoal_generator = SequentialBuilder(Linear_Config(subgoal_generator_config))
        prior = SequentialBuilder(Linear_Config(prior_config))
        flat_dynamics = SequentialBuilder(Linear_Config(dynamics_config))
        dynamics = SequentialBuilder(Linear_Config(dynamics_config))

        if self.robotics:
            ppc_config = {**prior_config}
            ppc_config['in_feature'] = self.state_dim
            prior_proprioceptive = SequentialBuilder(Linear_Config(ppc_config))
        else:
            prior_proprioceptive = None
        
        prior_wrapper_cls = PRIOR_WRAPPERS['gc_div_joint']

        self.inverse_dynamics_policy = prior_wrapper_cls(
            # components  
            prior_policy = prior,
            prior_proprioceptive = prior_proprioceptive,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            inverse_dynamics = inverse_dynamics,
            subgoal_generator = subgoal_generator,
            dynamics = dynamics,
            flat_dynamics = flat_dynamics,
            # architecture parameters
            ema_update = True,
            tanh = self.tanh,
            sample_interval = self.sample_interval,
            env_name = self.env_name
        )

        ### ----------------- Skill Enc / Dec Modules ----------------- ###
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))

        self.optimizers = {
            "skill_prior" : {
                "optimizer" : RAdam( self.inverse_dynamics_policy.prior_policy.parameters(), lr = self.lr ),
                "metric" : None
                # "metric" : "Prior_S"
                # "metric" : "Prior_GC"                
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
                "metric" : "Prior_GC"
            }, 
            "D" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.dynamics.parameters()},
                    {"params" : self.inverse_dynamics_policy.flat_dynamics.parameters()},
                ], lr = self.lr ),
                "metric" : "D"
                # "metric" : "Prior_GC"

            }, 
            "f" : {
                "optimizer" : RAdam( [
                    {"params" : self.inverse_dynamics_policy.subgoal_generator.parameters()},
                ], lr = self.lr ),
                "metric" : "F_skill_GT"
                # "metric" : "Prior_GC"
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
                "metric" : "recon_state"
            }

        if self.robotics:
            self.optimizers['ppc'] = {
                "optimizer" : RAdam(
                    [{'params' : self.inverse_dynamics_policy.prior_proprioceptive.parameters()}],            
                    lr = model_config.lr
                ),
                "metric" : None
            }




        # if self.env_name == "maze":
        #     visual_encoder_param_groups = [
        #         {'params' : self.visual_encoder.parameters()},
        #     ]

        #     self.optimizers['visual_encoder'] = {
        #         "optimizer" : RAdam(
        #             visual_encoder_param_groups,            
        #             lr = model_config.lr
        #         ),
        #         "metric" : "Rec_skill"
        #     }

        # Losses
        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            # 'reg' : ['kld', torch_dist.kl_divergence],
            'reg' : ['kld', kl_divergence],
            'prior' : ["nll", nll_dist] ,
            # 'prior' : ["nll", nll_dist2]
        }

        self.scaler = GradScaler()
        self.outputs = {}
        self.loss_dict = {}
        self.c = 0

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
            # KL (post || invD by subgoal from f)
            self.loss_dict['F_skill_GT'] = self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']).mean().item()
        
            # KL (post || state-conditioned prior)
            self.loss_dict['Prior_S']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['prior']).mean().item()
            
            # KL (post || goal-conditioned policy)
            self.loss_dict['Prior_GC']  = self.loss_fn('reg')(self.outputs['post_detach'], self.outputs['invD']).mean().item()
            
            # KL (invD || prior)
            self.loss_dict['Prior_GC_S']  = self.loss_fn('reg')(self.outputs['invD'], self.outputs['prior']).mean().item()
            
            # dummy metric 
            self.loss_dict['metric'] = self.loss_dict['Prior_GC']
            
            # subgoal by flat dynamics rollout
            reconstructed_subgoal = self.inverse_dynamics_policy.state_decoder(self.outputs['subgoal_rollout'])
            self.loss_dict['Rec_flatD_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['states'][:, -1, :]).item()
            self.loss_dict['Rec_D_subgoal'] = self.loss_fn('recon')(reconstructed_subgoal, self.outputs['subgoal_recon_D']).item()

            # state reconstruction 
            self.loss_dict['recon_state'] = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 


            # if (self.step + 1) % self.render_period == 0 and not self.training:
            #     num = (self.step + 1) // self.render_period
            #     i = 0 

            #     mp4_path = f"./imgs/{self.env_name}/video/video_{num}.mp4"
            #     self.render_funcs['imaginary_trajectory'](self.env, self.loss_dict['states_novel'][0], self.loss_dict['actions_novel'][0], self.c, mp4_path)

            
            #     subgoal_GT = self.render_funcs['scene']( self.env, self.outputs['states'][i, -1])
            #     subgoal_D = self.render_funcs['scene']( self.env,self.outputs['subgoal_recon_D'][i,])
            #     subgoal_F =self.render_funcs['scene'](self.env,self.outputs['subgoal_recon_f'][i])
            #     subgoal_F_skill = self.render_funcs['scene'](self.env,self.outputs['subgoal_recon_D_f'][i])


            #     img = np.concatenate((subgoal_GT, subgoal_D, subgoal_F, subgoal_F_skill), axis= 1)
            #     cv2.imwrite(f"./imgs/{self.env_name}/img/img_{num}.png", img)
            #     print(mp4_path)


    def forward(self, states, actions, G):
        
        N, T, _ = states.shape



        # if self.env_name == "maze":
        #     skill_states = states[:,:,:4].clone()
        #     # skill_states = states.clone()
        #     # with torch.no_grad():
        #     #     dec_states = self.inverse_dynamics_policy.state_encoder(states.view(N * T, -1)).view(N, T, -1)[:,:, :self.latent_env_dim]
        #     # dec_states = states[:, :, 4:].clone()
            
        #     # dec_states = self.visual_encoder(states[:, :, 4:].view(N * T, 1, 32, 32)).view(N, T, self.latent_state_dim)
        #     # dec_states = torch.zeros(N, T, self.latent_state_dim).cuda()
        #     # dec_states = self.visual_encoder(states[:, :, 4:].clone().view(N * T, -1)).view(N, T, -1)
        #     dec_states = states[:, :, :4].clone() 

        # else:
        #     skill_states = states.clone()
        #     dec_states = states.clone()

        # if self.env_name == "maze":
        #     skill_states = states[:, :, :4].clone()
        # else:
        #     skill_states = states.clone()
        
        # if self.env_name == "maze":
        #     pos_states = states[:,:,:4]
        #     visual_states = self.visual_encoder(states[:,:,4:].view(N * T, 1, 32, 32)).view(N, T, -1)
        #     skill_states = torch.cat((pos_states, visual_states), dim = -1)
        # else:
        #     skill_states = states.clone()

        skill_states = states.clone()


        # skill Encoder 
        enc_inputs = torch.cat( (skill_states.clone()[:,:-1], actions), dim = -1)
        q = self.skill_encoder(enc_inputs)[:, -1]
        q_clone = q.clone().detach()
        q_clone.requires_grad = False
        
        post = get_dist(q, tanh = self.tanh)
        post_detach = get_dist(q_clone, tanh = self.tanh)
        fixed_dist = get_fixed_dist(q_clone, tanh = self.tanh)

        if self.tanh:
            skill_normal, skill = post.rsample_with_pre_tanh_value()
            # Outputs
            z = skill.clone().detach()
            z_normal = skill_normal.clone().detach()

        else:
            skill = post.rsample()
            # self.outputs['z'] = skill.clone().detach()
            # self.outputs['z_normal'] = None
            z_normal = None
            z = skill.clone().detach()
        
        # Skill Decoder 
        # decode_inputs = self.dec_input(states[:,:,4:].clone(), skill, self.Hsteps)
        # if self.env_name == "maze":
        #     decode_inputs = self.dec_input(visual_states, skill, self.Hsteps)
        # else:
        #     decode_inputs = self.dec_input(states.clone(), skill, self.Hsteps)
        decode_inputs = self.dec_input(skill_states.clone(), skill, self.Hsteps)

        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        

        inputs = dict(
            states = states,
            G = G,
            skill = z
        )

        # skill prior
        self.outputs =  self.inverse_dynamics_policy(inputs, "train")
        
        # skill prior & inverse dynamics's target
        self.outputs['z'] = z
        self.outputs['z_normal'] = z_normal

        self.outputs['post'] = post
        self.outputs['post_detach'] = post_detach
        self.outputs['fixed'] = fixed_dist
        self.outputs['actions_hat'] = skill_hat
        self.outputs['actions'] = actions

    def compute_loss(self, actions):

        # ----------- Skill Recon & Regularization -------------- # 
        recon = self.loss_fn('recon')(self.outputs['actions_hat'], actions)
        reg = self.loss_fn('reg')(self.outputs['post'], self.outputs['fixed']).mean()
        

        # ----------- State/Goal Conditioned Prior -------------- # 
        if self.subgoal_loss == "prior":
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
        else:
            prior = self.loss_fn('reg')(
                self.outputs['post_detach'],
                self.outputs['prior'], # distributions to optimize
            ).mean()

            invD_loss = self.loss_fn('reg')(
                self.outputs['post_detach'],
                self.outputs['invD'], 
            ).mean() 

        # # ----------- Dynamics -------------- # 
        flat_D_loss = self.loss_fn('recon')(
            self.outputs['flat_D'],
            self.outputs['flat_D_target']
        ) 

        D_loss = self.loss_fn('recon')(
            self.outputs['D'],
            self.outputs['D_target']
        )         
        

        # # ----------- subgoal generator -------------- # 
        r_int_f = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_target'])
        r_int_D = self.loss_fn("recon")(self.outputs['subgoal_D'], self.outputs['subgoal_target'])
        r_int = r_int_f + r_int_D
        # r_int = self.loss_fn("recon")(self.outputs['subgoal_f'], self.outputs['subgoal_D'])

        reg_term = self.loss_fn("reg")(self.outputs['invD_detach'], self.outputs['invD_sub']).mean()
        # reg_term = self.loss_fn("reg")(self.outputs['post_detach'], self.outputs['invD_sub']).mean()

        F_loss = r_int + reg_term 

        loss = recon + reg * self.reg_beta + prior + invD_loss + flat_D_loss + D_loss + F_loss

        recon_state = self.loss_fn('recon')(self.outputs['states_hat'], self.outputs['states']) # ? 
        z_tilde = self.outputs['states_repr']
        z = self.outputs['states_fixed_dist']
        mmd_loss = compute_mmd(z_tilde, z)
        loss = loss + recon_state + mmd_loss


        if self.robotics:
            ppc_loss = self.loss_fn('prior')(
                self.outputs['z'],
                self.outputs['prior_ppc'], # proprioceptive for stitching
                self.outputs['z_normal'],
                tanh = self.tanh
            ).mean()

            loss += ppc_loss

            
        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "invD" : invD_loss.item(),
            "D" : D_loss.item(),
            "flat_D" : flat_D_loss.item(),
            "F" : F_loss.item(),
            "F_state" : r_int.item(),
            "r_int_f" : r_int_f.item(),
            "r_int_D" : r_int_D.item(),
            "F_skill_kld" : reg_term.item(),
            "skill_metric" : recon.item() + reg.item() * self.reg_beta,
            "Rec_state" : recon_state.item(),
            "mmd_loss" : mmd_loss.item(),
            "ppc_loss" : ppc_loss.item() if self.robotics else 0
        }       


        return loss
    
    @torch.no_grad()
    def rollout(self, inputs):
        result = self.inverse_dynamics_policy(inputs, self.rollout_method)

        if self.rollout_method == "rollout":
            # indexing outlier 
            c = result['c']
            states_rollout = result['states_rollout']
            skill_sampled = result['skill_sampled']      



            N, T = states_rollout.shape[:2]
            # if self.env_name == "maze":
            #     visual_states = self.visual_encoder(states_rollout[:,:,4:].view(N * T, 1, 32, 32)).view(N, T, -1)
            #     dec_inputs = self.dec_input(visual_states, skill_sampled, states_rollout.shape[1])
            # else:
            #     dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])

            dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])


            # dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])

            N, T, _ = dec_inputs.shape
            actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)
            

            states_novel = torch.cat((inputs['states'][:, :c+1], states_rollout), dim = 1)
            actions_novel = torch.cat((inputs['actions'][:, :c], actions_rollout), dim = 1)
            
            self.loss_dict['states_novel'] = states_novel[inputs['rollout']].detach().cpu()
            self.loss_dict['actions_novel'] = actions_novel[inputs['rollout']].detach().cpu()
            self.c = c

        else: # rollout2
            states_rollout = result['states_rollout']
            skills = result['skills']         
            dec_inputs = torch.cat((states_rollout[:, :-1], skills), dim = -1)
            N, T, _ = dec_inputs.shape
            actions_rollout = self.skill_decoder(dec_inputs.view(N * T, -1)).view(N, T, -1)

            states_novel = states_rollout
            actions_novel = actions_rollout
            self.loss_dict['states_novel'] = states_novel[inputs['rollout']].detach().cpu()
            self.loss_dict['actions_novel'] = actions_novel[inputs['rollout']].detach().cpu()
        


    def __main_network__(self, states, actions, G, rollout, validate = False):

        self(states, actions, G)
        loss = self.compute_loss(actions)
        self.loss_dict['KL_threshold'] = self.KL_threshold

        if not validate:
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
                
            loss.backward()

            for module_name, optimizer in self.optimizers.items():
                self.grad_clip(optimizer['optimizer'])
                optimizer['optimizer'].step()

        # ------------------ Rollout  ------------------ #
        rollout_inputs = dict(
            states = states,
            actions = actions,
            rollout = rollout, 
        )

        with torch.no_grad():
            self.eval()
            self.rollout(rollout_inputs)
        
        if self.training:
            self.train()

    def optimize(self, batch, e):
        # inputs & targets       
        # self.step += 1
        # print(self.step * 64)

        states, actions, G, rollout = batch.values()
        states, actions, G, rollout = states.float().cuda(), actions.cuda(), G.cuda(), rollout.cuda()
        self.__main_network__(states, actions, G, rollout)
        with torch.no_grad():
            self.get_metrics()
            self.inverse_dynamics_policy.soft_update()
        return self.loss_dict
    

    def validate(self, batch, e):
        states, actions, G, rollout = batch.values()
        states, actions, G, rollout = states.float().cuda(), actions.cuda(), G.cuda(), rollout.cuda()
        self.__main_network__(states, actions, G, rollout, validate= True)
        self.get_metrics()
        self.step += 1

        return self.loss_dict