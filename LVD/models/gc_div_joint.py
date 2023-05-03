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

        # if self.env_name == "maze":
        #     # self.state_dim = self.state_dim + self.latent_env_dim

        #     self.state_dim = self.state_dim + 32 * 32
        #     visual_ae = torch.load(self.visual_encoder_path)['model']
        #     self.visual_encoder = visual_ae.state_encoder.eval()
        #     self.visual_decoder = visual_ae.state_decoder.eval()

        # self.init_grad_clip_step = 100


        norm_cls = nn.LayerNorm
        # norm_cls = nn.BatchNorm1d
        act_cls = nn.Mish

        # self.env = gym.make("simpl-kitchen-v0")
        self.env = KitchenEnv_GC()

        self.qpo_dim = self.env.robot.n_jnt + self.env.robot.n_obj
        self.qv = self.env.init_qvel[:].copy()
        self.joint_learn = True

        if self.only_proprioceptive:
            self.state_dim = self.n_obj



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
            print("here?")

            self.latent_state_dim *= 2
            state_encoder = MultiModalEncoder(state_encoder_config)
            state_decoder = MultiModalDecoder(state_decoder_config)
        else:
            state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
            state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))
        
        # state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        # state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))


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
            in_feature = self.action_dim + 1028,
            # in_feature = self.action_dim + self.latent_state_dim,
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
            # state_dim = self.state_dim,
            # in_feature =  self.state_dim + self.latent_dim, # state_dim + latent_dim 
            state_dim = 1028,
            in_feature =  1028 + self.latent_dim, # state_dim + latent_dim

            hidden_dim = self.hidden_dim, 
            # out_dim = self.action_dim,
            out_dim = self.action_dim,

            norm_cls = nn.BatchNorm1d,
            act_cls = act_cls,
            block_cls = LinearBlock,
            bias = True,
            dropout = 0,
        )

        ## ----------------- Builds ----------------- ##

        ### ----------------- Skill Prior Modules ----------------- ###


        inverse_dynamics = InverseDynamicsMLP(Linear_Config(inverse_dynamics_config))
        subgoal_generator = SequentialBuilder(Linear_Config(subgoal_generator_config))
        prior = SequentialBuilder(Linear_Config(prior_config))
        flat_dynamics = SequentialBuilder(Linear_Config(dynamics_config))
        dynamics = SequentialBuilder(Linear_Config(dynamics_config))
        
        prior_wrapper_cls = PRIOR_WRAPPERS['gc_div_joint']

        self.inverse_dynamics_policy = prior_wrapper_cls(
            # components  
            prior_policy = prior,
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


    def render_video_compare(self, states, actions, mode):
        """
        rollout을 통해 만들어낸 trajectory의
        -state sequence를 강제로 세팅
        -초기 state를 세팅하고, actino을 환경상에서 수행
        두 개를 비교
        """

        imgs = []

        video_len = states.shape[0]

        if mode == "state":
            for i in range(video_len):
            # for i in range(self.Hsteps + 1):
                self.env.set_state(states[i][:self.qpo_dim], self.qv)
                imgs.append(self.env.render(mode = "rgb_array"))

        else:
            # action에서 qv가 달라서 보정 필요함.
            # 일단 action을 수행 -> qv가 맞게 세팅됨. 근데 위치는 다름.
            # set_state(next state, now_qv)
            # 초기 state를 세팅 후 render 
            self.env.set_state(states[0][:self.qpo_dim], self.qv)
            imgs.append(self.env.render(mode = "rgb_array"))

            # action을 수행. 그러나 data 수집 당시의 qv와 달라서 약간 달라짐. 강제로 교정 후 render
            self.env.step(actions[0].detach().cpu().numpy())
            now_qv = self.env.sim.get_state().qvel
            self.env.set_state(states[1][:self.qpo_dim], now_qv)
            
            # flat_d_len = 10 if self.rollout_method == "rollout" else 20
            flat_d_len = 10
            for i in range(flat_d_len -1):
                # render 
                imgs.append(self.env.render(mode = "rgb_array"))
                state, reward, done, info = self.env.step(actions[i].detach().cpu().numpy())

            last_img = self.env.render(mode = "rgb_array")
            imgs.append(last_img)
                  
            for i in range(self.Hsteps + 1, video_len):
            # for i in range(self.Hsteps + 1):
                imgs.append(last_img)


        return imgs


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

            # # renders
            # if (self.step + 1) % 4 == 0 and not self.training:
            #     i = 0

            #     task_obj = KitchenTask(subtasks = ['kettle'])
            #     num = (self.step + 1) // 4
            
            #     # if "states_novel" in self.loss_dict.keys():
            #     with self.env.set_task(task_obj):
            #         self.env.reset()
            #         imgs_state = self.render_video_compare(self.loss_dict['states_novel'][0], self.loss_dict['actions_novel'][0], mode = "state")
            #         self.env.reset()
            #         imgs_action = self.render_video_compare(self.loss_dict['states_novel'][0], self.loss_dict['actions_novel'][0], mode = "action")

            #     mp4_path = f"./imgs/video/video_{num}.mp4"
            #     out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (1200,400))

            #     for i in range(len(imgs_state)):
            #         # writing to a image array
            #         img_s = imgs_state[i].astype(np.uint8)
            #         img_a = imgs_action[i].astype(np.uint8)
            #         img = np.concatenate((img_s,img_a, np.abs(img_s - img_a)), axis = 1)
            #         text = f"S-A now {i} c {self.c}" if self.c != 0 else f"S-A now {i}"
            #         cv2.putText(img = img,    text = text, color = (255,0,0),  org = (400 // 2, 400 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)
            #         out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #     out.release() 

            #     self.env.set_state(self.outputs['states'][i, -1][:self.qpo_dim], self.qv)
            #     img_GT = self.env.render(mode = "rgb_array")
            #     self.env.set_state(self.outputs['subgoal_recon_D'][i][:self.qpo_dim], self.qv)
            #     img_D = self.env.render(mode = "rgb_array")
            #     self.env.set_state(self.outputs['subgoal_recon_f'][i][:self.qpo_dim], self.qv)
            #     img_F = self.env.render(mode = "rgb_array")
            #     self.env.set_state(self.outputs['subgoal_recon_D_f'][i][:self.qpo_dim], self.qv)
            #     img_F_D = self.env.render(mode = "rgb_array")

            #     skill = self.outputs['z_sub']
            #     dec_input = self.dec_input(self.outputs['states'], skill, self.Hsteps)
            #     N, T, _ = dec_input.shape
            #     raw_actions = self.skill_decoder(dec_input.view(N * T, -1)).view(N, T, -1)[i]
                
            #     with self.env.set_task(task_obj):
            #         self.env.reset()
            #         self.env.set_state(self.outputs['states'][i, 0][:self.qpo_dim], self.qv)
            #         for idx in range(self.Hsteps):
            #             self.env.step(raw_actions[idx].detach().cpu().numpy())

            #     img_sub_skill = self.env.render(mode = "rgb_array")

            #     img = np.concatenate((img_GT, img_D, img_F, img_F_D, img_sub_skill), axis= 1)
            #     cv2.imwrite(f"./imgs/img/img_{num}.png", img)
            #     print(mp4_path)




    def forward(self, states, actions, G):
        
        N, T, _ = states.shape

        # if self.env_name == "maze":
        #     # # imgs  : N, T, 32, 32
        #     N, T = imgs.shape[:2]
        #     # with torch.no_grad():
        #     #     visual_embedidng = self.visual_encoder(imgs.view(N * T, -1).float()).view(N, T, -1) # N, T ,32
        #     #     states = torch.cat((states, visual_embedidng), axis = -1)
        #     states = torch.cat((  states, imgs.view(N, T, -1)   ), dim = -1)
        


        # inputs = dict(
        #     states = states,
        #     G = G
        # )

        # # skill prior
        # self.outputs =  self.inverse_dynamics_policy(inputs, "train")
        
        if self.env_name == "maze":
            # skill_states = states[:,:,:4].clone()
            skill_states = states.clone()

        else:
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
        decode_inputs = self.dec_input(skill_states.clone(), skill, self.Hsteps)
        N, T = decode_inputs.shape[:2]
        skill_hat = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)
        

        inputs = dict(
            states = states,
            G = G,
            skill = skill if self.skill_concat else z
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
            "mmd_loss" : mmd_loss.item()
        }       


        return loss
    
    @torch.no_grad()
    def rollout(self, inputs):
        # if self.env_name == "maze":
        #     # imgs  : N, T, 32, 32
        #     states, imgs = inputs['states'], inputs['imgs']
        #     N, T =imgs.shape[:2]
        #     with torch.no_grad():
        #         visual_embedidng = self.visual_encoder(imgs.view(N * T, -1).float()).view(N, T, -1) # N, T ,32
        #         states = torch.cat((states, visual_embedidng), axis = -1)
        #         inputs['states'] = states

        # if self.env_name == "maze":
        #     # # imgs  : N, T, 32, 32
        #     states, imgs = inputs['states'], inputs['imgs']
        #     N, T = imgs.shape[:2]
        #     states = torch.cat((  states, imgs.view(N, T, -1)   ), dim = -1)
        #     inputs['states'] = states


        result = self.inverse_dynamics_policy(inputs, self.rollout_method)

        if self.rollout_method == "rollout":
            # indexing outlier 
            c = result['c']
            states_rollout = result['states_rollout']
            skill_sampled = result['skill_sampled']      
            
            # if self.env_name == "maze": 
            #     dec_inputs = self.dec_input(states_rollout[:, :, :4], skill_sampled, states_rollout.shape[1])
            # else:
            #     dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])
            dec_inputs = self.dec_input(states_rollout, skill_sampled, states_rollout.shape[1])


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
        # states, actions, G, rollout = batch.values()
        # states, actions, G, rollout = states.cuda(), actions.cuda(), G.cuda(), rollout.cuda()
        # self.__main_network__(states, actions, G, rollout)

        # inputs & targets       

        states, actions, G, rollout = batch.values()
        states, actions, G, rollout = states.float().cuda(), actions.cuda(), G.cuda(), rollout.cuda()

        self.__main_network__(states, actions, G, rollout)



        with torch.no_grad():
            self.get_metrics()
            self.inverse_dynamics_policy.soft_update()
        # self.step += 1

        return self.loss_dict
    
    
    def validate(self, batch, e):

        # states, actions, G, rollout  = batch.values()
        # states, actions, G, rollout = states.cuda(), actions.cuda(), G.cuda(), rollout.cuda()
        # self.__main_network__(states, actions, G, rollout, validate= True)


        states, actions, G, rollout = batch.values()
        states, actions, G, rollout = states.float().cuda(), actions.cuda(), G.cuda(), rollout.cuda()

        self.__main_network__(states, actions, G, rollout, validate= True)


        self.get_metrics()
        self.step += 1

        return self.loss_dict