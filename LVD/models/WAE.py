import torch
import torch.nn as nn
from torch.optim import *
from ..utils import *
from ..configs.build import *
from ..modules.base import *
from ..modules.subnetworks import *
from ..contrib.momentum_encode import update_moving_average


class WAE(BaseModule):
    """
    """
    def __init__(self, config):
        super().__init__(config)

        self.use_amp = False


        norm_cls = nn.LayerNorm

        # act_cls = nn.LeakyReLU
        act_cls = nn.Mish
        self.step= 0
        self.distributional = False

        if self.env_name == "maze":
            self.latent_dim = self.latent_env_dim
 
        ## ----------------- Configurations ----------------- ##
        # state encoder
        state_encoder_config = edict(
            n_blocks = self.n_Layers,
            in_feature = self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim , # when variational inference
            norm_cls =  norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            bias = True,
            dropout = 0,
            tanh = False
        )

        # state decoder
        state_decoder_config = edict(
            n_blocks = self.n_Layers,#self.n_processing_layers,
            in_feature = self.latent_dim, # when variational inference
            hidden_dim = self.hidden_dim, 
            out_dim = self.state_dim,
            norm_cls = norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            bias = True,
            dropout = 0
        )
        
        # for manipulation, sensor data task
        state_to_goal_config = edict(
            n_blocks = self.n_Layers,
            in_feature = self.latent_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim, # when variational inference
            norm_cls =  norm_cls,
            act_cls = act_cls, #nn.LeakyReLU,
            block_cls = LinearBlock,
            bias = True,
            dropout = 0,
            tanh = False
        )


        self.state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        self.state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))
        # self.state_to_goal = SequentialBuilder(Linear_Config(state_to_goal_config))

        
        self.target_state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())


        # self.optimizer = RAdam(
        #     [
        #         {'params' : self.state_encoder.parameters()},
        #         {'params' : self.state_decoder.parameters()},
        #         # {'params' : self.state_to_goal.parameters()}, # optional for manipulation and sensor task. 
        #     ],            
        #     lr = config.lr
        # )

        self.optimizers = {
            "state_enc_dec" : {
                "optimizer" : RAdam( [
                    {'params' : self.state_encoder.parameters()},
                    {'params' : self.state_decoder.parameters()},
                ], lr = self.lr ),
                "metric" : "rec"
                # "metric" : "Prior_GC"                
            },     
        }


        # Losses
        self.loss_fns = {
            'mse' : ['mse', nn.MSELoss()],
            'huber' : ['huber', nn.HuberLoss(delta = 0.01)],
            'l1' : ['l1', nn.L1Loss()],
        }

        self.outputs = {}
        self.loss_dict = {}

        self.binary = False




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


        self.loss_dict['metric'] = self.loss_dict['rec']
        


        # F.mse_loss(self.outputs['states'][:30], self.outputs['states'][:30])


    def forward(self, states):


        # if self.masked:
        #     _mask = torch.randn_like(states)
        #     input_mask = torch.where(_mask > self.mask_ratio, 1, 0).cuda()
        #     output_mask = torch.where(_mask <= self.mask_ratio, 1, 0).cuda()
        #     self.outputs['output_mask'] = output_mask
        #     self.outputs['input_mask'] = input_mask

        #     states = states * input_mask # ratio가 0일 때, input_mask는 전부 1, output mask는 전부 0 ? 

        # skill prior
        N = states.shape[0]

        # states = self.add_noise(states)
        self.outputs['states'] = states.clone()


        states_repr = self.state_encoder(states)

        states_hat = self.state_decoder(states_repr)

        self.outputs['states_repr'] = states_repr
        self.outputs['states_hat'] = states_hat

        mu, log_scale = torch.zeros(512, self.latent_dim * 2).cuda().chunk(2, -1)
        scale = log_scale.exp()
        dist = torch_dist.Normal(mu, scale)
        dist = torch_dist.Independent(dist, 1)  
        # 정규분포와 가깝게 만든다. 

        self.outputs['states_fixed_dist'] = dist.sample()

        # self.outputs['states_fixed_dist'] = torch.randn(512, self.latent_dim).cuda()

   
        # # 밀어 버리기~~ 
        # states_only_goal = states.clone()
        # states_only_goal[:, :9] = 0

        # with torch.no_grad():
        #     self.target_state_encoder.eval()
        #     self.outputs['goal_repr_target'] = self.target_state_encoder(states_only_goal)
        
        # self.outputs['goal_repr'] = self.state_to_goal(self.outputs['states_repr'].clone().detach())


    def compute_loss(self):

        # State Reconstruction
        if self.binary:
            recon_state = self.loss_fn("mse")(torch.sigmoid(self.outputs['states_hat']), self.outputs['states']) # ? 
        else:
            recon_state = self.loss_fn("mse")(self.outputs['states_hat'], self.outputs['states']) # ? 
        
        # MMD
        z_tilde = self.outputs['states_repr']
        z = self.outputs['states_fixed_dist']#.sample()
        mmd_loss = compute_mmd(z_tilde, z)

        # States to Goal        
        # recon_goal = self.loss_fn(self.recon_loss)(self.outputs['goal_repr'], self.outputs['goal_repr_target']) # ? 


        loss = recon_state + mmd_loss #+ recon_goal
        

        states_hat =  self.outputs['states_hat'].clone().detach()
        states =  self.outputs['states'].clone().detach()

        # for k, v in OBS_ELEMENT_INDICES.items():
        #     states[:, v] = states[:, v] * math.sqrt(len(v))
        #     states_hat[:, v] = states_hat[:, v] * math.sqrt(len(v))

        # metric = self.loss_fn(self.recon_loss)(states, states_hat) # ? 

        # with torch.no_grad():
        #     repr_diff = self.loss_fn("mse")(self.outputs['goal_repr_target'], self.outputs['states_repr']) # ? 


        self.loss_dict = {           
            # total
            "loss" : loss.item(), #+ prior_loss.item(),
            "rec" : recon_state.item(),
            "mmd" : mmd_loss.item(),
            # "rec_original_scale" : metric.item()
            # "recon_goal" : recon_goal.item(),
            # "repr_diff" : repr_diff.item()
        }       

        return loss
    
    def __main_network__(self, states, validate = False):
        # outputs = self(states, G, actions)
        # loss, loss_dict = self.compute_loss(outputs, actions)

        self(states)
        loss = self.compute_loss()

        if not validate:
            # update SPiRL + state enc / dec
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].zero_grad()
            loss.backward()
            # self.grad_clip(self.optimizer)
            for module_name, optimizer in self.optimizers.items():
                optimizer['optimizer'].step()            
            update_moving_average(self.target_state_encoder, self.state_encoder)

    def optimize(self, batch, e):
        self.loss_dict = {}
        self.outputs = {}

        # inputs & targets       
        states = batch['states']
        states = states.cuda()

        if len(states.shape) == 4:
            self.binary = True
            N, T = states.shape[:2]
            states = states.view(N * T, -1).float() # flatten 



        self.__main_network__(states)
        with torch.no_grad():
            self.get_metrics()
        self.step += 1
        
        return self.loss_dict
    
    
    def validate(self, batch, e):
        self.loss_dict = {}
        self.outputs = {}
        # inputs & targets       
        states = batch['states']
        states = states.cuda()

        if len(states.shape) == 4:
            N, T = states.shape[:2]
            states = states.view(N * T, -1).float() # flatten 

        self.__main_network__(states, True)
        with torch.no_grad():
            self.get_metrics()
        self.step += 1

        return self.loss_dict