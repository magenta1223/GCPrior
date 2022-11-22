from proposed.configs.models import *
from proposed.modules.base import *
from proposed.modules.subnetworks import *
import torch
import torch.nn as nn
from torch.optim import *
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from proposed.utils import *

from proposed.contrib.momentum_encode import EMA, update_moving_average

# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 


def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

class GCIDSkillPrior(BaseModule):
    """
    """

    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = True

        self.step = 0
        self.Hsteps = 10

        # submodules
        ## skill prior module
        inverse_dynamics_config = edict(
            z_dim = self.latent_dim, 
            n_blocks = self.n_processing_layers, 
            in_feature = self.latent_dim * 2, # state dim 
            hidden_dim = self.hidden_dim, # 128
            out_dim = self.latent_dim * 2, # * 2 when variational inference
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        encoder_config = edict(
            in_feature = self.action_dim + self.state_dim,
            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim * 2, # * 2 when variational inference
            n_layers= 1,
            bias = False,
            batch_first = True,
            dropout = 0,
            linear_cls = LinearBlock,
            rnn_cls = nn.LSTM,
            act_cls = nn.LeakyReLU,
            true = True,
        )

        decoder_config = edict(
            n_blocks = self.n_processing_layers,
            state_dim = self.state_dim,
            z_dim = self.latent_dim, 
            in_feature = self.latent_dim + self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.action_dim,
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        dynamics_config = edict(
            n_blocks = 2,
            in_feature = self.latent_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2,
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        state_encoder_config = edict(
            n_blocks = self.n_processing_layers,
            in_feature = self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2, # * 2 when variational inference
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        state_decoder_config = edict(
            n_blocks = self.n_processing_layers,
            in_feature = self.latent_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.state_dim,
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        subgoal_generator_config = edict(
            n_blocks = 5,
            in_feature = self.latent_dim * 2, # state_dim + latent_dim 
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2,
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        inverse_dynamics = PriorNetwork(Linear_Config(inverse_dynamics_config))
        state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))
        subgoal_g = SequentialBuilder(Linear_Config(subgoal_generator_config))
        # dynamics = SequentialBuilder(Linear_Config(dynamics_config))
        ema_updater = EMA(0.99)

        self.skill_prior = PriorWrapper(
            mode = "gcid",
            prior_policy = inverse_dynamics,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            subgoal_generator = subgoal_g,
            # dynamics = dynamics,
            ema_updater = ema_updater
        )
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))


        # optimizer
        self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        # Losses

        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
            'prior_metric' : ["kld", torch_dist.kl_divergence]
        }


        self.scaler = GradScaler(self.use_amp)


        self.target_kl = 1
        self.i_term = 0
        self.Kp = 1e-2
        self.Ki = 5e-4
        self.beta_min = 0.000001
        self.beta_max = 0.001
        self.prev_beta = 0

        self.cvae = False

    def beta(self, kl_error):
        if self.cvae:
            # Control VAE
            p_term = self.Kp / (1 + np.exp(kl_error))

            if  self.beta_min <= self.prev_beta <= self.beta_max:
                self.i_term -= self.Ki * kl_error

            beta_t = p_term + self.i_term + self.beta_min
            beta = min(max(beta_t, self.beta_min), self.beta_max)        

            return beta
        else:
            return self.reg_beta

    def dec_input(self, states, z, steps):
        return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def loss_fn(self, key, index = 1):
        return self.loss_fns[key][index]

    def forward(self, relabeled_inputs, post_inputs, actions, G):
        # skill prior
        outputs = self.skill_prior(relabeled_inputs, G)

        # ------------------------ DO NOT EDIT ------------------------ #

        enc_inputs = torch.cat( (actions, post_inputs[:,:-1]), dim = -1)
        q = self.skill_encoder(enc_inputs)[:, -1]

        outputs['post'] = get_dist(q)
        outputs['post_detach'] = get_dist(q, detached= True)
        outputs['fixed'] = get_fixed_dist(q)


        z = outputs.post.rsample() # samples by reparameterize trick. 

        decode_inputs = self.dec_input(post_inputs, z, self.Hsteps)

        N, T = decode_inputs.shape[:2]

        # prevent lookahead     
        outputs['skill_hat'] = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)


        with torch.no_grad():
            dec_inputs_prior = self.dec_input(post_inputs, outputs.prior_detach.sample(), self.Hsteps)
            outputs['skill_hat_prior'] = self.skill_decoder(dec_inputs_prior.view(N * T, -1)).view(N, T, -1)


        return outputs, N, T

    def compute_loss(self, outputs, skill, val = False):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(outputs.skill_hat, skill)
        reg = self.loss_fn('reg')(outputs.post, outputs.fixed).mean()
        prior = self.loss_fn('prior')(
            outputs.post_detach, # target distributions
            outputs.prior, # distributions to optimize
        ).mean()

        error = self.target_kl- reg.item()



        # ----------- Add -------------- # 
        # State Reconstruction
        recon_state = self.loss_fn('recon')(outputs.s_hat, outputs.states)

        # State Regularization
        reg_state = self.loss_fn('reg')(outputs.hsd, outputs.hs_target).mean()

        # predicted h_tH NLL
        reg_h_th = self.loss_fn('prior')(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()


        # ----------- Metrics ----------- #
        with torch.no_grad():
            prior_hat = self.loss_fn('reg')(outputs.prior_hat, outputs.prior_detach).mean()
            reg_prior_hat = self.loss_fn('reg')(outputs.post_detach, outputs.prior_hat).mean()
            # dynamics_kld = self.loss_fn('reg')(outputs.dynamics, outputs.hsd_tH).mean() 
            reg_h_th_kld = self.loss_fn('reg')(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()
            prior_kld  = self.loss_fn('reg')(outputs.post_detach, outputs.prior_detach).mean() 


        if val:
            beta = self.prev_beta
        else:
            beta =  self.beta(error)

        # loss = recon + reg * self.reg_beta  + prior + dynamics + subgoal
        loss = recon + reg * beta  + prior + recon_state + reg_h_th + reg_state * 0.15 #+ dynamics_loss

        metric = recon.item() + reg.item() * beta + prior_kld.item()

        self.prev_beta = beta

        loss_dict = {           
            # total
            "loss" : loss.item(),
            # spirl
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Pri" : prior_kld.item(),
            
            # proposed
            # "Rec_state" : state_recon.item(), # 별로 안중요함 
            # "Reg_H" : state_vae.item(),
            # "Reg_H_tH" : reg_h_th.item(),
            "Reg_H_tH_kl" : reg_h_th_kld.item(),
            "Pri_hat" : prior_hat.item(),
            "Reg_Pri_hat" : reg_prior_hat.item(),
            # "D_kld" : dynamics_kld.item(),
            "metric" : metric,
            # "beta" : beta
        }       

        return loss, loss_dict



    def optimize(self, batch, e):
        # inputs & targets       
 
        relabeled_inputs, states, actions, G = batch.values()
        relabeled_inputs, states, actions, G = relabeled_inputs.cuda(), states.cuda(), actions.cuda(), G.cuda()
        
        with autocast(self.use_amp):
            # forwarding
            outputs, N, T = self(relabeled_inputs, states, actions, G)
            loss, loss_dict = self.compute_loss(outputs, actions)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # 
        if self.step < self.init_grad_clip_step:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.init_grad_clip) # 학습 초기에 prior loss가 너무 큼. 
            
        self.scaler.step(self.optimizer)
        self.scaler.update()  
        self.optimizer.zero_grad()
        self.step += 1

        self.skill_prior.ma_state_enc()

        return loss_dict
    
    def validate(self, batch, e):
        # inputs & targets          
        relabeled_inputs, states, actions, G = batch.values()
        relabeled_inputs, states, actions, G = relabeled_inputs.cuda(), states.cuda(), actions.cuda(), G.cuda()
        
        # forwarding
        with autocast(self.use_amp), torch.no_grad():
            # forwarding
            outputs, N, T = self(relabeled_inputs, states, actions, G)
            loss, loss_dict = self.compute_loss(outputs, actions)

        return loss_dict