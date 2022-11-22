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

class GCID_VICREG_SkillPrior(BaseModule):
    """
    """

    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = True

        self.step = 0
        self.Hsteps = 10

        # ----------------- SUBMODULES ----------------- #

        ## ----------------- Configurations ----------------- ##

        ### ----------------- skill prior modules ----------------- ###

        # state encoder
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

        # state decoder
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

        # subgoal generator
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


        # inverse dynamics
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

        # prior projection
        prior_proj_config = edict(
            n_blocks = 1,
            in_feature = self.latent_dim, # state_dim + latent_dim 
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim,
            norm_cls = None,
            act_cls = None, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )


        ### ----------------- posterior modules ----------------- ###

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

        post_proj_config = edict(
            n_blocks = 1,
            in_feature = self.latent_dim, # state_dim + latent_dim 
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim,
            norm_cls = None,
            act_cls = None, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )


        ## ----------------- Builds ----------------- ##

        inverse_dynamics = PriorNetwork(Linear_Config(inverse_dynamics_config))
        state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))
        state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))
        subgoal_g = SequentialBuilder(Linear_Config(subgoal_generator_config))
        prior_proj = SequentialBuilder(Linear_Config(prior_proj_config))
        ema_updater = EMA(0.99)

        self.skill_prior = PriorWrapper(
            mode = "vic",
            prior_policy = inverse_dynamics,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            subgoal_generator = subgoal_g,
            prior_proj = prior_proj,
            # dynamics = dynamics,
            ema_updater = ema_updater
        )
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))
        self.post_proj = SequentialBuilder(Linear_Config(post_proj_config))

        # optimizer
        self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        # Losses

        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
            'prior_metric' : ["kld", torch_dist.kl_divergence],
            'variance' : ["var", V_loss],
            # 'invariance' : ["invar", nn.MSELoss()],
            'invariance' : ["invar", nn.CosineSimilarity()],
            'covariance' : ['cov', COV_loss]
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
        """
        Control VAE
        """
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

    @staticmethod
    def dec_input(states, z, steps):
        return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def loss_fn(self, key, index = 1):
        return self.loss_fns[key][index]

    def forward(self, relabeled_inputs, post_inputs, actions, G):
        
        inputs = edict(
            states = relabeled_inputs,
            G = G
        )
        
        # skill prior
        outputs = self.skill_prior(inputs)

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

        # ------------------------------------------------------------- #

        # For VICREG

        outputs['post_proj'] = self.post_proj(z)

        return outputs, N, T
 
    def compute_loss(self, outputs, skill, val = False):
        # ----------- SPiRL -------------- # 

        recon = self.loss_fn('recon')(outputs.skill_hat, skill)
        reg = self.loss_fn('reg')(outputs.post, outputs.fixed).mean()
        prior = self.loss_fn('prior')(
            outputs.post_detach, # target distributions
            outputs.prior, # distributions to optimize
        ).mean()

        # -------------- GCID -------------- # 
        # State Reconstruction
        recon_state = self.loss_fn('recon')(outputs.s_hat, outputs.states)

        # State Regularization
        reg_state = self.loss_fn('reg')(outputs.hsd, outputs.hs_target).mean()

        # predicted h_tH NLL
        reg_h_th = self.loss_fn('prior')(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()


        # -------------- VICREG -------------- # 
        # outputs.pr_proj
        # outputs.post_proj

        x, y = outputs.pr_proj, outputs.post_proj
        
        # Variance Term 
        VAR = self.loss_fn('variance')(x, y) * 0.1

        # Invariance Term. Replaced by Prior Loss

        # Covariance Term
        COV = self.loss_fn('covariance')(x, y) * 0.01
         

        # ----------- Metrics ----------- #
        with torch.no_grad():
            prior_hat = self.loss_fn('reg')(outputs.prior_hat, outputs.prior_detach).mean()
            reg_prior_hat = self.loss_fn('reg')(outputs.post_detach, outputs.prior_hat).mean()
            # dynamics_kld = self.loss_fn('reg')(outputs.dynamics, outputs.hsd_tH).mean() 
            reg_h_th_kld = self.loss_fn('reg')(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()
            prior_kld  = self.loss_fn('reg')(outputs.post_detach, outputs.prior_detach).mean() 
            # INV = self.loss_fn('invariance')(outputs.pr_proj, outputs.pr_proj_cycle).mean()
        



        # loss = recon + reg * self.reg_beta  + prior + dynamics + subgoal
        # loss = recon + reg * self.reg_beta  + prior + recon_state + reg_h_th + reg_state * 0.15 
        # SPiRL
        loss = recon + reg * self.reg_beta  + prior
        # state reconstruction + imaginary state
        loss += recon_state + reg_h_th + reg_state * 0.15 
        # additional penalty 학습 안정화에 좋음
        # prior_proj : learned linear function이고
        # 출력은 VAR, COV에 의해서 각 차원이 독립적인 feature로 regularize되고 있음.
        # normalize > similarity 계산해서 mixin ratio로 써도 무방할 것
        loss += VAR + COV 


        # loss = recon + reg * self.reg_beta  + prior + recon_state + reg_h_th + reg_state * 0.15 + VAR + INV + COV


        metric = recon.item() + reg.item() * self.reg_beta  + prior_kld.item()


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
            # "Sim" : INV.item(),
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