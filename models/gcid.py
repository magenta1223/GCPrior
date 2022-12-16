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

from adabelief_pytorch import AdaBelief


from proposed.contrib.sam import SAM

# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 


def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

class GCID_SkillPrior(BaseModule):
    """
    """

    def __init__(self, model_config):
        super().__init__(model_config)

        self.use_amp = False

        self.step = 0
        self.Hsteps = 10

        if self.norm == "bn":
            norm_cls = nn.BatchNorm1d
        else:
            norm_cls = nn.LayerNorm

        # ----------------- SUBMODULES ----------------- #

        ## ----------------- Configurations ----------------- ##

        ### ----------------- skill prior modules ----------------- ###

        # state encoder
        state_encoder_config = edict(
            n_blocks = self.n_processing_layers,
            in_feature = self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2 if self.distributional else self.latent_dim, # * 2 when variational inference
            norm_cls =  norm_cls,
            act_cls = nn.Mish, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        # state decoder
        state_decoder_config = edict(
            n_blocks = self.n_processing_layers,
            in_feature = self.latent_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.state_dim,
            norm_cls = norm_cls,
            act_cls = nn.Mish, #nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        # subgoal generator
        subgoal_generator_config = edict(
            n_blocks = 5,
            in_feature = self.latent_dim * 2, # state_dim + latent_dim 
            # in_feature = self.state_dim * 2, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.latent_dim * 2 if self.distributional else self.latent_dim,
            norm_cls = norm_cls,
            act_cls = nn.Mish, #nn.LeakyReLU,
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
            norm_cls = norm_cls,
            act_cls = nn.Mish,
            block_cls = LinearBlock,
            true = True,
        )



        ### ----------------- posterior modules ----------------- ###

        encoder_config = edict(
            # in_feature = self.action_dim + self.state_dim,
            in_feature = self.action_dim + self.latent_dim if self.use_learned_state else self.action_dim + self.state_dim,
            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim * 2, # * 2 when variational inference
            n_layers= 1,
            bias = False,
            batch_first = True,
            dropout = 0,
            linear_cls = LinearBlock,
            rnn_cls = nn.LSTM,
            act_cls = nn.Mish,
            norm_cls = norm_cls,
            true = True,
        )

        decoder_config = edict(
            n_blocks = 5, #self.n_processing_layers,
            state_dim = self.state_dim,
            z_dim = self.latent_dim, 
            in_feature = self.latent_dim * 2 if self.use_learned_state else self.latent_dim + self.state_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim, 
            out_dim = self.action_dim,
            norm_cls = norm_cls,
            act_cls = nn.Mish,
            block_cls = LinearBlock,
            true = True,
        )



        ## ----------------- Builds ----------------- ##

        inverse_dynamics = PriorNetwork(Linear_Config(inverse_dynamics_config))
        inverse_dynamics_eval = PriorNetwork(Linear_Config(inverse_dynamics_config))

        state_encoder = SequentialBuilder(Linear_Config(state_encoder_config))

        state_decoder = SequentialBuilder(Linear_Config(state_decoder_config))
        subgoal_g = SequentialBuilder(Linear_Config(subgoal_generator_config))

        self.skill_prior = PriorWrapper(
            mode = "gcid", #"vic" if self.vic else "gcid",
            prior_policy = inverse_dynamics,
            prior_policy_eval = inverse_dynamics_eval,
            state_encoder = state_encoder,
            state_decoder = state_decoder,
            subgoal_generator = subgoal_g,
            ema_update = True,
            # prior_proj = prior_proj if self.vic else None,
            direct = self.direct,
            distributional = self.distributional
        )
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))
        self.skill_decoder_eval = DecoderNetwork(Linear_Config(decoder_config))

        self.skill_decoder_eval.requires_grad_ = False
        
        # optimizer
        self.optimizer = RAdam(
        # self.optimizer = AdaBelief(
        # self.optimizer = AdamW(
        # self.optimizer = Adam(
            [
                {'params' : self.skill_prior.parameters(), "weight_decay" : self.wdp},
                {'params' : self.skill_encoder.parameters(),  "weight_decay" : self.wde},
                {'params' : self.skill_decoder.parameters(),  "weight_decay" : self.wdd},

            ],            
            lr = model_config.lr,
            # rectify= False
        )

        print(self.optimizer)

        # Losses

        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
            'prior_metric' : ["kld", torch_dist.kl_divergence],
            'wasserstein' : ['wst', W2Normal],
            'l2penalty' : ['l2', L2Norm],
        }


        # self.scaler = GradScaler(self.use_amp)


    def dec_input(self, states, hidden_states, z, steps, detach = False):
        if detach:
            z = z.clone().detach()
        
        if self.use_learned_state:
            return torch.cat((hidden_states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        else:
            return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def loss_fn(self, key, index = 1):
        return self.loss_fns[key][index]


    def get_metrics(self, loss_dict, outputs, states, state_labels):
        # ----------- Metrics ----------- #
        self.skill_decoder_eval.eval()
        # self.skill_decoder_eval.requires_grad_ = False

        with torch.no_grad():

            loss_dict['Pri']  = self.loss_fn('reg')(outputs.post_detach, outputs.prior_detach).mean()
            loss_dict['Pri_hat'] = self.loss_fn('reg')(outputs.prior_hat, outputs.prior_detach).mean()
            loss_dict['Reg_Pri_hat'] = self.loss_fn('reg')(outputs.post_detach, outputs.prior_hat).mean()



            non_branching_indices = state_labels[:, 0] == 1
            non_branching_indices = non_branching_indices.cpu().numpy()

            prior_mean = outputs.prior_detach.base_dist.loc.cpu().numpy()
            prior_std = outputs.prior_detach.base_dist.scale.cpu().numpy()
            prior_hat_std = outputs.prior_hat.base_dist.scale.cpu().numpy()

            loss_dict['prior_mean'] = prior_mean[non_branching_indices].mean()
            loss_dict['prior_std'] = prior_std[non_branching_indices].mean()
            loss_dict['prior_hat_std'] = prior_hat_std[non_branching_indices].mean()
            
            # decode_inputs = self.dec_input(states, outputs.prior_detach.sample(), self.Hsteps, True)
            # N, T = decode_inputs.shape[:2]
            # skill_hat_prior = self.skill_decoder_eval(decode_inputs.view(N * T, -1)).view(N, T, -1)
            # loss_dict['prior_recon'] = F.mse_loss(skill_hat_prior, actions)

            # decode_inputs = self.dec_input(states, outputs.prior_hat.sample(), self.Hsteps, True)
            # skill_hat_prior = self.skill_decoder_eval(decode_inputs.view(N * T, -1)).view(N, T, -1)
            # loss_dict['prior_hat_recon'] = F.mse_loss(skill_hat_prior, actions)

            # prior를 업데이트할 때에만 문제가 생김.
            # 개웃긴점은 똑같은 짓을 validation에서 해도 안생김 (?) 
            # 하지말자 (?)

        loss_dict['metric'] = loss_dict['Rec_skill'] + loss_dict['Reg'] * self.reg_beta  + loss_dict['Pri']
        # self.optimizer.zero_grad()
        return loss_dict

    def forward(self, relabeled_inputs, post_inputs, actions, G):
        
        inputs = edict(
            states = relabeled_inputs,
            G = G
        )
        
        # skill prior
        outputs = self.skill_prior(inputs)

        # ------------------------ DO NOT EDIT ------------------------ #
        if self.use_learned_state:
            enc_inputs = torch.cat( (actions, outputs.hs_reshaped[:,:-1]), dim = -1)
        else:
            enc_inputs = torch.cat( (actions, post_inputs[:,:-1]), dim = -1)


    
        q = self.skill_encoder(enc_inputs)[:, -1]

        outputs['post'] = get_dist(q)
        outputs['post_detach'] = get_dist(q, detached= True)
        outputs['fixed'] = get_fixed_dist(q)


        z = outputs.post.rsample() # samples by reparameterize trick. 

        decode_inputs = self.dec_input(post_inputs, outputs.hs_reshaped, z, self.Hsteps)

        N, T = decode_inputs.shape[:2]

        # prevent lookahead     
        outputs['skill_hat'] = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)

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


        if self.distributional:
            # State Regularization
            reg_state = self.loss_fn('reg')(outputs.hsd, outputs.hsd_target).mean()

            # predicted h_tH NLL
            if self.direct:
                reg_h_th = self.loss_fn(self.subgoal_loss)(outputs.prior_detach, outputs.prior_hat).mean()
            else:
                reg_h_th = self.loss_fn(self.subgoal_loss)(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()

        else:
            # predicted h_tH NLL
            # 안되면 일단 그 뭐냐.. prior_hat의 sigma 찍어봐야 됨. 
            if self.direct:
                reg_h_th = self.loss_fn(self.subgoal_loss)(outputs.prior_detach, outputs.prior_hat).mean()
            else:
                reg_h_th = self.loss_fn(self.subgoal_loss)(outputs.hsd_tH, outputs.h_tH_hat_dist).mean()



        # SPiRL
        loss = recon + reg * self.reg_beta  + prior

        # state reconstruction + imaginary state
        loss += recon_state 
        loss += reg_h_th  # direct로 변경 ? 

        if self.distributional:
            loss += reg_state * self.state_reg 

        if self.L2:
            penlaty = self.loss_fn('l2penalty')(self.skill_prior.state_encoder) +  self.loss_fn('l2penalty')(self.skill_prior.subgoal_generator) + self.loss_fn('l2penalty')(self.skill_prior.prior_policy)
            # penlaty = penlaty * 0.01
            loss += penlaty
        
        # metric = recon.item() + reg.item() * self.reg_beta  + prior_kld.item()


        loss_dict = {           
            # total
            "loss" : loss.item(),

            # spirl
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Reg_H_tH" : reg_h_th.item(),

        }       

        return loss, loss_dict



    def optimize(self, batch, e):
        # inputs & targets       
        relabeled_inputs, states, actions, G, state_labels = batch.values()
        relabeled_inputs, states, actions, G, state_labels = relabeled_inputs.cuda(), states.cuda(), actions.cuda(), G.cuda(), state_labels.cuda()
        
        
        # forwarding
        outputs, N, T = self(relabeled_inputs, states, actions, G)
        loss, loss_dict = self.compute_loss(outputs, actions)
        loss.backward()

        if self.step < self.init_grad_clip_step:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.init_grad_clip) # 학습 초기에 prior loss가 너무 큼. 

        self.optimizer.step()
        self.optimizer.zero_grad()

        with torch.no_grad():
            loss_dict = self.get_metrics(loss_dict, outputs, states, state_labels)
            self.skill_prior.ma_state_enc()
            self.skill_prior.copy_weights()
            self.skill_decoder_eval.load_state_dict(self.skill_decoder.state_dict())
            for p in self.skill_decoder_eval.parameters():
                p.requires_grad_(False)


        self.step += 1
        



        return loss_dict
    
    def validate(self, batch, e):
        # inputs & targets          
        relabeled_inputs, states, actions, G, state_labels = batch.values()
        relabeled_inputs, states, actions, G, state_labels = relabeled_inputs.cuda(), states.cuda(), actions.cuda(), G.cuda(), state_labels.cuda()
        
        # forwarding
        # with autocast(self.use_amp), torch.no_grad():
        with torch.no_grad():
            # forwarding
            outputs, N, T = self(relabeled_inputs, states, actions, G)
            loss, loss_dict = self.compute_loss(outputs, actions)
            loss_dict = self.get_metrics(loss_dict, outputs, states, state_labels)

        return loss_dict