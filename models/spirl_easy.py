from proposed.configs.models import *
from proposed.modules.base import *
from proposed.modules.subnetworks import *
import torch
import torch.nn as nn
from torch.optim import *
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from proposed.utils import *

# 앞이 estimate = q_hat_dist
# target은 q_dist에서 샘플링한 값. 


def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

class SkillPrior(BaseModule):
    """
    """

    def __init__(self, model_config):
        super(SkillPrior, self).__init__(model_config)

        self.use_amp = True

        # self.init_grad_clip = model_config.init_grad_clip
        # self.init_grad_clip_step = model_config.init_grad_clip_step
        # self.reg_beta = model_config.reg_beta


        self.step = 0
        self.Hsteps = 10

        # self.cond_decode = model_config.cond_decode # 

        # submodules
        ## skill prior module
        prior_config = edict(
            z_dim = self.latent_dim, 
            n_blocks = self.n_processing_layers, 
            in_feature = self.state_dim, # state dim 
            hidden_dim = self.hidden_dim, # 128
            out_dim = self.latent_dim * 2, # 10 * 2
            norm_cls = nn.BatchNorm1d,
            act_cls = nn.LeakyReLU,
            block_cls = LinearBlock,
            true = True,
        )

        encoder_config = edict(
            z_dim = self.latent_dim, 
            in_feature = self.action_dim + self.state_dim if self.cond_decode else  self.action_dim, # state_dim + latent_dim 
            hidden_dim = self.hidden_dim,
            out_dim = self.latent_dim * 2, 
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


        skill_prior = PriorNetwork(Linear_Config(prior_config))

        self.skill_prior = PriorWrapper(mode = "default", prior_policy = skill_prior)

        ## skill encoder
        self.skill_encoder = SequentialBuilder(RNN_Config(encoder_config))

        ## closed-loop skill decoder
        self.skill_decoder = DecoderNetwork(Linear_Config(decoder_config))

        # optimizer
        self.optimizer = RAdam(self.parameters(), lr = 1e-3)

        # Losses

        prior_losses = ["nll", "kld"]

        loss_fns = {
            "recon" : {
                'mse' : nn.MSELoss(),
                'nll' : nll_loss2,
            },

            "reg" : {
                'kld' : torch_dist.kl_divergence ,
            },

            "prior" : {
                'kld' : torch_dist.kl_divergence ,
                # 'nll' : nll_loss,   
                'nll' : nll_dist
            }

        }



        self.loss_fns = {
            'recon' : ['mse', nn.MSELoss()],
            'reg' : ['kld', torch_dist.kl_divergence] ,
            'prior' : ["nll", nll_dist] ,
            'prior_metric' : ["kld", torch_dist.kl_divergence]
        }



        self.scaler = GradScaler(self.use_amp)


    def beta(self, e):
        return kl_annealing(e, self.reg_beta, self.target_kl, 0.9)

    def dec_input(self, states, z, steps):
        return torch.cat((states[:,:steps], z[:, None].repeat(1, steps, 1)), dim=-1)

    def get_loss_fn(self, key, index = 1):
        return self.loss_fns[key][index]

    def forward(self, states, actions):

        if self.cond_decode:
            enc_inputs = torch.cat( (actions, states[:,:-1]), dim = -1)
            q = self.skill_encoder(enc_inputs)[:, -1]

        else:
            # skill posterior
            q =  self.skill_encoder(actions)[:,-1]

        # skill prior
        outputs = self.skill_prior(states[:,0])

        outputs['post'] = get_dist(q)
        outputs['post_detach'] = get_dist(q, detached= True)
        outputs['fixed'] = get_fixed_dist(q)


        z = outputs.post.rsample() # samples by reparameterize trick. 

        decode_inputs = self.dec_input(states, z, self.Hsteps)

        N, T = decode_inputs.shape[:2]

        # prevent lookahead     
        outputs['skill_hat'] = self.skill_decoder(decode_inputs.view(N * T, -1)).view(N, T, -1)


        with torch.no_grad():
            dec_inputs_prior = self.dec_input(states, outputs.prior_detach.sample(), self.Hsteps)
            outputs['skill_estimate_prior'] = self.skill_decoder(dec_inputs_prior.view(N * T, -1)).view(N, T, -1)


        return outputs, N, T


    def compute_loss(self, outputs, skill):
        # ----------- SPiRL -------------- # 

        recon = self.get_loss_fn('recon')(outputs.skill_hat, skill)
        reg = self.get_loss_fn('reg')(outputs.post, outputs.fixed).mean()
        prior = self.get_loss_fn('prior')(
            outputs.post_detach, # target distributions
            outputs.prior, # distributions to optimize
        ).mean()

        with torch.no_grad():
            prior_kld  = self.get_loss_fn('reg')(outputs.post_detach, outputs.prior_detach).mean() 


        # ----------- Add -------------- # 
        
        loss = recon + reg * self.reg_beta  + prior #+ recon_masked
        metric = recon.item() + reg.item() * self.reg_beta + prior_kld.item()

        # self.prev_beta = beta

        loss_dict = {           
            "loss" : loss.item(),
            "Rec_skill" : recon.item(),
            "Reg" : reg.item(),
            "Pri" : prior.item(),
            "Pri_kld" : prior_kld.item(),
            "metric" : metric,
            # "beta" : beta
        }       

        return loss, loss_dict




    def optimize(self, batch, e):
        # inputs & targets          
        states, actions, masks = batch.values()
        states, actions, masks = states.cuda(), actions.cuda(), masks.cuda()
        
        with autocast(self.use_amp):
            # forwarding
            outputs, N, T = self(states, actions)
            # loss
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

        return loss_dict
    
    def validate(self, batch, e):
        # inputs & targets          
        states, actions, masks = batch.values()
        states, actions, masks = states.cuda(), actions.cuda(), masks.cuda()
        
        # forwarding
        with autocast(self.use_amp):
            outputs, N, T = self(states, actions)
            loss, loss_dict = self.compute_loss(outputs, actions)

        return loss_dict