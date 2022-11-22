from torch.optim import *
from proposed.prior_train.trainer import *
from proposed.configs.models import *
from proposed.modules.base import *
# from proposed.models.gcspirl import *
from proposed.models.gcid import GCIDSkillPrior

import argparse
from proposed.contrib.simpl.env.kitchen import *

import gym
import d4rl
from proposed.envs.base import *

from proposed.utils import *


model = torch.load("/home/magenta1223/skill-based/SiMPL/proposed/weights/log35_31.bin")

# unseen pair에 대한 실험
env = gym.make("kitchen-single-v0")
loss_fn  = nn.MSELoss()


_model = model['model'].eval()

prior_module = _model.skill_prior
state_enc = prior_module.state_encoder
state_dec = prior_module.state_decoder
subgoal_g = prior_module.subgoal_generator
target_state_enc = prior_module.target_state_encoder



def test(state, G):
    loss_fn = nn.MSELoss()
    N, T, _ = state.shape
    with torch.no_grad():
        # state reconstrunction test
        state_hat = state_dec(state_enc(state.view(N * T, -1))).view(N, T, -1)
        recon_loss = loss_fn(state, state_hat)
    
        # cycle test : state enc
        ht = state_enc(state.view(N * T, -1)).view(N, T, -1)
        h_tH_hat = subgoal_g(torch.cat((ht[:,0], target_state_enc(G)), dim = -1 ))
        h_tH_hat_hat = state_enc(state_dec(h_tH_hat))
        cycle_loss = loss_fn(h_tH_hat, h_tH_hat_hat)


        # cycle test : target state enc
        ht = target_state_enc(state.view(N * T, -1)).view(N, T, -1)
        h_tH_hat = subgoal_g(torch.cat((ht[:,0], target_state_enc(G)), dim = -1 ))
        h_tH_hat_hat = target_state_enc(state_dec(h_tH_hat))
        cycle_loss2 = loss_fn(h_tH_hat, h_tH_hat_hat)

    return recon_loss, cycle_loss, cycle_loss2

# ------------- UNKNOWN TASK ------------- #

task_obj = KitchenTask(["slide cabinet"])
with env.set_task(task_obj):
    state = env.reset()

state_unk = prep_state(state, device = _model.device)
_G_unk = deepcopy(state_unk)
G_unk = torch.zeros_like(_G_unk)
G_unk[:, :30] = _G_unk[:, 30:]



# recon 0.0739, state_enc 0.3608, state_enc_ma 0.0078
# recon 0.0006, state_enc 0.4025, state_enc_ma 0.0008

# ------------- KNOWN TASK ------------- #


task_obj = KitchenTask(["kettle"])
with env.set_task(task_obj):
    state = env.reset()

state_known = prep_state(state, device = _model.device)
_G_known = deepcopy(state_known)
G_known = torch.zeros_like(_G_known)
G_known[:, :30] = _G_known[:, 30:]



# ------------- TRAINING SET ------------- #

args = edict(
    reg_beta = 0.0005,
    epochs = 70,
    mode = "gcid",
    min = 0,
    max = -1,
    warmup = 20
)

args['goal_range'] = (int(args.min), int(args.max))

train_conf, train_loader = get_loader("aa", "train", **args)
test_conf, test_loader = get_loader("aa", "test", **args)

for batch in train_loader:
    break


relabeled_inputs, states, actions, G = batch.values()
relabeled_inputs, states, actions, G = relabeled_inputs.cuda(), states.cuda(), actions.cuda(), G.cuda()



test(state_unk.unsqueeze(1), G_unk)
test(state_known.unsqueeze(1), G_known)
test(states, G)



# UNK recon 0.0739, state_enc 0.3608, state_enc_ma 0.0078
# KWN recon 0.0006, state_enc 0.4025, state_enc_ma 0.0008
# TRN recon 0.0168, state_enc 0.3373, state_enc_ma 0.0023

# 확실히 UNK가 recon이 잘 안됨. 
# 그러면 recon 값으로 penalized
# 그냥 task conditioned로 하자. 