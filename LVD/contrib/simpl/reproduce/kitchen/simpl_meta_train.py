import d4rl

import gym
import simpl.env.kitchen
from simpl.env.kitchen import KitchenTasks, KitchenEnv_GC
from .kitchen_vis import draw_kitchen



# from LVD.envs.kitchen import KitchenEnv_GC

# env = gym.make('simpl-kitchen-v0')
env = KitchenEnv_GC() # goal conditioned kitchen
train_tasks = KitchenTasks.train_tasks
config = dict(
    policy=dict(hidden_dim=128, n_hidden=5, prior_state_dim = 30,),
    qf=dict(hidden_dim=128, n_hidden=5),
    n_qf=2,
    encoder=dict(hidden_dim=128, n_hidden=2, init_scale=1, prior_scale=1), 
    simpl=dict(init_enc_prior_reg=1e-3, target_enc_prior_kl=2, # task encoder와 prior간의 kl : ok
               init_enc_post_reg=1e-4, target_enc_post_kl=10, # task encoder와 posterior간의 kl
               init_policy_prior_reg=0.05, target_policy_prior_kl=0.1, # policy와 prior간의 kl : ok
               init_policy_post_reg=0.03, target_policy_post_kl=4, kl_clip=6, prior_state_dim = 30), # policy와 posterior간의 kl 
    enc_buffer_size=3000, # 3000
    buffer_size=3000, # 3000
    e_dim = 6,
    time_limit=280,
    n_epoch=500,
    
    train=dict(batch_size=1024, reuse_rate=256,
               n_prior_batch=3, n_post_batch=27,
               prior_enc_size=2, post_enc_size=1024)
)
visualize_env = draw_kitchen

