
# envs & utils
import gym
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb

# simpl contribs
from proposed.contrib.simpl.collector import Buffer
from proposed.contrib.simpl.torch_utils import itemize

# models
import torch.nn as nn
from proposed.configs.models import Linear_Config
from proposed.modules.base import *
from proposed.modules.subnetworks import *
# from proposed.rl.sac_modules.gcid import SAC
from proposed.rl.sac_module import SAC


import d4rl
from simpl.env.kitchen.kitchen import *
from proposed.envs.base import *
import argparse
from proposed.rl.vis import *
from proposed.utils import *
from proposed.collector.gcid import LowFixedHierarchicalTimeLimitCollector



# # Generating single task policy with SAC 
ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


def render_task(env, policy, low_actor, G):
    imgs = []
    state = env.reset()
    done = False
    time_step = 0
    time_limit = 280
    print("rendering..")


    while not done and time_step < time_limit: 
        if time_step % 10 == 0:
            high_action = policy.act(state, G)

        with low_actor.condition(high_action), low_actor.expl():
            low_action = low_actor.act(state)
        
        state, reward, done, info = env.step(low_action)
        img = env.render(mode = "rgb_array")
        imgs.append(img)
        time_step += 1
    print("done!")
    return imgs


def simpl_fine_tune_iter(collector, trainer, batch_size, reuse_rate, task, G):
    # ------------- Initialize log ------------- #
    log = {}

    # ------------- Collect Data ------------- #
    with trainer.policy.expl(), collector.low_actor.expl() : #, collector.env.step_render():
        episode = collector.collect_episode(trainer.policy, G)

    if np.array(episode.dones).sum() != 0: # success 
        print("success")
        # infos = episode.infos
        # imgs = []
        # for info in infos:
        #     imgs.append(info['img'])
        # visualize(imgs = imgs, task_name = task, prefix = 'success')
    
    trainer.buffer.enqueue(episode) 
    log['tr_return'] = sum(episode.rewards)

    if trainer.buffer.size < batch_size:
        # 모종의 이슈로 버퍼에 수집이 덜 됨. 
        return log

    # train
    n_step = int(reuse_rate * len(episode) / batch_size)

    step_inputs = edict(
        batch_size = batch_size,
        G = G
    )
    
    
    for i in range(max(n_step, 1)):
        stat = trainer.step(step_inputs)
        # trainer.buffer에 수집된 데이터 있음
        # 모든 시점에 대해 H-distance를 가지는 데이터 페어를 가지고
        # transition probability & state encoder를 별개로 업데이트


    log.update(itemize(stat))

    return log

def train_single_task(env, tasks, args):

    # ------------- Set Task ------------- #
    if len(tasks) == 1:
        task_obj = KitchenTask(subtasks = [tasks[0]])
    else:
        task_obj = KitchenTask(subtasks= tasks)
    
    # ------------- Hyper Parameteres ------------- #
    time_limit = int(args.time_limit)
    wandb_project_name = "single task SAC"
    buffer_size = 20000
    n_episode = 1000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = int(args.hidden_dim)
    n_hidden = int(args.n_hidden)
    latent_dim = 10

    # ------------- Logger ------------- #
    wandb.init(
        project=wandb_project_name,
        name = "multi-task"
    )


    # ------------- Module Configuration ------------- #
    policy_config = edict(
        n_blocks = n_hidden, 
        in_feature = state_dim, # state dim 
        hidden_dim = hidden_dim, # 128
        out_dim = latent_dim * 2 ,
        norm_cls = None,
        act_cls = nn.LeakyReLU,
        block_cls = LinearBlock,
        true = True,
    )

    qf_config = edict(
        n_blocks = n_hidden, 
        in_feature = state_dim + latent_dim, # state dim 
        hidden_dim = hidden_dim, # 128
        out_dim = 1, # 10 * 2
        norm_cls = None,
        act_cls = nn.LeakyReLU,
        block_cls = LinearBlock,
        true = True,
    )

    # ------------- Modules ------------- #

    ## ------------- X-spirl Modules ------------- ##
    load = torch.load(f"/home/magenta1223/skill-based/SiMPL/{args.path}")
    model = load['model'].eval()

    ## ------------- Q-functions ------------- ##
    qfs = [ MLPQF(Linear_Config(qf_config))  for _ in range(2)]

    ## ------------- High Policy ------------- ##
    policy = MMPHSG(Linear_Config(policy_config), model.skill_prior)        

    ## ------------- Prior Policy ------------- ##
    prior_policy = model.skill_prior
    # TODO
    # state encoder, subgoal generator는 gradient 계산 해서 forwarding 하게 바꿔야 함
    # 그래야겠지?
    # 그러면 inference method를 다시 짜야 함

    ## ------------- Low Decoder ------------- ##
    low_actor = model.skill_decoder

    # ----------------------------------- # 

    # ------------- Buffers & Collectors ------------- #
    buffer = Buffer(state_dim, action_dim, buffer_size)
    collector = LowFixedHierarchicalTimeLimitCollector(env, low_actor, horizon=10, time_limit=time_limit)

    
    # ------------- Goal setting ------------- #
    with env.set_task(task_obj):
        init_state = env.reset()

    _G = init_state[30:]
    G = np.zeros_like(init_state)
    G[:30] = _G
    G = prep_state(G, model.device)


    # ------------- RL agent ------------- #
    sac_config = {'auto_alpha': True, 'kl_clip': 20, 'target_kl': 5, 'increasing_alpha': True,}
    self = SAC(policy, prior_policy, qfs, buffer, **sac_config)
    self = self.cuda()
    config = {'batch_size': 256, 'reuse_rate': 256, 'task' : tasks, "G" : G}


    # ------------- Train RL ------------- #
    with env.set_task(task_obj):
        # log에 success rate추가 .
        for episode_i in range(n_episode+1):
            log = simpl_fine_tune_iter(collector, self, **config)
            log['episode_i'] = episode_i
            # visualize
            if (episode_i + 1) % 5 == 0:
                # sr = success_rate(env, policy, low_actor, time_limit)
                # print(sr)
                imgs = render_task(env, self.policy, low_actor, G)
                visualize(imgs = imgs, task_name = "_".join(tasks))
            if args.wandb:
                wandb.log(log)
        if args.wandb:
            wandb.finish()    
        


args = edict(
    wandb = False, 
    path = "proposed/weights/log66_end.bin",
    time_limit = 280,
    n_hidden  =3, 
    hidden_dim = 128 
)


env = gym.make("simpl-kitchen-v0")
# env = gym.make("kitchen-single-v0")

# ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
# for task in ALL_TASKS:
tasks = [ 'slide cabinet']
# tasks = ['slide cabinet']
# ------------- Set Task ------------- #
if len(tasks) == 1:
    task_obj = KitchenTask(subtasks = [tasks[0]])
else:
    task_obj = KitchenTask(subtasks= tasks)

# ------------- Hyper Parameteres ------------- #
time_limit = int(args.time_limit)
wandb_project_name = "single task SAC"
buffer_size = 20000
n_episode = 1000
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = int(args.hidden_dim)
n_hidden = int(args.n_hidden)
latent_dim = 10

# ------------- Logger ------------- #
wandb.init(
    project=wandb_project_name,
    name = "multi-task"
)


# ------------- Module Configuration ------------- #
policy_config = edict(
    n_blocks = n_hidden, 
    in_feature = state_dim, # state dim 
    hidden_dim = hidden_dim, # 128
    out_dim = latent_dim * 2 ,
    norm_cls = None,
    act_cls = nn.LeakyReLU,
    block_cls = LinearBlock,
    true = True,
)

qf_config = edict(
    n_blocks = n_hidden, 
    in_feature = state_dim + latent_dim, # state dim 
    hidden_dim = hidden_dim, # 128
    out_dim = 1, # 10 * 2
    norm_cls = None,
    act_cls = nn.LeakyReLU,
    block_cls = LinearBlock,
    true = True,
)

# ------------- Modules ------------- #

## ------------- X-spirl Modules ------------- ##
load = torch.load(f"/home/magenta1223/skill-based/SiMPL/{args.path}")
model = load['model'].eval()

## ------------- Q-functions ------------- ##
qfs = [ MLPQF(Linear_Config(qf_config))  for _ in range(2)]

## ------------- High Policy ------------- ##
policy = MMPHSG(Linear_Config(policy_config), model.skill_prior)        

## ------------- Prior Policy ------------- ##
prior_policy = model.skill_prior
# TODO
# state encoder, subgoal generator는 gradient 계산 해서 forwarding 하게 바꿔야 함
# 그래야겠지?
# 그러면 inference method를 다시 짜야 함

## ------------- Low Decoder ------------- ##
low_actor = model.skill_decoder

# ----------------------------------- # 

# ------------- Buffers & Collectors ------------- #
buffer = Buffer(state_dim, action_dim, buffer_size)
collector = LowFixedHierarchicalTimeLimitCollector(env, low_actor, horizon=10, time_limit=time_limit)


# ------------- Goal setting ------------- #
with env.set_task(task_obj):
    init_state = env.reset()

_G = init_state[30:]
G = np.zeros_like(init_state)
G[:30] = _G
G = prep_state(G, model.device)


# ------------- RL agent ------------- #
sac_config = {'auto_alpha': True, 'kl_clip': 20, 'target_kl': 5, 'increasing_alpha': True,}
self = SAC(policy, prior_policy, qfs, buffer, **sac_config)
self = self.cuda()
config = {'batch_size': 256, 'reuse_rate': 256, 'task' : tasks, "G" : G}



self.buffer