
# envs & utils
import gym

from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb

# simpl contribs
from proposed.contrib.simpl.collector import Buffer
from proposed.contrib.simpl.collector.collector import TimeLimitCollector
from proposed.contrib.simpl.torch_utils import itemize

# models
import torch.nn as nn
from proposed.configs.models import Linear_Config
from proposed.modules.base import *
from proposed.modules.subnetworks import *
from proposed.rl.sac_modules.SAC import SAC


import d4rl
from simpl.env.kitchen.kitchen import *



import argparse


# # Generating single task policy with SAC 
ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

env = gym.make('impl-kitchen-v0')


def simpl_fine_tune_iter(collector, trainer, batch_size, reuse_rate):
    log = {}

    # collect
    with trainer.policy.expl():
        episode = collector.collect_episode(trainer.policy)

    trainer.buffer.enqueue(episode)
    log['tr_return'] = sum(episode.rewards)

    if trainer.buffer.size < batch_size:
        return log

    # train
    n_step = int(reuse_rate * len(episode) / batch_size)
    for i in range(max(n_step, 1)):
        stat = trainer.step(batch_size)


    log.update(itemize(stat))

    return log


def train_single_task(env, task, args):
    task = KitchenTask(subtasks = [task])
    # gpu = 0
    time_limit = 280
    # policy_vis_period = 20
    wandb_project_name = "single task SAC"
    buffer_size = 20000
    n_episode = 1000

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    wandb.init(
        project=wandb_project_name
    )

    collector = TimeLimitCollector(env, time_limit=time_limit)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 128
    n_hidden = 3


    # 돌아는 가는데 학습이 안됨.
    # env.set_task

    policy_config = edict(
        n_blocks = n_hidden, 
        in_feature = state_dim, # state dim 
        hidden_dim = hidden_dim, # 128
        out_dim = action_dim * 2, # 10 * 2
        norm_cls = None,
        act_cls = nn.LeakyReLU,
        block_cls = LinearBlock,
        true = True,
    )

    qf_config = edict(
        n_blocks = n_hidden, 
        in_feature = state_dim + action_dim, # state dim 
        hidden_dim = hidden_dim, # 128
        out_dim = 1, # 10 * 2
        norm_cls = None,
        act_cls = nn.LeakyReLU,
        block_cls = LinearBlock,
        true = True,
    )


    # ready networks
    policy = MLPPolicy(Linear_Config(policy_config))        
    qfs = [ MLPQF(Linear_Config(qf_config))  for _ in range(2)]
    
    buffer = Buffer(state_dim, action_dim, buffer_size)


    sac_config = {'auto_alpha': True, 'kl_clip': 20, 'target_kl': 5, 'increasing_alpha': True, "reward" : args.reward}
    self = SAC(policy, None, qfs, buffer, **sac_config)
    config = {'batch_size': 256, 'reuse_rate': 256}

    if args.reward == "dense":
        with env.set_task(task), env.set_dense():
            for episode_i in range(n_episode+1):
                log = simpl_fine_tune_iter(collector, self, **config)
                log['episode_i'] = episode_i
                # if episode_i % policy_vis_period == 0:
                #     plt.close('all')
                #     plt.figure()
                #     log['policy_vis'] = visualize_env(plt.gca(), env, list(buffer.episodes)[-20:])
                wandb.log(log)

    else:
        with env.set_task(task):
            for episode_i in range(n_episode+1):
                log = simpl_fine_tune_iter(collector, self, **config)
                log['episode_i'] = episode_i
                # if episode_i % policy_vis_period == 0:
                #     plt.close('all')
                #     plt.figure()
                #     log['policy_vis'] = visualize_env(plt.gca(), env, list(buffer.episodes)[-20:])
                wandb.log(log)






def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reward", default = "sparse")
    parser.add_argument("-p", "--path", default = "sparse")
    
    args = parser.parse_args()
    task = KitchenTask(subtasks = ["bottom burner"])

    if args.reward == "dense":
        env = gym.make('impl-kitchen-v0') # 
    else:
        env = gym.make("simpl-kitchen-v0")

    for task in ALL_TASKS:
        train_single_task(env, task, args)


if __name__ == "__main__":
    main()