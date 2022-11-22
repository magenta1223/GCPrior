# # goal generator가 잘 하는지 검증
# # 학습된 모델을 불러와서, env.set_task(task)
# # 


# # envs & utils
# import gym
# import numpy as np
# from easydict import EasyDict as edict
# import matplotlib.pyplot as plt
# import wandb

# # simpl contribs
# from simpl.collector import Buffer
# from simpl.nn import itemize
# from reproduce.kitchen import simpl_fine_tune as module
# # models
# import torch.nn as nn
# from proposed.configs.models import Linear_Config
# from proposed.modules.base import *
# from proposed.modules.subnetworks import *
# from proposed.rl.SAC import SAC

# import d4rl
# from simpl.env.kitchen.kitchen import *
# from proposed.envs.base import *
# import argparse
# from proposed.rl.vis import *
# from proposed.utils import *

# from proposed.collector.collector import LowFixedHierarchicalTimeLimitCollectorGC


# def set_state_force(self, state, use_goal = False):
#     if use_goal:
#         given_qpos = state[30:].copy()
#     else:
#         given_qpos = state[:30].copy()
    
#     reset_vel = self.init_qvel[:].copy()
#     self.robot.reset(self, given_qpos, reset_vel)

# setattr(gym.Env, "set_state_force", set_state_force)
# ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


# env = gym.make("kitchen-single-v0")

# load = torch.load("/home/magenta1223/skill-based/SiMPL/proposed/weights/log61_end.bin")
# model = load['model'].eval()

# fig, ax = plt.subplots(4, 2)

# for i, task in enumerate(ALL_TASKS):
#     task_obj = KitchenTask([task])

#     with env.set_task(task_obj):

#         init_state = env.reset()
#         generated_goal_state = model.goal_generator.mix_pseudo_g_obj(init_state)
#         generated_goal_state = generated_goal_state.detach().cpu().numpy()[0]
#         env.set_state_force(generated_goal_state, True)
#         img_pseudo_goal = env.render(mode = "rgb_array")
        
#         env.reset()
#         env.set_state_force(init_state, True)
#         img_goal = env.render(mode = "rgb_array")

#     col, row = divmod(i, 4)

#     ax[row][col] .set_title(task)
#     ax[row][col].axis("off")
#     ax[row][col].imshow(img_pseudo_goal)

# plt.show()