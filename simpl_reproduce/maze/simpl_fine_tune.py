import gym
# import simpl.env.maze
# from simpl.env.maze import Size20Seed0Tasks

from .maze_vis import draw_maze
from LVD.envs import *

env = Maze_GC(**maze_config)
# tasks = Size20Seed0Tasks.flat_test_tasks
tasks = [ MazeTask_Custom(task)  for task in MAZE_TASKS]


config = dict(
    constrained_sac=dict(auto_alpha=True, kl_clip=5,
                         target_kl=1, increasing_alpha=True),
    buffer_size=20000,
    n_prior_episode=20,
    time_limit=2000,
    n_episode=1000,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = draw_maze
