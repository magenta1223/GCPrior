import gym
from LVD.envs.kitchen import KitchenEnv_GC, KITCHEN_TASKS, KitchenTask_GC
from .kitchen_vis import draw_kitchen, draw_kitchen_low


env = KitchenEnv_GC()
# tasks = KITCHEN_TASKS

tasks = [KitchenTask_GC(t) for t in KITCHEN_TASKS]

config = dict(
    constrained_sac=dict(auto_alpha=True, kl_clip=20,
                         target_kl=5, increasing_alpha=True, prior_state_dim = 30),
    buffer_size=20000,
    n_prior_episode=20,
    time_limit=280,
    n_episode=1000,
    train=dict(batch_size=256, reuse_rate=256),
    prior_state_dim = 30
)
visualize_env = draw_kitchen
visualize_env_low = draw_kitchen_low