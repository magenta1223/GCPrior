# envs & utils
import gym
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import wandb
import argparse
import cv2

# simpl contribs
# from proposed.contrib.simpl.collector import Buffer

# models
import torch.nn as nn
from torch.nn import functional as F

import d4rl




from proposed.LVD.envs import ENV_TASK
from LVD.configs.build import Linear_Config
from LVD.modules.base import *
from LVD.modules.policy import *
from LVD.modules.subnetworks import *
# from LVD.rl.sac_gcid import SAC
from LVD.rl.sac_sc import SAC

from LVD.contrib.simpl.torch_utils import itemize
# from LVD.rl.vis import *
from LVD.utils import *
# from LVD.collector.gcid import LowFixedHierarchicalTimeLimitCollector
from LVD.collector.spirl import LowFixedHierarchicalTimeLimitCollector
from LVD.collector.storage import Buffer_H
from proposed.LVD.rl.rl_utils import *


from LVD.configs.env import ENV_CONFIGS

from LVD.contrib.simpl.reproduce.maze.maze_vis import draw_maze



def render_task(env, env_name, policy, low_actor, tanh = False):
    imgs = []
    state = env.reset()

    low_actor.eval()
    policy.eval()

    done = False
    time_step = 0
    # time_limit = 280
    
    if env_name == "kitchen":
        time_limit = 280
    else:
        time_limit = 2000

    print("rendering..")
    count = 0

    while not done and time_step < time_limit: 
        if time_step % 10 == 0:
            if tanh:
                high_action_normal, high_action, loc, scale = policy.act(state)
            else:
                high_action, loc, scale = policy.act(state)

        with low_actor.condition(high_action), low_actor.expl():
            low_action = low_actor.act(state)
        
        state, reward, done, info = env.step(low_action)
        img = env.render(mode = "rgb_array")
        imgs.append(img)
        time_step += 1
    print("done!")

    return imgs

seed_everything()



def train_policy_iter(collector, trainer, episode_i, batch_size, reuse_rate, project_name, precollect):
    # ------------- Initialize log ------------- #
    log = {}

    # ------------- Collect Data ------------- #
    with trainer.policy.expl(), collector.low_actor.expl() : #, collector.env.step_render():
        episode, G = collector.collect_episode(trainer.policy)

    if np.array(episode.dones).sum() != 0: # success 
        print("success")

    trainer.buffer.enqueue(episode.as_high_episode()) 
    log['tr_return'] = sum(episode.rewards)

    # train
    n_step = int(reuse_rate * len(episode) / batch_size)

    if trainer.buffer.size < batch_size or episode_i < precollect or "zeroshot" in project_name:
        return log, False
    
    
    if episode_i == precollect:
        step_inputs = dict(
            batch_size = batch_size,
            G = G,
            episode = episode_i,
        )
        # Q-warmup
        trainer.warmup_Q(step_inputs)


    for i in range(max(n_step, 1)):
        step_inputs = dict(
            batch_size = batch_size,
            G = G,
            episode = episode_i,
        )
        stat = trainer.step(step_inputs)

    # log.update(itemize(stat))

    return log, True



def train_single_task(env, env_name, tasks, task_cls, args):

    # ------------- Set Task ------------- #

    print(tasks)
    
    if env_name == "maze":
        init_loc_candidate = np.array(env.reset_locations)
        goal_loc = tasks[1]
        dist = np.abs(init_loc_candidate - goal_loc).sum(axis = 1)
        
        # filter
        # cond1 = dist >= 20
        # cond2 = dist < 24
        cond1 = dist >= 24
        cond2 = dist < 28

        init_loc_candidate = init_loc_candidate[np.where(cond1 & cond2)]
        
        # sort 
        dist = np.abs(init_loc_candidate - goal_loc).sum(axis = 1)
        init_loc_candidate = init_loc_candidate[dist.argsort()][-10:]
        # sample 
        init_loc = init_loc_candidate[random.sample(range(len(init_loc_candidate)), 1)[0]]


        # init_loc_candidate = np.array(env.reset_locations)
        # goal_loc = tasks[1]
        # dist = np.abs(init_loc_candidate - goal_loc).sum(axis = 1)
        # init_loc_candidate = init_loc_candidate[dist.argsort()][-10:]
        # # 그래도 할만한 애들을 뽑아야지.. 
        # init_loc = init_loc_candidate[random.sample(range(len(init_loc_candidate)), 1)[0]]




        # init_loc = [10, 10]
        task_obj = task_cls(init_loc, goal_loc)

        # init_loc, goal_loc = tasks[0], tasks[1]
        # task_obj = task_cls(init_loc, goal_loc)
    else:
        task_obj = task_cls(tasks)


    ## ------------- Spirl Modules ------------- ##
    # load = torch.load(f"/home/magenta1223/skill-based/SiMPL/{args.path}")
    load = torch.load(args.path)

    model = load['model'].eval()

    # ------------- Hyper Parameteres ------------- #
    buffer_size = 20000
    n_episode = args.n_episode
    # state_dim = env.observation_space.shape[0]
    # if env_name == "kitchen":
    #     state_dim = 60
    #     prior_state_dim = 30
    # else:
    #     # state_dim =  env.observation_space.shape[0]
    #     # state_dim = 39 + 6
    #     # prior_state_dim = 39
    #     state_dim = 4
    #     prior_state_dim = 4

    


    latent_dim = 10
    
    # ------------- Module Configuration ------------- #
    policy_config = edict(
        n_blocks = args.n_hidden, 
        in_feature = args.policy_state_dim, # state dim 
        hidden_dim = args.hidden_dim, # 128
        out_dim = latent_dim * 2 ,
        norm_cls = nn.LayerNorm,
        act_cls = nn.Mish,
        block_cls = LinearBlock,
        bias = True,
        # learned_state = args.learned_state,
        dropout = 0
    )

    qf_config = edict(
        n_blocks = args.n_hidden, 
        in_feature = args.policy_state_dim + latent_dim, # state dim 
        hidden_dim = args.hidden_dim, # 128
        out_dim = 1, # 10 * 2
        norm_cls =  nn.LayerNorm,
        act_cls = nn.Mish,
        block_cls = LinearBlock,
        bias = True,
        dropout = 0
    )

    # ------------- Modules ------------- #
    ## ------------- Q-functions ------------- ##
    qfs = [ MLPQF(Linear_Config(qf_config))  for _ in range(2)]

    ## ------------- High Policy ------------- ##

    # policy = HighPolicy(Linear_Config(policy_config), model.skill_prior)        
    policy = HighPolicy_SC(Linear_Config(policy_config), model.skill_prior, args.prior_state_dim)        



    ## ------------- Prior Policy ------------- ##
    prior_policy = deepcopy(model.skill_prior.prior_policy)
    prior_policy.requires_grad_(False) 

    ## ------------- Low Decoder ------------- ##
    low_actor = deepcopy(model.skill_decoder.eval())

    # ------------- Buffers & Collectors ------------- #
    buffer = Buffer_H(args.policy_state_dim, latent_dim, buffer_size, tanh = model.tanh)
    



    collector = LowFixedHierarchicalTimeLimitCollector(env, env_name, low_actor, horizon=10, time_limit=args.time_limit, tanh = model.tanh)

    
    # ------------- Goal setting ------------- #

    
    print(model.tanh)


    # ------------- RL agent ------------- #
    sac_config = {
        'policy' : policy,
        'prior_policy' : prior_policy, 
        # 'policy_optim' : model.subgoal_optimizer,
        # 'consistency_optim' : model.optimizer,
        'q_warmup_steps' : args.q_warmup, 
        'buffer' : buffer, 
        'qfs' : qfs,
        'discount' : 0.99,
        'tau' : 0.0005,
        'policy_lr' : args.policy_lr,
        'qf_lr' : 3e-4,
        'alpha_lr' : 3e-4,
        'prior_policy_lr' : 1e-5,
        'auto_alpha': args.auto_alpha,
        # 'target_kl': 2.5, # prior에서 벗어난 행동의 허용치 이면서 동시에 목표치
        'target_kl_start' : args.target_kl_start,
        'target_kl_end' : args.target_kl_end,
        'kl_clip': 10, # 
        'increasing_alpha': args.only_increase,
        'init_alpha' : args.init_alpha,
        'goal_conditioned' : True,
        'tune_idp' : False,
        'warmup' : 0,
        'kl_decay' : 0.99,
        'tanh' : model.tanh,
        'action_dim' : model.action_dim,
        
    }

    sac_config['kl_clip'] = args.target_kl_start * 1.5
        
    # ------------- Logger ------------- #

    self = SAC(sac_config)
    self = self.cuda()


    config = {'batch_size': 256, 'reuse_rate': args.reuse_rate, "project_name" : args.wandb_project_name, "precollect" : args.precollect}

    if args.env_name != "maze":
        task_name = "-".join([ t[0].upper() for t in tasks])
    else:
        # task_name = " to ".join([ f"[{t[0]},{t[1]}]" for t in tasks])
        task_name = task_obj.__repr__()



    # env 제한 
    state_processor = StateProcessor(env_name= args.env_name)
    # ------------- Train RL ------------- #
    with env.set_task(task_obj):
        state = env.reset()
        print("TASK : ",  state_processor.state_goal_checker(state, mode = "goal") )
                # log에 success rate추가 .
        # ep = 0
        ewm_rwds = 0
        early_stop = 0
        for episode_i in range(n_episode+1):


            log, updated = train_policy_iter(collector, self, episode_i, **config)
            log['episode_i'] = episode_i
            # log['task_name'] = task_name
            log[f'{task_name}_return'] = log['tr_return']
            del log['tr_return']

            if (episode_i + 1) % args.render_period == 0:
                if env_name == "maze":
                    log[f'{task_name}_policy_vis'] = draw_maze(plt.gca(), env, list(self.buffer.episodes)[-args.render_period:])
                else:
                    imgs = render_task(env, env_name, self.policy, low_actor, tanh = model.tanh)
                    imgs = np.array(imgs).transpose(0, 3, 1, 2)
                    log[f'{task_name}_rollout'] = wandb.Video(np.array(imgs), fps=32)


                # imgs = render_task(env, env_name, self.policy, low_actor, tanh = model.tanh)
                # imgs = np.array(imgs).transpose(0, 3, 1, 2)
                # log[f'{task_name}_rollout'] = wandb.Video(np.array(imgs), fps=32)

            wandb.log(log)
            plt.cla()

            ewm_rwds = 0.8 * ewm_rwds + 0.2 * log[f'{task_name}_return']

            if ewm_rwds > args.early_stop_threshold:
                early_stop += 1
            else: # 연속으로 넘겨야 함. 
                early_stop = 0
            
            if early_stop == 10:
                print("Converged enough. Early Stop!")
                break
    
    # weights_path = "/home/magenta1223/skill-based/SiMPL/proposed/weights/sac"
    # task_name = "-".join(tasks)
    # torch.save(self, f"{weights_path}/{task_name}.bin")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "kitchen", type = str)
    parser.add_argument("--wandb", action = "store_true")    
    parser.add_argument("-rp", "--render_period", default = 10, type = int)
    parser.add_argument("-ne", "--n_episode", default = 300, type = int)
    parser.add_argument("-p", "--path", default = "")
    parser.add_argument("--wandb_project_name", default = "GCPolicy_Level")    



    # parser.add_argument("-tl", "--time_limit", default = 280, type = int)
    # parser.add_argument("-nh", "--n_hidden", default = 5, type = int)
    # parser.add_argument("-hd", "--hidden_dim", default = 128, type =int)
    # parser.add_argument("-kls", "--target_kl_start", default = 20, type =float)
    # parser.add_argument("-kle", "--target_kl_end", default = 5, type =float)
    # parser.add_argument("-a", "--init_alpha", default = 0.1, type =float)
    # parser.add_argument("--only_increase", action = "store_true", default = False)    
    # parser.add_argument("--auto_alpha", action = "store_true", default = False)     

    # parser.add_argument("--reuse_rate", default = 256, type = int)
    # parser.add_argument("-plr", "--policy_lr", default = 3e-4, type =float)


    # parser.add_argument("-qwu", "--q_warmup", default = 5000, type =int)
    # # parser.add_argument("-qwu", "--q_warmup", default = 0, type =int)
    # parser.add_argument("-qwe", "--q_weight", default = 1, type =int)
    # parser.add_argument("-pc", "--precollect", default = 10, type = int)


    args = parser.parse_args()


    print(ENV_CONFIGS[args.env_name])
    
    env_config = ENV_CONFIGS[args.env_name]("sc")

    env_default_conf = {**env_config.attrs}

    for k, v in env_default_conf.items():
        setattr(args, k, v)



    print(args)

    env_cls = ENV_TASK[args.env_name]['env_cls']
    task_cls = ENV_TASK[args.env_name]['task_cls']
    ALL_TASKS = ENV_TASK[args.env_name]['tasks']
    configure = ENV_TASK[args.env_name]['cfg']


    if configure is not None:
        env = env_cls(**configure)
    else:
        env = env_cls()

    run_name = f"p:{args.path}_plr:{args.policy_lr}_a:{args.init_alpha}_qw:{args.q_warmup}{args.q_weight}"

    # ------------- Logger ------------- #
    wandb.init(
        project = args.wandb_project_name + args.env_name,
        
        name = run_name,

        # name = f"LEVEL {str(args.level)}", 
        config = {
            "alpha" : args.init_alpha,
            'policy_lr' : args.policy_lr,
            # 'prior_policy_lr' : sac_config['prior_policy_lr'],
            'target_kl_end' : args.target_kl_end,
            'warmup' : 0, #sac_config['warmup'],
            "pretrained_model" : f"{args.path}",
            'q_warmup_steps' : args.q_warmup, 
            'precollect' : args.precollect, 


        },
    )


    for tasks in ALL_TASKS:
        train_single_task(env, args.env_name, tasks, task_cls, args)
    wandb.finish()    


if __name__ == "__main__":
    main()