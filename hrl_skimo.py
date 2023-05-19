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


from LVD.envs import ENV_TASK
from LVD.configs.build import Linear_Config
from LVD.modules.base import *
from LVD.modules.policy import *
from LVD.modules.subnetworks import *
# from LVD.rl.sac_gcid import SAC
from LVD.rl.sac_skimo import SAC

from LVD.contrib.simpl.torch_utils import itemize
from LVD.rl.vis import *
from LVD.utils import *
from LVD.collector.skimo import LowFixedHierarchicalTimeLimitCollector
from LVD.collector.storage import Buffer_TT
from LVD.rl.rl_utils import *
from LVD.configs.env import ENV_CONFIGS
from simpl_reproduce.maze.maze_vis import draw_maze


seed_everything()


# reward prediction 
# CEM planning 

def render_task(env, env_name, policy, low_actor, tanh, qfs):
    imgs = []
    state = env.reset()


    processor = StateProcessor(env_name = env_name)


    G = processor.get_goals(state)
    state = processor.state_process(state)

    low_actor.eval()
    policy.eval()

    done = False
    time_step = 0
    time_limit = 280
    print("rendering..")
    count = 0

    while not done and time_step < time_limit: 
        if time_step % 10 == 0:
            if tanh:
                high_action = policy.act(state, G, qfs)
            else:
                high_action = policy.act(state, G)

        with low_actor.condition(high_action), low_actor.expl():
            low_action = low_actor.act(state)
        
        state, reward, done, info = env.step(low_action)
        state = processor.state_process(state)
        img = env.render(mode = "rgb_array")
        imgs.append(img)
        time_step += 1
    print("done!")
    return imgs, reward





def train_policy_iter(collector, trainer, episode_i, batch_size, reuse_rate, project_name, precollect):
    # ------------- Initialize log ------------- #
    log = {}

    # ------------- Collect Data ------------- #
    with trainer.policy.expl(), collector.low_actor.expl() : #, collector.env.step_render():
        episode, G = collector.collect_episode(trainer.policy, trainer.qfs)

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

    task_obj = task_cls(tasks)


    ## ------------- Spirl Modules ------------- ##
    # load = torch.load(f"/home/magenta1223/skill-based/SiMPL/{args.path}")
    load = torch.load(args.path)

    model = load['model'].eval()

    # ------------- Hyper Parameteres ------------- #
    buffer_size = 20000
    n_episode = args.n_episode
    # state_dim = env.observation_space.shape[0]
    if env_name == "kitchen":
        state_dim = 30
    else:
        # state_dim =  env.observation_space.shape[0]
        state_dim = 4

    latent_dim = 10
    
    learned_state_dim = model.prior_policy.state_encoder.out_dim

    # ------------- Module Configuration ------------- #
    policy_config = edict(
        n_blocks = args.n_hidden, 
        in_feature = learned_state_dim, # state dim 
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
        in_feature = learned_state_dim + latent_dim, # state dim 
        hidden_dim = args.hidden_dim, # 128
        out_dim = 1, # 10 * 2
        norm_cls = nn.LayerNorm,
        act_cls = nn.Mish,
        block_cls = LinearBlock,
        bias = True,
        dropout = 0
    )

    reward_config = edict(
        n_blocks = args.n_hidden, 
        in_feature = learned_state_dim + latent_dim, # state dim 
        hidden_dim = args.hidden_dim, # 128
        out_dim = 1, # 10 * 2
        norm_cls = nn.LayerNorm,
        act_cls = nn.Mish,
        block_cls = LinearBlock,
        bias = True,
        dropout = 0
    )

    ## ------------- Prior Policy ------------- ##

    prior_policy = deepcopy(model.prior_policy)
    prior_policy.requires_grad_(False) 





    # ------------- Modules ------------- #
    ## ------------- Q-functions ------------- ##
    qfs = [ MLPQF(Linear_Config(qf_config))  for _ in range(2)]
    reward_function = SequentialBuilder(Linear_Config(reward_config))

    
    planning_horizons = {
        "maze" : 10,
        "kitchen" : 3
    }

    planning_horizon = planning_horizons[args.env_name]


    ## ------------- High Policy ------------- ##
    # Skimo config
    skimo_config = dict(
        reward_function = reward_function,
        prior_policy = prior_policy,
        min_scale = 0.001,
        prior_state_dim = state_dim,
        planning_horizon = planning_horizon, # TODO env별로 다름
        skill_dim = latent_dim,
        num_elites = 64,
        cem_momentum = 0.1,
        cem_temperature = 0.5,
        num_policy_traj = 512,
        num_sample_traj = 25,
        cem_iter = 6,
        rl_discount = 0.99,
        step_interval = 25000,
        tanh = model.tanh,
        _step = 0,
        warmup_steps = 5000
    )

    policy = HighPolicy_Skimo(Linear_Config(policy_config), skimo_config)        


    ## ------------- Low Decoder ------------- ##
    low_actor = deepcopy(model.skill_decoder.eval())

    # ------------- Buffers & Collectors ------------- #
    buffer = Buffer_TT(state_dim, latent_dim, planning_horizon, buffer_size)
    collector = LowFixedHierarchicalTimeLimitCollector(env, env_name, low_actor, horizon=10, time_limit=args.time_limit, tanh = model.tanh)

    
    # ------------- Goal setting ------------- #

    
    print(model.tanh)


    # ------------- RL agent ------------- #
    sac_config = {
        'policy' : policy,
        'prior_policy' : prior_policy, 
        # 'policy_optim' : model.subgoal_optimizer,
        # 'consistency_optim' : model.optimizer,
        'finetune' : args.finetune, 
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
        'gcprior' : args.gcprior,
        "use_hidden" : args.use_hidden
        
    }

    sac_config['kl_clip'] = args.target_kl_start * 1.5
        
    # ------------- Logger ------------- #
    self = SAC(sac_config)
    self = self.cuda()



    # config = {'batch_size': 256, 'reuse_rate': 256, "G" : G, "project_name" : args.wandb_project_name}
    config = {'batch_size': 256, 'reuse_rate': args.reuse_rate, "project_name" : args.wandb_project_name, "precollect" : args.precollect}

    task_name = str(task_obj)


    # env 제한 

    weights_path = f"./weights/{args.env_name}/skimo/sac"
    os.makedirs(weights_path, exist_ok= True)


    torch.save({
        "model" : self,
        "collector" : collector,
        "task" : task_obj,
        "env" : env,
    }, f"{weights_path}/{task_name}.bin")

    
    state_processor = StateProcessor(env_name= args.env_name)


    # ------------- Train RL ------------- #
    with env.set_task(task_obj):
        state = env.reset()
        print("TASK : ",  state_processor.state_goal_checker(state, env, mode = "goal") )
        # log에 success rate추가 .
        # ep = 0
        for episode_i in range(n_episode+1):


            log, updated = train_policy_iter(collector, self, episode_i, **config)
            log['episode_i'] = episode_i
            # log['task_name'] = task_name
            log[f'return'] = log['tr_return']
            del log['tr_return']

            if (episode_i + 1) % args.render_period == 0:
                # imgs, reward = render_task(env, env_name, self.policy, low_actor, tanh = model.tanh, qfs= self.qfs)
                # imgs = np.array(imgs).transpose(0, 3, 1, 2)
                if env_name == "maze":
                    log[f'policy_vis'] = draw_maze(plt.gca(), env, list(self.buffer.episodes)[-20:])
                else:
                    imgs, reward = render_task(env, env_name, self.policy, low_actor, tanh = model.tanh)
                    imgs = np.array(imgs).transpose(0, 3, 1, 2)
                    if args.env_name == "maze":
                        fps = 100
                    else:
                        fps = 50
                    log[f'rollout'] = wandb.Video(np.array(imgs), fps=fps, caption= str(reward))

                # log[f'rollout'] = wandb.Video(np.array(imgs), fps=32, caption= str(reward))




            new_log = {}
            for k, v in log.items():
                new_log[f"{task_name}/{k}"] = v


            wandb.log(new_log)
            plt.cla()


    torch.save({
        "model" : self,
        "collector" : collector,
        "task" : task_obj,
        "env" : env,
    }, f"{weights_path}/{task_name}.bin")



    
    # weights_path = "/home/magenta1223/skill-based/SiMPL/proposed/weights/sac"
    # task_name = "-".join(tasks)
    # torch.save(self, f"{weights_path}/{task_name}.bin")
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default = "kitchen", type = str)
    parser.add_argument("--wandb", action = "store_true")    
    parser.add_argument("--norm", action = "store_true", default= False)    
    parser.add_argument("-p", "--path", default = "")
    parser.add_argument("--wandb_project_name", default = "GCPolicy_Level")    
    parser.add_argument("-rp", "--render_period", default = 10, type = int)
    parser.add_argument("-qwu", "--q_warmup", default = 5000, type =int)
    parser.add_argument("-qwe", "--q_weight", default = 1, type =int)
    parser.add_argument("-pc", "--precollect", default = 10, type = int)


    args = parser.parse_args()

    print(args)
    env_config = ENV_CONFIGS[args.env_name]("gc_div_joint")
    env_default_conf = {**env_config.attrs}

    for k, v in env_default_conf.items():
        setattr(args, k, v)




    env_cls = ENV_TASK[args.env_name]['env_cls']
    task_cls = ENV_TASK[args.env_name]['task_cls']
    ALL_TASKS = ENV_TASK[args.env_name]['tasks']
    configure = ENV_TASK[args.env_name]['cfg']

    if hasattr(args, "relative") and configure is not None:
        configure['relative'] = args.relative


    if configure is not None:
        env = env_cls(**configure)
    else:
        env = env_cls()
    print(env)
    
    if args.env_name == "kitchen":
        args.init_alpha = 0.05

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