import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import os

from simpl.alg.simpl import ContextPriorResidualNormalMLPPolicy, LowFixedGPUWorker, Simpl
from simpl.alg.pearl import SetTransformerEncoder
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector, ConcurrentCollector
from simpl.nn import itemize
from simpl.rl import MLPQF

from tqdm import tqdm

import torch



import gym
import simpl.env.kitchen
from simpl.env.kitchen import KitchenTasks

from reproduce.kitchen.kitchen_vis import draw_kitchen

from datetime import datetime

# torch.multiprocessing.set_start_method('spawn')

# env = gym.make('simpl-kitchen-v0')
# train_tasks = KitchenTasks.train_tasks
# config = dict(
#     policy=dict(hidden_dim=128, n_hidden=5),
#     qf=dict(hidden_dim=128, n_hidden=5),
#     n_qf=2,
#     encoder=dict(hidden_dim=128, n_hidden=2, init_scale=1, prior_scale=1),
#     simpl=dict(init_enc_prior_reg=1e-3, target_enc_prior_kl=2,
#                init_enc_post_reg=1e-4, target_enc_post_kl=10,
#                init_policy_prior_reg=0.05, target_policy_prior_kl=0.1,
#                init_policy_post_reg=0.03, target_policy_post_kl=4, kl_clip=6),
#     enc_buffer_size=3000,
#     buffer_size=3000,
#     e_dim = 6,
#     time_limit=280,
#     n_epoch=500,
#     train=dict(batch_size=256, reuse_rate=256,
#                n_prior_batch=3, n_post_batch=27,
#                prior_enc_size=2, post_enc_size=1024)
# )
# visualize_env = draw_kitchen


# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# horizon = 10 # load['horizon']
# high_action_dim = 10 # load['z_dim']
# worker_gpus = 0

# encoder = SetTransformerEncoder(state_dim, high_action_dim, config['e_dim'], **config['encoder'])

# # load = torch.load("/home/magenta1223/skill-based/SiMPL/proposed/weights/SkillPrior_092823_020epoch.bin")
# # spirl_low_policy = load['model'].skill_decoder.eval().requires_grad_(False)
# # spirl_prior_policy = load['model'].skill_prior.eval().requires_grad_(False)

# load = torch.load("/home/magenta1223/skill-based/SiMPL/spirl_pretrained_kitchen.pt")
# spirl_low_policy = load['spirl_low_policy'].eval().requires_grad_(False) # skill decoder
# spirl_prior_policy = load['spirl_prior_policy'].eval().requires_grad_(False) # skill prior



# # collector
# spirl_low_policy.explore = False
# collector = LowFixedHierarchicalTimeLimitCollector(
#     env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit']
# )
# conc_collector = ConcurrentCollector([
#     LowFixedGPUWorker(collector, gpu)
#     for gpu in [worker_gpus]
# ])

# print("Prepare networks.. \n")
# # ready networks
# # task-encoder 
# # high-level policy w/ skill-prior & skill 
# high_policy = ContextPriorResidualNormalMLPPolicy(
#     spirl_prior_policy, state_dim, high_action_dim, config['e_dim'],
#     **config['policy']
# )

# # task에 대한 분포임
# with env.set_task(train_tasks[0]):
#     state = env.reset()
# z = encoder.encode([],sample = True)

# with high_policy.expl(), high_policy.condition(z):
#     high_policy.act(state)  





os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # debug
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1" # d4rl ignore warning

def simpl_warm_up_buffer(
    conc_collector, policy, train_tasks, enc_buffers, buffers,
    post_enc_size, batch_size
):
    
    task_indices = range(len(train_tasks))
    while len(task_indices) > 0:
        for task_idx in task_indices:
            z = encoder.encode([], sample=True)
            with policy.expl(), policy.condition(z):
                conc_collector.submit(train_tasks[task_idx], policy)

        episodes = conc_collector.wait()
        for episode, task_idx in zip(episodes, task_indices):
            enc_buffers[task_idx].enqueue(episode.as_high_episode())

        task_indices = [
            task_idx for task_idx, enc_buffer in enumerate(enc_buffers)
            if enc_buffer.size < post_enc_size
        ]
    task_indices = range(len(train_tasks))
    while len(task_indices) > 0:
        for task_idx in task_indices:
            z = encoder.encode([enc_buffers[task_idx].sample(post_enc_size)], sample=True)
            with policy.expl(), policy.condition(z):
                conc_collector.submit(train_tasks[task_idx], policy)

        episodes = conc_collector.wait()

        for episode, task_idx in zip(episodes, task_indices):
            buffers[task_idx].enqueue(episode.as_high_episode())

        task_indices = [
            task_idx for task_idx, buffer in enumerate(buffers)
            if buffer.size < batch_size
        ]


def simpl_meta_train_iter(
    conc_collector, trainer, train_tasks, *,
    batch_size, reuse_rate,
    n_prior_batch, n_post_batch, prior_enc_size, post_enc_size
):
    """
    Task Encoder를 
    """ 

    log = {}

    # collect
    device = trainer.device
    trainer.policy.to('cpu')


    # task embedding을 얻는 과정
    # - from task prior 이게 task embedding을 학습하기 위한 데이터를 얻는 과정;
    for task in train_tasks:
        # predefined distribution에서 z를 샘플링
        z = encoder.encode([], sample=True) 
        # trainer: high-level policy network
        # trainer.policy.condition: 입력받은 z를 policy의 z로 변경해주는 context manager
        # trainer.policy.expl(): exploration mode로 설정하는 context manager
        with trainer.policy.expl(), trainer.policy.condition(z):
            conc_collector.submit(task, trainer.policy) # task와 policy를 통해 수집. 
    prior_episodes = conc_collector.wait() # 뭔지 몰라도 수집한 episode를 뱉는다. 
    prior_high_episodes = [episode.as_high_episode() for episode in prior_episodes] # hierarchical episode에 구현됨. MDP에서 low actions가 아닌, skill embedding이 사용된 high episode를 사용
    for high_episode, enc_buffer, buffer in zip(prior_high_episodes, trainer.enc_buffers, trainer.buffers):
        enc_buffer.enqueue(high_episode) # 뭐.. 그냥 큐에 넣는다고 생각
        buffer.enqueue(high_episode) # 

    # - from task posterior 여기는 skil
    for task, enc_buffer in zip(train_tasks, enc_buffers):
        # encoder buffer에서 정해진 크기의 샘플을 얻음 
        z = encoder.encode([enc_buffer.sample(post_enc_size)], sample=True)
        # 역시 environment와 통신
        # 여길 바꿔야 됨.
        # 그래도 돼?
        with trainer.policy.expl(), trainer.policy.condition(z):
            conc_collector.submit(task, trainer.policy)
    post_episodes = conc_collector.wait()
    post_high_episodes = [episode.as_high_episode() for episode in post_episodes]
    # 사후 에피소드를 얻고
    for high_episode, buffer in zip(post_high_episodes, trainer.buffers):
        buffer.enqueue(high_episode)
    # 사후 보상들을 리스트로 
    tr_returns = [sum(episode.rewards) for episode in post_episodes]
    log.update({
        'avg_tr_return': np.mean(tr_returns),
        'tr_return': {task_idx: tr_return for task_idx, tr_return in enumerate(tr_returns)}
    })

    trainer.policy.to(device)

    # meta train
    n_prior = sum([len(episode) for episode in prior_high_episodes])
    n_post = sum([len(episode) for episode in post_high_episodes])
    n_step = reuse_rate * (n_prior + n_post) / (n_prior_batch + n_post_batch) / batch_size
    for _ in range(max(int(n_step), 1)):
        stat = trainer.step(n_prior_batch, n_post_batch, batch_size, prior_enc_size, post_enc_size)
    log.update(itemize(stat))

    return log




if __name__ == '__main__':
    print(datetime.now().strftime("%m-%d %H%M"))
    torch.multiprocessing.set_start_method('spawn')

    import_pathes = {
        'maze_40t': 'maze.simpl_meta_train_40t',
        'maze_20t': 'maze.simpl_meta_train_20t',
        'kitchen': 'kitchen.simpl_meta_train',
        'kitchen_ot': 'kitchen.simpl_meta_train_ot',
    }

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-w', '--worker-gpus', required=True, type=int, nargs='+')
    parser.add_argument('-s', '--spirl-pretrained-path', required=True)

    parser.add_argument('-t', '--policy-vis_period', type=int)
    parser.add_argument('-p', '--wandb-project-name')
    parser.add_argument('-r', '--wandb-run-name')
    parser.add_argument('-a', '--save_file_path')
    parser.add_argument('--use_posterior', default= False, type = bool)

    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, train_tasks, config, visualize_env = module.env, module.train_tasks, module.config, module.visualize_env

    gpu = args.gpu
    worker_gpus = args.worker_gpus
    spirl_pretrained_path = args.spirl_pretrained_path 
    policy_vis_period = args.policy_vis_period or 10
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.simpl_meta_train.' + wandb.util.generate_id()
    save_filepath = args.save_file_path or f'./{wandb_run_name}.pt'
    use_posterior = args.use_posterior
    

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(spirl_pretrained_path)

    print("Loading Skill Priors")
    # load pre-trained SPiRL
    load = torch.load(spirl_pretrained_path, map_location='cpu')
    horizon = 10 # load['horizon']
    high_action_dim = 10 # load['z_dim']
    
    # spirl
    #spirl_low_policy = load['spirl_low_policy'].eval().requires_grad_(False) # skill decoder
    #spirl_prior_policy = load['spirl_prior_policy'].eval().requires_grad_(False) # skill prior
    spirl_low_policy = load['model'].skill_decoder.eval().requires_grad_(False).cpu()
    if use_posterior:
        spirl_prior_policy = load['model'].skill_encoder.eval().requires_grad_(False).cpu()

    else:
        spirl_prior_policy = load['model'].skill_prior.eval().requires_grad_(False).cpu()


    print("Collecting.. \n")
    # collector
    spirl_low_policy.explore = False
    collector = LowFixedHierarchicalTimeLimitCollector(
        env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit'],
        use_posterior = use_posterior
    )
    conc_collector = ConcurrentCollector([
        LowFixedGPUWorker(collector, gpu)
        for gpu in worker_gpus
    ])

    print("Prepare networks.. \n")
    # ready networks
    # task-encoder 
    encoder = SetTransformerEncoder(state_dim, high_action_dim, config['e_dim'], **config['encoder'])
    # high-level policy w/ skill-prior & skill 
    high_policy = ContextPriorResidualNormalMLPPolicy(
        spirl_prior_policy, state_dim, high_action_dim, config['e_dim'],
        **config['policy']
    )
    qfs = [MLPQF(state_dim+config['e_dim'], high_action_dim, **config['qf']) for _ in range(config['n_qf'])]
    print("Networks ready!\n")      

    print("Prepare buffers..\n")  
    # ready buffers
    enc_buffers = [
        Buffer(state_dim, high_action_dim, config['enc_buffer_size'])
        for _ in range(len(train_tasks))
    ]
    buffers = [
        Buffer(state_dim, high_action_dim, config['buffer_size'])
        for _ in range(len(train_tasks))
    ]
    print("Buffers ready! \n")
    print("Warmup Buffer ... \n")
    simpl_warm_up_buffer(
        conc_collector, high_policy, train_tasks, enc_buffers, buffers,
        config['train']['post_enc_size'], config['train']['batch_size']
    )
    print("Prepare SiMPL..\n") 
    # meta train
    trainer = Simpl(high_policy, spirl_prior_policy, qfs, encoder, enc_buffers, buffers, **config['simpl']).to(gpu)

    wandb.init(
        project=wandb_project_name, name=wandb_run_name,
        config={**config, 'gpu': gpu, 'spirl_pretrained_path': args.spirl_pretrained_path}
    )

    print("Start Train")
    for epoch_i in tqdm(range(1, config['n_epoch']+1)):
        log = simpl_meta_train_iter(conc_collector, trainer, train_tasks, **config['train'])
        log['epoch_i'] = epoch_i
        if epoch_i % policy_vis_period == 0:
            plt.close('all')
            n_row = int(np.ceil(len(train_tasks)/10))
            fig, axes = plt.subplots(n_row, 10, figsize=(20, 2*n_row))
            for task_idx, (task, buffer) in enumerate(zip(train_tasks, buffers)):
                with env.set_task(task):
                    visualize_env(axes[task_idx//10][task_idx%10], env, list(buffer.episodes)[-20:])
            log['policy_vis'] = fig
        wandb.log(log)

    torch.save({
        'encoder': encoder, # task encoder
        'high_policy': high_policy, # policy network
        'qfs': qfs, # q functions. SAC구현대로 2개 사용
        'policy_post_reg': trainer.policy_post_reg().item() # ? 
    }, save_filepath)
    conc_collector.close()
