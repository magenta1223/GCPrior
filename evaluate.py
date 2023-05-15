import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from LVD.utils import seed_everything
from LVD.envs import ENV_TASK

def formatter(performances):
    mean = str(np.round(np.mean(performances), 2))
    std = str(np.round(np.std(performances), 2))
    return f"{mean} +- {std}"

def process_task(weights_path, task_name):
    x = 31
    success_rates = []
    rewards = []

    load = torch.load(weights_path)
    agent = load['model'].eval()
    collector = load['collector']
    task_obj = load['task']
    env = load['env']

    for seed in range(5):
        seed_everything(666 + (2**seed))
        count = 0
        rwds = 0
        with env.set_task(task_obj), agent.policy.expl(), collector.low_actor.expl():
            for i in range(x):
                if "simpl" in weights_path:
                    episode = collector.collect_episode(agent.policy)
                else:
                    episode, G = collector.collect_episode(agent.policy)
                # rewards.append(sum(episode.rewards))
                rwds += sum(episode.rewards)

                if np.array(episode.dones).sum() != 0:  # success
                    count += 1

        rate = count * 100 / x
        success_rates.append(rate)
        rewards.append(rwds / x)
        # SUCCESS_RATE[task_name].append(rate)
        # print(SUCCESS_RATE[task_name])
    return success_rates, rewards, task_name

def main(args):
    task_cls = ENV_TASK[args.env_name]['task_cls']
    ALL_TASKS = [ str(task_cls(t))  for t in ENV_TASK[args.env_name]['tasks']]

    SUCCESS_RATE = {}
    RWDS = {}

    for i, task in enumerate(tqdm(ALL_TASKS, desc="Processing files")):
        if args.method == "simpl":
            weights_path = f"./weights/{args.env_name}/{args.method}/{task}.bin"
        else:
            weights_path = f"./weights/{args.env_name}/{args.method}/sac/{task}.bin"

        if not os.path.exists(weights_path):
            print(f"{task} does not exist")
            continue

        success_rates, rewards, task_name = process_task(weights_path, task)
        SUCCESS_RATE[task_name] = success_rates
        RWDS[task_name] = rewards

    data = {
        "env" : [  args.env_name  for _ in range(len(SUCCESS_RATE))],
        "method" : [  args.method  for _ in range(len(SUCCESS_RATE))],
        "task" : [  k  for k, v in SUCCESS_RATE.items()],
        "success_rate" : [  formatter(v)  for k, v in SUCCESS_RATE.items()],
        "rewards" : [  formatter(v)  for k, v in RWDS.items()],
        "weight_path" : [  f"./weights/{args.env_name}/{args.method}/sac/{k}.bin"  for k, v in SUCCESS_RATE.items()],
    }

    data = pd.DataFrame(data)

    if os.path.exists("./results.csv"):
        result = pd.read_csv("./results.csv")
        result = pd.concat((result, data), axis = 0)
        result = result.drop_duplicates(['env', 'method', 'task', 'weight_path'])


    result.to_csv("./results.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default= "none", choices= ['kitchen', 'maze', 'carla'])
    parser.add_argument("--method", default= "sc", choices= ['sc', 'simpl', 'skimo', 'gc_div_joint'])
    args = parser.parse_args()
    main(args)