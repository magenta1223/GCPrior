import os
import random
import time
from datetime import datetime

import torch
from torch.nn import functional as F
import torch.distributions as torch_dist
from torch.distributions.kl import register_kl

import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES, BONUS_THRESH



from .contrib.dists import TanhNormal

import cv2
from .envs import ENV_TASK


# --------------------------- Seed --------------------------- #


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    # torch.use_deterministic_algorithms(True)


# --------------------------- Env-Model IO --------------------------- #

def prep_state(states, device):
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype = torch.float32)

    if len(states.shape) < 2:
        states = states.unsqueeze(0)

    states = states.to(device)
    return states
    


def goal_checker_kitchen(state):
    achieved_goal = []
    for obj, indices in OBS_ELEMENT_INDICES.items():
        g = OBS_ELEMENT_GOALS[obj]
        distance = np.linalg.norm(state[indices] -g)   
        if distance < BONUS_THRESH:
            achieved_goal.append(obj)
    task = ""

    for sub_task in ['microwave', 'kettle', 'bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet']:
        if sub_task in achieved_goal:
            task += sub_task[0].upper() 
    return task

def goal_checker_calvin(goal_state):
    achieved_goal = []
    for obj, indices in CALVIN_EL_INDICES.items():
        g = CALVIN_EL_GOALS[obj]
        distance = np.linalg.norm(goal_state[indices] -g)   
        if distance < CALVIN_BONUS_THRESH:
            achieved_goal.append(obj)

    task = ""
    
    for subtask in ['open_drawer', 'turn_on_lightbulb', 'move_slider_left', 'turn_on_led']:
        if subtask in achieved_goal:
            if subtask == "open_drawer":
                task += "OpenD_"
            elif subtask == "turn_on_lightbulb":
                task += "TurnB_"
            elif subtask == "move_slider_left":
                task += "MoveS_"
            else:
                task += "TurnL_"

    return task[:-1]


def goal_checker_maze(state, env):
    # return state[4:]
    # complete_threshold = 1.0
    # goal_dist = np.linalg.norm(state[:2] - state[-2:])
    # completed = (goal_dist <= complete_threshold)

    # return "Success" if completed else "Fail"
    # return (state[:2] * env.size).astype(np.uint8)
    return (state[:2] * 1).astype(int)


    # if ((state[:2] - state[2:]) ** 2).sum() < 0.1:
    #     return "Done"
    # else:
    #     return "NO"


def get_goal_kitchen(state):
    # state[:9] = 0
    # return state[30:]
    return state[30:]

def get_goal_calvin(state):
    # state[:9] = 0
    # return state[30:]
    return state[39:]

def get_goal_maze(state):
    # state[:9] = 0
    # return state[30:]
    # return state[4:]
    # return state[32 + 40:]
    return np.concatenate((state[-2:], [0,0]), axis = 0)


def goal_transform_kitchen(state):
    state[:9] = 0
    return state[:30]
    # return state[9:30]


def goal_transform_calvin(state):
    # return state[21:]
    return state[15:21]

def goal_transform_maze(state):
    # return state[21:]
    # return state[2:]
    return state[32:34]



def state_process_kitchen(state):
    return state[:30]

def state_process_calvin(state):
    # return state[:21]
    # return state[:39]

    return state[:21]

def state_process_maze(state):
    # return state[:21]
    # return state[:39]
    # return state[32:36]
    return state[:-2]



# --------------------------- Distribution --------------------------- #

def get_dist(model_output, log_scale = None,  detached = False, tanh = False):
    if detached:
        model_output = model_output.clone().detach()
        model_output.requires_grad = False

    if log_scale is None:
        mu, log_scale = model_output.chunk(2, -1)
    else:
        mu = model_output

    log_scale = log_scale.clamp(-10, 2)
    scale = log_scale.exp()

    if tanh:
        return TanhNormal(mu, scale)
    else:
        dist = torch_dist.Normal(mu, scale)
        return torch_dist.Independent(dist, 1)

def get_fixed_dist(model_output,  tanh = False):
    model_output = model_output.clone().detach()
    mu, log_scale = torch.zeros_like(model_output).chunk(2, -1)
    scale = log_scale.exp()
    if tanh:
        return TanhNormal(mu, scale)
    else:
        dist = torch_dist.Normal(mu, scale)
        return torch_dist.Independent(dist, 1)  

def nll_dist(z, q_hat_dist, pre_tanh_value = None, tanh = False):
    if tanh:
        return - q_hat_dist.log_prob(z, pre_tanh_value)
    else:
        return - q_hat_dist.log_prob(z)

def kl_divergence(dist1, dist2, *args, **kwargs):
    return torch_dist.kl_divergence(dist1, dist2)

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch

def kl_branching_point(p, q, state_labels):
    def ignore_non_branching(dist, indices):
        if isinstance(dist, torch_dist.Independent):
            loc = dist.base_dist.loc.clone()
            scale = dist.base_dist.scale.clone()
            new_loc = loc[indices]
            new_scale = scale[indices]
            return torch_dist.Independent(torch_dist.Normal(new_loc, new_scale), 1)
        else:
            loc = dist._normal.base_dist.loc.clone()
            scale = dist._normal.base_dist.scale.clone()
            new_loc = loc[indices]
            new_scale = scale[indices]
            return TanhNormal(new_loc, new_scale)

    # N, T, 1임
    branching_indices = state_labels[:, 0] == 0
    p, q = ignore_non_branching(p, branching_indices), ignore_non_branching(q, branching_indices)
    return torch_dist.kl_divergence(p, q)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size) radius 

def compute_mmd(x, y):
    """
    하나는 loc, 하나는 sample 넣어야 됨. 
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y) # y, y는 의미가 없지 않나.. ? 어차피 gradient 없는디
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


# --------------------- Helper Class --------------------- # 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class StateProcessor:

    def __init__(self, env_name):
        self.env_name = env_name

        self.__get_goals__ = {
            "kitchen" : get_goal_kitchen,
            "calvin"  : get_goal_calvin,
            "maze"    : get_goal_maze
        }

        self.__goal_checkers__ = {
            "kitchen" : goal_checker_kitchen,
            "calvin"  : goal_checker_calvin,
            "maze"  : goal_checker_maze
        }

        self.__state2goals__ = {
            "kitchen" : goal_transform_kitchen,
            "calvin"  : goal_transform_calvin,
            "maze"  : goal_transform_maze
        }
        self.__state_processors__ = {
            "kitchen" : state_process_kitchen,
            "calvin"  : state_process_calvin,
            "maze"  : state_process_maze         
        }

    def get_goals(self, state):
        return self.__get_goals__[self.env_name](state)

    def goal_checker(self, goal):
        return self.__goal_checkers__[self.env_name] (goal)
    
    def state2goal(self, state):
        return self.__state2goals__[self.env_name](state)
    
    def state_process(self, state):
        return self.__state_processors__[self.env_name](state)
    
    def state_goal_checker(self, state, env, mode = "state"):
        """
        Check the state satisfiy which goal state
        """
        if self.env_name == "maze":
            if mode =="state":
                return self.__goal_checkers__[self.env_name](state, env) 
            else:
                return self.__goal_checkers__[self.env_name](state[-2:], env)

        if mode =="state":
            return self.__goal_checkers__[self.env_name](self.__state2goals__[self.env_name](state)) 
        else:
            return self.__goal_checkers__[self.env_name](self.__get_goals__[self.env_name](state)) 


class Scheduler_Helper(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, module_name = ""):

        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, cooldown, min_lr, eps, verbose)
        self.module_name = module_name

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of {}`s group {} to {:.4e}.'.format(epoch_str, self.module_name, i, new_lr))
                    
