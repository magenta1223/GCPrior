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


def goal_checker_maze(state):
    return (state[:2] * 1).astype(int)



def get_goal_kitchen(state):
    return state[30:]

def get_goal_maze(state):
    return state[-2:]


def goal_transform_kitchen(state):
    state[:9] = 0
    return state[:30]


def goal_transform_maze(state):
    return state[:2]

def state_process_kitchen(state):
    return state[:30]

def state_process_maze(state):
    return state[:-2]


def get_goal_calvin():
    return 
def state_process_calvin():
    return 

def goal_transform_calvin():
    return 
def goal_checker_calvin():
    return 

# --------------------------- Distribution --------------------------- #

def get_dist(model_output, log_scale = None, scale = None,  detached = False, tanh = False):
    if detached:
        model_output = model_output.clone().detach()
        model_output.requires_grad = False

    if log_scale is None and scale is None:
        mu, log_scale = model_output.chunk(2, -1)
        scale = log_scale.clamp(-10, 2).exp()
    else:
        mu = model_output
        if log_scale is not None:
            scale = log_scale.clamp(-10, 2).exp()

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

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) 
    y = y.unsqueeze(0) 
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) 

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y) 
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
            "maze"    : get_goal_maze
        }

        self.__goal_checkers__ = {
            "kitchen" : goal_checker_kitchen,
            "maze"  : goal_checker_maze
        }

        self.__state2goals__ = {
            "kitchen" : goal_transform_kitchen,
            "maze"  : goal_transform_maze
        }
        self.__state_processors__ = {
            "kitchen" : state_process_kitchen,
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
                return self.__goal_checkers__[self.env_name](state) 
            else:
                return self.__goal_checkers__[self.env_name](state[-2:])

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
                    


