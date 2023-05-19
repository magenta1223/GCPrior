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



# --------------------------- Env, Model Utils --------------------------- #

def prep_state(states, device):
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype = torch.float32)

    if len(states.shape) < 2:
        states = states.unsqueeze(0)

    states = states.to(device)
    return states
    

## ------- goal checker ------- ## 

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

def goal_checker_carla(state):
    return (state[-3:]).astype(int)

def goal_checker_calvin():
    return 

## ------- get goal state from state ------- ## 


def get_goal_kitchen(state):
    return state[30:]

def get_goal_maze(state):
    return state[-2:]

def get_goal_calvin():
    return 


def get_goal_carla(state):
    return state[-3 :-1] # only x, y

## ------- s ------- ## 


def goal_transform_kitchen(state):
    state[:9] = 0
    return state[:30]


def goal_transform_maze(state):
    return state[:2]

def goal_transform_calvin():
    return 

def goal_transform_carla(state):
    return state[12 :14]

## ------- env state -> obs ------- ## 

# SENSOR_SCALE = {
#     "control": (1, slice(0,3)),
#     "acceleration": (1, slice(0,3)),
#     "velocity": (10, slice(0,3)),
#     "angular_velocity": (10, slice(0,3)),
#     "location": (100, slice(0,2)),
#     "rotation": (10, slice(0,3)), # only steer 
#     "forward_vector": (1, slice(0,3)),  # remove
#     "target_location": (100, slice(0,0)), # remove 
# }


SENSOR_SCALE = {
    "control": (1, slice(0,3)),
    "acceleration": (1, slice(0,3)),
    "velocity": (1, slice(0,3)),
    "angular_velocity": (1, slice(0,3)),
    "location": (1/10, slice(0,2)),
    "rotation": (1/180, slice(0,3)), # only steer 
    "forward_vector": (1, slice(0,0)),  
    "target_location": (1, slice(0,0)), # remove 
}


SENSORS = ["control", "acceleration", "velocity", "angular_velocity", "location", "rotation", "forward_vector", "target_location"]





def state_process_kitchen(state):
    return state[:30]

def state_process_maze(state):
    return state[:-2]


def state_process_calvin():
    return 

def state_process_carla(state, normalize = False):
    if len(state.shape) == 2:
        obs_dict = { key : state[:, i*3 : (i+1)*3 ]   for i, key in enumerate(SENSORS)}
        prep_obs_dict = {}

        for k, (scale, idx) in SENSOR_SCALE.items():
            prep_obs_dict[k] = obs_dict[k][:, idx] * scale

            
            # contorl : all
            # acceleration : all
            # vel : all
            # angular vel : all
            # loc : all
            # rot : only y
            
        state = np.concatenate( [v for k, v in prep_obs_dict.items() if v.any()], axis = -1)
        return state
    else: 
        obs_dict = { key : state[i*3 : (i+1)*3 ]   for i, key in enumerate(SENSORS)}
        prep_obs_dict = {}

        for k, (scale, idx) in SENSOR_SCALE.items():
            prep_obs_dict[k] = obs_dict[k][idx]

            # raw_obs = obs_dict[k][idx] / scale
            # if raw_obs.
            # prep_obs_dict[k] = obs_dict[k][idx] / scale
            # print(k, prep_obs_dict[k])
            # contorl : all
            # acceleration : all
            # vel : all
            # angular vel : all
            # loc : all
            # rot : only y

        state = np.concatenate( [
            prep_obs_dict['control'],
            prep_obs_dict['acceleration'],
            prep_obs_dict['velocity'],
            prep_obs_dict['angular_velocity'],
            prep_obs_dict['location'],
            prep_obs_dict['rotation'],
            prep_obs_dict['forward_vector'],

        ], axis = -1)

        # state = np.concatenate( [v for k, v in prep_obs_dict.items() if v.any()], axis = -1)
        return state


    # xy = state[12:14] 
    # return np.concatenate((state[:12], xy, state[15:-6]), axis = -1)
    # return np.concatenate((state[:14], state[15:-6]), axis = -1) 
    # return state[:21]



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
            "maze"    : get_goal_maze,
            "carla" : get_goal_carla
        }

        self.__goal_checkers__ = {
            "kitchen" : goal_checker_kitchen,
            "maze"  : goal_checker_maze,
            "carla" : goal_checker_carla

        }

        self.__state2goals__ = {
            "kitchen" : goal_transform_kitchen,
            "maze"  : goal_transform_maze,
            "carla"  : goal_transform_carla,

        }
        self.__state_processors__ = {
            "kitchen" : state_process_kitchen,
            "maze"  : state_process_maze,
            "carla" : state_process_carla
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
                    


