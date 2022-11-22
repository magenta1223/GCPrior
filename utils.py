import torch
import torch.distributions as torch_dist
import math
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

import d4rl

from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES, BONUS_THRESH

def prep_state(states, device):
    if isinstance(states, np.ndarray):
        states = torch.tensor(states, dtype = torch.float32)

    if len(states.shape) < 2:
        states = states.unsqueeze(0)

    states = states.to(device)
    return states
    

def get_dist(model_output, log_scale = None,  detached = False):
    if detached:
        model_output = model_output.detach()

    if log_scale is None:
        mu, log_scale = model_output.chunk(2, -1)

    else:
        mu = model_output

    log_scale = log_scale.clamp(-10, 2)
    scale = log_scale.exp()
    dist = torch_dist.Normal(mu, scale)
    return torch_dist.Independent(dist, 1)

def get_fixed_dist(model_output):
    mu, log_scale = torch.zeros_like(model_output).chunk(2, -1)
    scale = log_scale.exp()
    dist = torch_dist.Normal(mu, scale)
    return torch_dist.Independent(dist, 1)

def nll_dist(q_dist, q_hat_dist):
    return - q_hat_dist.log_prob(q_dist.sample()).mean()

def log_prob(mu, sigma, log_sigma, val):
    return -1 * ((val - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi))

def nll_loss2(reconstructed, actions):
    mu_p, log_sigma_p, sigma_p = reconstructed, 0, 1

    return - log_prob(mu_p, sigma_p, log_sigma_p, actions).mean()

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)



def goal_checker(state):
    
    achieved_goal = []
    for obj, indices in OBS_ELEMENT_INDICES.items():
        g = OBS_ELEMENT_GOALS[obj]
        distance = np.linalg.norm(state[indices] -g)   
        if distance < BONUS_THRESH:
            achieved_goal.append(obj)
    return achieved_goal




def V_loss(x, y):
    """
    Variance Loss
    """
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    return std_loss

def COV_loss(x, y):
    """
    """
    B, D = x.shape

    cov_x = (x.T @ x) / (B - 1)
    cov_y = (y.T @ y) / (B - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)

    return cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()