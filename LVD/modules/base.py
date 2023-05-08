import torch
import torch.nn as nn
from torch.nn import functional as F

from contextlib import contextmanager
import numpy as np
from copy import deepcopy


# from proposed.utils import get_dist
from ..utils import get_dist


import math

class BaseModule(nn.Module):
    """
    Module class equipped with common methods for logging & distributions 
    """
    def __init__(self, config):
        super(BaseModule, self).__init__()
        self.set_attrs(config)
        self._device = nn.Parameter(torch.zeros(1))

    # set configs
    def set_attrs(self, config = None):
        if config is not None:
            try:
                for k, v in config.attrs.items():
                    setattr(self, k, deepcopy(v))
            except:
                for k, v in config.items():
                    setattr(self, k, deepcopy(v))           

    def forward(self, x):
        return NotImplementedError

    @property
    def device(self):
        return self._device.device



class SequentialBuilder(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.build(config)
        self.explore = None

    def build(self, config):
        layers = []
        for args in config.build:
            cls, args = args[0], args[1:]
            layers.append(cls(*args))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, *args, **kwargs):
        out = x 
        for layer in self.layers:
            out = layer(out)
            if isinstance(out, tuple): # rnn
                out = out[0]
        return out


    def dist(self, *args, detached = False):
        result = self(*args)

        if detached:
            return get_dist(result, tanh = self.tanh), get_dist(result, detached= True, tanh = self.tanh)
        else:
            return get_dist(result, tanh = self.tanh)



    # @contextmanager
    # def turn_off_tracking_running_stat(self):
    #     """
    #     for bn
    #     """
    #     def _tracking_running_stat(layer, status):
    #         if isinstance(layer, nn.BatchNorm1d):
    #             layer.track_running_stats = status
    #         elif hasattr(layer, "layers"): 
    #             for _layer in layer.layers:
    #                 _tracking_running_stat(_layer, status)
    #         else:
    #             pass      
    #     for layer in self.layers:
    #         _tracking_running_stat(layer, False)
    #     yield
    #     for layer in self.layers:
    #         _tracking_running_stat(layer, True)


    
    # from simpl
    @contextmanager
    def no_expl(self):
        explore = self.explore
        self.explore = False
        yield
        self.explore = explore

    @contextmanager
    def expl(self):
        explore = self.explore
        self.explore = True
        yield
        self.explore = explore

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_cls = None, act_cls = None, bias = False, dropout = 0):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim,  bias = bias)]
        if norm_cls is not None:
            layers.append(norm_cls(out_dim))

        if act_cls is not None:
            if act_cls == nn.LeakyReLU:
                layers.append(act_cls(0.2, True))
            else:
                layers.append(act_cls(inplace= True))
        
        if dropout != 0:
            layers.append(nn.Dropout1d(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# from simpl
class ContextPolicyMixin:
    z_dim = NotImplemented
    z = None

    @contextmanager
    def condition(self, z):
        if type(z) != np.ndarray or z.shape != (self.z_dim, ):
            raise ValueError(f'z should be np.array with shape {self.z_dim}, but given : {z}')
        prev_z = self.z
        self.z = z
        yield
        self.z = prev_z

    def act(self, state):
        if self.z is None:
            raise RuntimeError('z is not set')
        state_z = np.concatenate([state, self.z], axis=0)
        return super(ContextPolicyMixin, self).act(state_z)

    def dist(self, batch_state_z, tanh = False):
        return super(ContextPolicyMixin, self).dist(batch_state_z, tanh= tanh)

    def dist_with_z(self, batch_state, batch_z, tanh = False):
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.dist(batch_state_z, tanh= tanh)
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        
        return x.view(-1, np.prod(x.shape[1:]))