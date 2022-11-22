import torch
import torch.nn as nn
from contextlib import contextmanager
import numpy as np
from proposed.utils import get_dist

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
                    setattr(self, k, v)
            except:
                for k, v in config.items():
                    setattr(self, k, v)           

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
        self._device = nn.Parameter(torch.zeros(1))


    def build(self, config):
        layers = []
        for args in config.build:
            cls, args = args[0], args[1:]
            layers.append(cls(*args))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = x 
        for layer in self.layers:
            out = layer(out)
            if isinstance(out, tuple): # rnn
                out = out[0]
        return out


    def dist(self, *args, detached = False):
        result = self(*args)

        if detached:
            return get_dist(result), get_dist(result, detached= True)
        else:
            return get_dist(result)

    
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

    def dist(self, batch_state_z):
        return super(ContextPolicyMixin, self).dist(batch_state_z)

    def dist_with_z(self, batch_state, batch_z):
        batch_state_z = torch.cat([batch_state, batch_z], dim=-1)
        return self.dist(batch_state_z)


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_cls = None, act_cls = None, bias = False):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim, bias = bias)]
        if norm_cls is not None:
            # 일단 BN만 가정
            layers.append(norm_cls(out_dim))

        if act_cls is not None:
            if act_cls == nn.LeakyReLU:
                layers.append(act_cls(0.2, True))
            else:
                layers.append(act_cls(inplace= True))


        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

