# https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
import torch

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    # normalization layer : hard update
    # for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
    #     old_weight, up_weight = ma_params.data, current_params.data
    #     ma_params.data = ema_updater.update_average(old_weight, up_weight)

    for (n1, current_params), (n2, ma_params) in zip(current_model.state_dict().items(), ma_model.state_dict().items()):
        if "running_" in n1:
            up_weight =  current_params.data
            ma_params.data = ema_updater.update_average(None, up_weight)
            # ma_params.data.mul_(0)
            # torch.add(ma_params.data, current_params.data, alpha = 0, out = ma_params.data)
        else:
            # ma_params.data.mul_(0.99)
            # torch.add(ma_params.data, current_params.data, alpha = 0.01, out = ma_params.data)          
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = ema_updater.update_average(old_weight, up_weight)