import torch

def itemize(d):
    stat = {}
    for k, v in d.items():
        if type(v) == dict:
            stat[k] = itemize(v)
        elif type(v) == torch.Tensor:
            stat[k] = v.item()
        else:
            stat[k] = v
    return stat

class ToDeviceMixin:
    device = None

    def to(self, device):
        self.device = device

        parent = super(ToDeviceMixin, self)
        if hasattr(parent, 'to'):
            return parent.to(device)
        else:
            return self