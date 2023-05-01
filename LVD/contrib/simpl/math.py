import numpy as np
import torch.distributions as torch_dist


def clipped_kl(a, b, clip=20):
    kls = torch_dist.kl_divergence(a, b)
    scales =  kls.detach().clamp(0, clip) / kls.detach()
        
    # x * ( max(x, 20) / x )
    # 20보다 큰 x는 20이 될거고, 나머지는 1이 됨. 
    # 그냥 clamp하는 것과 차이가 없는데? 이걸로 어떻게 수렴시킴 무친놈아.. 

    return kls*scales

def inverse_softplus(x):
    return float(np.log(np.exp(x) - 1))

def inverse_sigmoid(x):
    return float(-np.log(1/x - 1))
