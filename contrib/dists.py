from torch.distributions import Distribution, Normal
from torch.nn import functional as F
import torch


# Tanh Normal Distribution
# https://github.com/navneet-nmk/pytorch-rl/blob/master/Distributions/distributions.py
class TanhNormal(Distribution):
    """
        Represent distribution of X where
            X ~ tanh(Z)
            Z ~ N(mean, std)
        Note: this is not very numerically stable.
        """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        super(TanhNormal, self).__init__()
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

        self._batch_shape = self.normal.batch_shape

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return F.tanh(z), z
        else:
            return F.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_sigmoid_value: arcsigmoid(x)
        :return:
        """
        value= value.clamp( -1 + self.epsilon, 1 - self.epsilon )
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                    (1+value) / (1 - value)
                )/2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
                1- value*value + self.epsilon
        )

    def sample(self, return_pre_tanh_value=False):
        z = self.normal.sample()
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pre_tanh_value=False):
        z = self.normal.rsample()
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


    def __repr__(self):
        return "TanhNormal"