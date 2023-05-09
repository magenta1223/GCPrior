# from proposed.modules.base import BaseModule
# from proposed.utils import *
# from proposed.contrib.momentum_encode import update_moving_average
from ...modules.base import BaseModule
from ...utils import *
from ...contrib.momentum_encode import update_moving_average

class StateConditioned_Prior(BaseModule):
    """
    TODO 
    1) 필요한 모듈이 마구마구 바뀌어도 그에 맞는 method 하나만 만들면
    2) RL이나 prior trainin 에서는 동일한 method로 호출 가능하도록
    """

    def __init__(self, **submodules):
        super().__init__(submodules)

        self.methods = {
            "train" : self.__train__,
            "eval" : self.__eval__,
        }

    
    def forward(self, inputs, mode = "train", *args, **kwargs):
        return self.methods[mode](inputs, *args, **kwargs)


    def __train__(self, inputs, *args):
        """
        State only Conditioned Prior
        inputs : dictionary
            -  states 
        return: state conditioned prior, and detached version for metric
        """
        
        # states = inputs['states']
        states, G = inputs['states'], inputs['G']

        N, T, _ = states.shape


        # -------------- State Enc / Dec -------------- #
        prior = self.prior_policy.dist(states[:, 0])
        policy_skill = self.highlevel_policy.dist(torch.cat((states[:,0], G), dim = -1))

        return dict(
            prior = prior,
            states = states,
            policy_skill = policy_skill,
        )

    def __eval__(self, inputs, *args):
        """
        State only Conditioned Prior
        inputs : dictionary
            -  states 
        return: state conditioned prior, and detached version for metric
        """
        states, G = inputs['states'], inputs['G']
        if len(states.shape) < 2:
            states = states.unsqueeze(0)     

        
        # states = states[:, :self.prior_policy.in_feature]
        policy_skill = self.highlevel_policy.dist(torch.cat((states, G.cuda()), dim = -1))

        prior = self.prior_policy.dist(states)
        return dict(
            prior = prior,
            policy_skill = policy_skill,
        )
