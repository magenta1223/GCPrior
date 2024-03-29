from ..contrib.simpl.collector.hierarchical import HierarchicalEpisode 
from ..utils import *

# from ..contrib.simpl.collector.hierarchical import HierarchicalEpisode
# from ..utils import GOAL_CHECKERS

import numpy as np
from copy import deepcopy
import torch
from ..utils import StateProcessor


class HierarchicalTimeLimitCollector:
    def __init__(self, env, env_name, horizon, time_limit=None, tanh = False):
        self.env = env
        self.env_name = env_name
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.tanh = tanh
        self.state_processor = StateProcessor(env_name= self.env_name)

    def collect_episode(self, low_actor, high_actor):
        state, done, t = self.env.reset(), False, 0

        episode = HierarchicalEpisode(state)
        low_actor.eval()
        high_actor.eval()
        
        while not done and t < self.time_limit:

            if t % self.horizon == 0:
                if self.tanh:
                    high_action_normal, high_action, loc, scale = high_actor.act(state)
                    data_high_action_normal, data_high_action = high_action_normal, high_action
                else:
                    high_action, loc, scale = high_actor.act(state)
                    data_high_action = high_action
            else:
                data_high_action = None
            
            # print(high_action.shape)
            
            with low_actor.condition(high_action):
                low_action = low_actor.act(state)

            state, reward, done, info = self.env.step(low_action)

            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            
            if self.tanh:
                if data_high_action is not None:
                    _data_high_action = np.concatenate((data_high_action_normal, data_high_action, loc, scale), axis = 0)
                else:
                    _data_high_action = None
                episode.add_step(low_action, _data_high_action, state, reward, data_done, info)
            else:
                if data_high_action is not None:
                    # data_high_action = np.concatenate((data_high_action, loc, scale), axis = 0)
                    pass
                else:
                    data_high_action = None

                episode.add_step(low_action, data_high_action, state, reward, data_done, info)
            t += 1
        

        # print(GOAL_CHECKERS[self.env_name](STATE_PROCESSOR[self.env_name](state)))
        if self.env_name != "maze":
            print( self.state_processor.state_goal_checker(state)  )
        else:
            print( self.state_processor.state_goal_checker(state)  )



        return episode, None

    
class LowFixedHierarchicalTimeLimitCollector(HierarchicalTimeLimitCollector):
    def __init__(self, env, env_name, low_actor, horizon, time_limit=None, tanh = False):
        super().__init__(env, env_name, horizon, time_limit, tanh)
        self.low_actor = low_actor

    def collect_episode(self, high_actor):
        return super().collect_episode(self.low_actor, high_actor)
