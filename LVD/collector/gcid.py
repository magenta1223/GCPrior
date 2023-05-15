from ..contrib.simpl.collector.hierarchical import HierarchicalEpisode 
from ..utils import StateProcessor

# from ..contrib.simpl.collector.hierarchical import HierarchicalEpisode
# from ..utils import GOAL_CHECKERS

import numpy as np
from copy import deepcopy
import torch






class HierarchicalTimeLimitCollector:
    def __init__(self, env, env_name, horizon, time_limit=None, tanh = False):
        self.env = env
        self.env_name = env_name
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.tanh = tanh

        self.state_processor = StateProcessor(env_name= self.env_name)


    def collect_episode(self, low_actor, high_actor, verbose = True):
        state, done, t = self.env.reset(), False, 0

        # G = GOAL_TRANSFORM[self.env_name](state)
        # state = STATE_PROCESSOR[self.env_name](state)
        
        G = self.state_processor.get_goals(state)
        state = self.state_processor.state_process(state)

        # print(f"G : {self.state_processor.goal_checker(G)}")

        episode = HierarchicalEpisode(state)
        low_actor.eval()
        high_actor.eval()
        
        while not done and t < self.time_limit:

            # print(goal_checker(state[30:]))
            if t % self.horizon == 0:
                if self.tanh:
                    high_action_normal, high_action, loc, scale = high_actor.act(state, G)
                    data_high_action_normal, data_high_action = high_action_normal, high_action
                else:
                    high_action, loc, scale = high_actor.act(state, G)
                    data_high_action = high_action
            else:
                data_high_action = None
            
            # print(high_action.shape)
            
            with low_actor.condition(high_action):
                low_action = low_actor.act(state)


            state, reward, done, info = self.env.step(low_action)
            # state = STATE_PROCESSOR[self.env_name](state)
            state = self.state_processor.state_process(state)


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
                    data_high_action = np.concatenate((data_high_action, loc, scale), axis = 0)
                else:
                    data_high_action = None

                episode.add_step(low_action, data_high_action, state, reward, data_done, info)
            t += 1
            
                
    
        # print(GOAL_CHECKERS[self.env_name](STATE_PROCESSOR[self.env_name] (state)))
        if verbose:
            print( self.state_processor.state_goal_checker(state, self.env)  )
        
        # print(np.array(episode.actions)[:, 0].mean())
        # print(np.mean([info['vel'] for info in episode.infos]))
        print(len(episode))


    

        return episode, torch.tensor(G, dtype = torch.float32).unsqueeze(0)

    
class LowFixedHierarchicalTimeLimitCollector(HierarchicalTimeLimitCollector):
    def __init__(self, env, env_name, low_actor, horizon, time_limit=None, tanh = False):
        super().__init__(env, env_name, horizon, time_limit, tanh)
        self.low_actor = low_actor

    def collect_episode(self, high_actor):
        return super().collect_episode(self.low_actor, high_actor)
