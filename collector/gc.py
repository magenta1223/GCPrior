from proposed.contrib.simpl.collector.hierarchical import HierarchicalEpisode 
from proposed.utils import get_dist
import numpy as np

class HierarchicalTimeLimitCollectorGC:
    def __init__(self, env, horizon, time_limit=None, g_agent = True):
        self.env = env
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.g_agent = g_agent
    def collect_episode(self, low_actor, high_actor, prior_module, goal_generator):
        env_state, done, t = self.env.reset(), False, 0
        episode = HierarchicalEpisode(env_state)

        # goal을 초기에 설정하고 계속 바꿔주고 있음.
        # 말이 안될 것은 없다. 움직이고 > 현재 상태에서 goal알아보고 > 반복

        if self.g_agent:
            state_input = goal_generator.mix_pseudo_g_obj(env_state, detached = True).cpu().numpy()    
            state = state_input[0]
        else:
            state_input = env_state
            state = env_state


        while not done and t < self.time_limit:
            if t % self.horizon == 0:
                # horizon마다 action을 바꿈. 
                high_action = high_actor.act(state_input) # 여기를 state하고 action 둘 다 들어가게. posterior 사용 시
                data_high_action = high_action
            else:
                data_high_action = None


            with low_actor.condition(high_action):
                low_action = low_actor.act(state)

            env_state, reward, done, info = self.env.step(low_action)
            if self.g_agent:
                state_input = goal_generator.mix_pseudo_g_obj(env_state, detached = True).cpu().numpy() 
                state = state_input[0]

            else:
                state_input = env_state
                state = env_state

            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']

            episode.add_step(low_action, data_high_action, state, reward, data_done, info)
            t += 1

        return episode

    
class LowFixedHierarchicalTimeLimitCollectorGC(HierarchicalTimeLimitCollectorGC):
    def __init__(self, env, low_actor, horizon, time_limit=None, g_agent = True):
        super().__init__(env, horizon, time_limit, g_agent)
        self.low_actor = low_actor
        
    def collect_episode(self, high_actor, prior_module, goal_generator):
        return super().collect_episode(self.low_actor, high_actor, prior_module, goal_generator)