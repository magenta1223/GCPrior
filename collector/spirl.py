from proposed.contrib.simpl.collector.hierarchical import HierarchicalEpisode 
from proposed.utils import get_dist
import numpy as np

class HierarchicalTimeLimitCollector:
    def __init__(self, env, horizon, time_limit=None):
        self.env = env
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf

    def collect_episode(self, low_actor, high_actor):
        state, done, t = self.env.reset(), False, 0
        episode = HierarchicalEpisode(state)


        while not done and t < self.time_limit:
            if t % self.horizon == 0:
                # horizon마다 action을 바꿈. 
                high_action = high_actor.act(state) # 여기를 state하고 action 둘 다 들어가게. posterior 사용 시
                data_high_action = high_action
            else:
                data_high_action = None


            with low_actor.condition(high_action):
                low_action = low_actor.act(state)

            state, reward, done, info = self.env.step(low_action)


            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']

            episode.add_step(low_action, data_high_action, state, reward, data_done, info)
            t += 1

        return episode

    
class LowFixedHierarchicalTimeLimitCollector(HierarchicalTimeLimitCollector):
    def __init__(self, env, low_actor, horizon, time_limit=None):
        super().__init__(env, horizon, time_limit)
        self.low_actor = low_actor

    def collect_episode(self, high_actor):
        return super().collect_episode(self.low_actor, high_actor)