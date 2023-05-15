from .storage import Episode
from ....utils import StateProcessor
import numpy as np

class HierarchicalEpisode(Episode):
    def __init__(self, init_state):
        super().__init__(init_state)
        self.low_actions = self.actions
        self.high_actions = []


    
    def add_step(self, low_action, high_action, next_state, reward, done, info):
        # MDP transitions
        super().add_step(low_action, next_state, reward, done, info)
        self.high_actions.append(high_action)

    def as_high_episode(self):
        """
        high-action은 H-step마다 값이 None 이 아
        """

        high_episode = Episode(self.states[0])
        prev_t = 0
        for t in range(1, len(self)):
            if self.high_actions[t] is not None:
                # high-action은 H-step마다 값이 None이 아니다.
                # raw episode를 H-step 단위로 끊고, action을 high-action으로 대체해서 넣음. 
                high_episode.add_step(
                    self.high_actions[prev_t], self.states[t],
                    sum(self.rewards[prev_t:t]), self.dones[t], self.infos[t]
                )
                prev_t = t
        
        high_episode.add_step(
            self.high_actions[prev_t], self.states[-1],
            sum(self.rewards[prev_t:]), self.dones[-1], self.infos[-1]
        )
        high_episode.raw_episode = self
        return high_episode


class HierarchicalTimeLimitCollector:
    def __init__(self, env, horizon, env_name, time_limit=None):
        self.env = env
        self.horizon = horizon
        self.time_limit = time_limit if time_limit is not None else np.inf
        self.state_processor = StateProcessor(env_name)

        

    def collect_episode(self, low_actor, high_actor):
        state, done, t = self.env.reset(), False, 0
        episode = HierarchicalEpisode(state)

        G = self.state_processor.get_goals(state)
        # state = self.state_processor.state_process(state)
        print(f"G : {self.state_processor.goal_checker(G)}")

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
    def __init__(self, env, low_actor, horizon, env_name, time_limit=None):
        super().__init__(env, horizon, env_name, time_limit)
        self.low_actor = low_actor
        
    def collect_episode(self, high_actor):
        return super().collect_episode(self.low_actor, high_actor)
