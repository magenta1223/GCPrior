from copy import deepcopy
import d4rl
import gym
import numpy as np
from proposed.contrib.spirl.kitchen_data_loader import D4RLSequenceSplitDataset
from easydict import EasyDict as edict

from proposed.utils import goal_checker
import random


class D4RLGoalRelabelingDataset(D4RLSequenceSplitDataset):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size)

        for k, v in kwargs.items():
            setattr(self, k, v)


    def sample_indices(self, states): 
        if self.goal_range[1] == -1:
            start_idx = np.random.randint(0, states.shape[0] - self.subseq_len - self.goal_range[0] - 1)
        else:
            start_idx = np.random.randint(0, states.shape[0] - self.subseq_len - self.goal_range[1] - 1)

        if self.last:
            return start_idx, len(states) - 1

        range_min = start_idx + self.subseq_len+ self.goal_range[0]
        if self.goal_range[1] == -1:
            range_max = len(states)
        else:
            range_max = start_idx + self.subseq_len + self.goal_range[1] - 1
  
        sample_range = (range_min, range_max)

        return start_idx, random.sample(range(*sample_range), 1)[0]

    def prep(self, goal_state):
        if not self.g_agent:
            # 여기서 goal obj를 지우면 됨
            goal_state[:self.n_obj] = 0
        return goal_state

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx, goal_idx = self.sample_indices(seq.states)
        states = deepcopy(seq.states[start_idx : start_idx+self.subseq_len])
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]
        _gs = self.prep(deepcopy(seq.states[goal_idx][:30]))

        # expanded pseudo goal
        goal_state  =np.stack([_gs for _ in range(self.subseq_len)], 0)
        # goal relabeling 
        states[:, 30:] = goal_state

        output = edict(
            states=states,
            actions=actions,
            pad_mask=np.ones((self.subseq_len,)),
        )

        return output


class D4RLDIDDataset(D4RLSequenceSplitDataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.sub_goal_idx = 10
    
        


    def sample_indices(self, states): 
        """
        1) n-step 이후를 subgoal로 봄
        2) final goal은 적당히 먼 미래의 상태로
        """


        start_idx = np.random.randint(0, states.shape[0] - self.subseq_len - self.goal_range[1] - 1)


        range_min = start_idx + self.subseq_len+ self.goal_range[0]
        
        if self.goal_range[1] == -1:
            range_max = len(states)
        else:
            range_max = start_idx + self.subseq_len + self.goal_range[1] - 1
  
        sample_range = (range_min, range_max)

        return start_idx, random.sample(range(*sample_range), 1)[0]

    def prep(self, goal_state):
        if not self.g_agent:
            # 여기서 goal obj를 지우면 됨
            goal_state[:self.n_obj] = 0
        return goal_state

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()

        # sample indices 
        start_idx, goal_idx = self.sample_indices(seq.states)
        
        # trajectory
        states = deepcopy(seq.states[start_idx : start_idx+self.subseq_len])
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]
        
        # goal final
        G = deepcopy(seq.states[goal_idx])
        # only env state
        G[ : self.n_obj] = 0 # agent state 밀기 
        G[ self.n_obj + self.n_env : ] = 0  # goal state 밀기

        # G = G[self.n_obj : self.n_obj + self.n_env]

        output = edict(
            states=states,
            actions=actions,
            G = G,
            pad_mask=np.ones((self.subseq_len,)),
        )

        return output

class D4RLGCIDDataset(D4RLDIDDataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
    
    

    def prep(self, goal_state):
        if not self.g_agent:
            # 여기서 goal obj를 지우면 됨
            goal_state[:self.n_obj] = 0
        return goal_state

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()

        # sample indices 
        start_idx, goal_idx = self.sample_indices(seq.states)
        
        # trajectory
        post_inputs = seq.states[start_idx : start_idx+self.subseq_len].copy()
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1].copy()
        relabeled_inputs = post_inputs.copy()
        
        # goal final
        G = seq.states[goal_idx].copy()
        G[ : self.n_obj] = 0 # only env state
        G[ self.n_obj + self.n_env : ] = 0  # goal state 밀기
        # G_seq = np.stack([ G[:self.n_obj + self.n_env].copy() for _ in range(post_inputs.shape[0])], axis = 0)
        # relabeled_inputs[:, 30:] = G_seq

        output = edict(
            relabeled_inputs = relabeled_inputs,
            states=post_inputs,
            actions=actions,
            G = G
        )

        return output