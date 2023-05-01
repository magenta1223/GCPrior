from copy import deepcopy
import numpy as np
from ...contrib.spirl.kitchen_data_loader import D4RLSequenceSplitDataset
from easydict import EasyDict as edict

import random
import math
# from proposed.collector.storage import Offline_Buffer
from ...collector.storage import Offline_Buffer


OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3


class D4RL_StateConditionedDataset(D4RLSequenceSplitDataset):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)


        for k, v in self.spec.items():
            setattr(self, k, v) 

        for k, v in kwargs.items():
            setattr(self, k, v)    

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = dict(
            states = seq.states[start_idx:start_idx+self.subseq_len, :self.n_obj + self.n_env],
            actions = seq.actions[start_idx:start_idx+self.subseq_len-1],
            # pad_mask = np.ones((self.subseq_len,)),
            # state_labels = seq.state_labels[start_idx:start_idx+self.subseq_len]
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]

        return output

class D4RL_StateConditioned_Diversity_Dataset(D4RL_StateConditionedDataset):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
        
        self.__mode__ = "skill_learning"
        
        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        self.buffer_prev = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 50, max_size= 100000)
        self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 50, max_size= 100000)

    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__

    def enqueue(self, states, actions):
        self.buffer_now.enqueue(states, actions)

    def update_buffer(self):
        print("BUFFER RESET!!! ")        
        self.buffer_prev.copy_from(self.buffer_now)
        self.buffer_now.reset()

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()


    def __skill_learning__(self):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = dict(
            states = seq.states[start_idx:start_idx+self.subseq_len, :self.n_obj + self.n_env],
            actions = seq.actions[start_idx:start_idx+self.subseq_len-1],
            # pad_mask = np.ones((self.subseq_len,)),
            # state_labels = seq.state_labels[start_idx:start_idx+self.subseq_len]
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() > 0.95:
            states, actions = self.buffer_prev.sample()

            # trajectory
            output = edict(
                states=states[:self.subseq_len],
                actions=actions[:self.subseq_len-1],
            )

            return output
        else:
            return self.__skill_learning__()


class D4RL_GoalConditionedDataset(D4RL_StateConditionedDataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    # def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
    #     super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """

        goal_max_index = len(states) - 1 # 마지막 state가 이상함. 
        start_idx = np.random.randint(min_idx, states.shape[0] - self.subseq_len - 1)
        
        if self.last:
            return start_idx, goal_max_index

        # start + sub_seq_len 이후 중 아무거나 하나
        goal_index = np.random.randint(start_idx + self.subseq_len, goal_max_index)
        # _min_ = min(start_idx + self.subseq_len + 50, goal_max_index- 1)
        # _max_ = goal_max_index
        # goal_index = np.random.randint(_min_, _max_)


        # 적절한 planning을 위해 relabeled 

        return start_idx, goal_index
        


    def __getitem__(self, index):
        # sample start index in data range
        seq = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq.states)
        
        # trajectory
        states = seq.states[start_idx : start_idx+self.subseq_len][:, :self.n_obj + self.n_env]
        actions = seq.actions[start_idx:start_idx+self.subseq_len-1]
        
        # goal final
        G = deepcopy(seq.states[goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state
        # G[ self.n_obj + self.n_env : ] = 0  # goal state 밀기 어차피 분리해서 받을거 .


        output = edict(
            # relabeled_inputs = relabeled_inputs,
            states=states,
            actions=actions,
            G = G,
        )

        return output


class D4RL_GoalConditioned_Diversity_Dataset(D4RL_GoalConditionedDataset):
    """
    """

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }


        if self.only_proprioceptive:
            self.state_dim = self.n_obj
        else:
            self.state_dim = self.n_obj + self.n_env
        
        print("STATE DIM", self.state_dim)
        
        # 10 step 이후에 skill dynamics로 추론해 error 누적 최소화 
        self.buffer_prev = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 19, max_size= int(1e5))
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= int(1e5))
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 23, max_size= 1024)
        
        # rollout method
        skill_length = self.subseq_len - 1
        if self.rollout_method == "rollout":
            # 0~11 : 1 skill
            # 12~  : 1skill per timestep
            # total 100 epsiode planning
            # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
            
            rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
            self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)

        else:
            rollout_length = 2 * skill_length + ((self.plan_H - skill_length * 2) // skill_length)
            self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)



    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__

    def enqueue(self, states, actions):
        self.buffer_now.enqueue(states, actions)

    def update_buffer(self):
        print("BUFFER RESET!!! ")        
        self.buffer_prev.copy_from(self.buffer_now)
        # self.buffer_now.reset()

    def __getitem__(self, index):
        # mode에 따라 다른 sampl 
        return self.__getitem_methods__[self.mode]()
        
    def __skill_learning__(self):
        
        seq_skill = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq_skill.states)

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        # states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.n_obj + self.n_env]
        states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.state_dim]
        actions = seq_skill.actions[start_idx:start_idx+self.subseq_len-1]

        # if self.only_proprioceptive:
        #     states = states[:, :self.n_obj]
        
        # hindsight relabeling 
        G = deepcopy(seq_skill.states[goal_idx])[:self.n_obj + self.n_env]
        G[ : self.n_obj] = 0 # only env state


        # G = deepcopy(seq_skill.states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]


        output = edict(
            states=states,
            actions=actions,
            G = G,
            rollout = True
            # rollout = True if start_idx < 280 - self.plan_H else False
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio:
            # T, state_dim + action_dim
            # states, actions = self.buffer_prev.sample()
            states, actions = self.buffer_now.sample()

            if self.only_proprioceptive:
                states = states[:, :self.n_obj]

            # hindsight relabeling 
            goal_idx = -1
            # G = deepcopy(states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]
            G = deepcopy(states[goal_idx])[:self.n_obj + self.n_env]
            G[ : self.n_obj] = 0 # only env state

            # trajectory
            output = edict(
                states=states[:self.subseq_len],
                actions=actions[:self.subseq_len-1],
                G=G,
                rollout = False,
                # start_idx = 999 #self.novel
            )

            return output
        else:
            return self.__skill_learning__()




class Kitchen_AEDataset(D4RL_StateConditionedDataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

        # for k, v in kwargs.items():
        #     setattr(self, k, v)

        self.scale = 5e-3

        states = deepcopy(self.seqs[0].states)
        for seq in self.seqs:
            states = np.concatenate((states, deepcopy(seq.states)), axis=  0)
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # sample start index in data range        
        state = deepcopy(self.states[index])[:self.n_obj + self.n_env]

        # if random.random() > 0.8:
        #     state += np.random.randn(*state.shape) * self.scale
        
        # 여기서 env state에 해당하는 경우, 각 object별로 차지하는 자릿수가 다름. 
        # 그래서 많은 공간을 차지하는 object의 경우 representation에 과반영된다.
        # 이를 막기 위해 각 object error scale을 맞춰야 함. 
        
        # for k, v in OBS_ELEMENT_INDICES.items():
        #     state[v] = state[v] / math.sqrt(len(v))

        # if np.random.rand() > 0.8:
        #     pass
        # else:
        #     state[:self.n_obj] = 0

        return dict(states = state)
