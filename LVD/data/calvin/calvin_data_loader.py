from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict

import random
import math

# from proposed.contrib.spirl.data_loader import Dataset
# from proposed.collector.storage import Offline_Buffer

from ...collector.storage import Offline_Buffer
from ...contrib.spirl.data_loader import Dataset

class CALVIN_Dataset(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args,  **kwargs):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        # env
        # self.seqs = np.load("/home/magenta1223/skill-based/SiMPL/proposed/contrib/calvin/dataset/calvin_states.pkl", allow_pickle= True)
        # split dataset into sequences
        self.seqs = np.load("./LVD/contrib/calvin/dataset/calvin_states.pkl", allow_pickle= True)


        self.dataset_len = sum([  seq['obs'].shape[0] for seq in self.seqs])



        for k, v in self.spec.items():
            setattr(self, k, v) 

        for k, v in kwargs.items():
            setattr(self, k, v)    


        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq['obs'].shape[0] - self.subseq_len - 1)
        output = edict(
            states=seq['obs'][start_idx:start_idx+self.subseq_len][:, :self.n_obj + self.n_env],
            actions=seq['actions'][start_idx:start_idx+self.subseq_len-1],
        )

        return output

    def _sample_seq(self, index = False):
        try:
            if index:
                print(f"index {index}")
                return self.seqs[index]
            else:
                return np.random.choice(self.seqs[self.start:self.end])
        except:
            return self.seqs[-1]

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset_len / self.subseq_len)


class CALVIN_Diversity_Dataset(CALVIN_Dataset):
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
        output = edict(
            states=seq['obs'][start_idx:start_idx+self.subseq_len],
            actions=seq['actions'][start_idx:start_idx+self.subseq_len-1],
        )

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


class CALVIN_GoalConditionedDataset(CALVIN_Dataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)


    def sample_indices(self, states, min_idx = 0): 
        """
        return :
            - start index of sub-trajectory
            - goal index for hindsight relabeling
        """
        max_index = len(states) - 1 # 마지막 state가 이상함. 
        start_idx = np.random.randint(min_idx, states.shape[0] - self.subseq_len - 1)
        
        if self.last:
            return start_idx, max_index

        # start + sub_seq_len 이후 중 아무거나 하나
        return start_idx, np.random.randint(start_idx + self.subseq_len, max_index)
        
    def __getitem__(self, index):
        # sample start index in data range
        seq = deepcopy(self._sample_seq())
        start_idx, goal_idx = self.sample_indices(seq.states)
        
        # trajectory
        states = seq['obs'][start_idx : start_idx+self.subseq_len][:, :self.n_obj + self.n_env]
        actions = seq['actions'][start_idx:start_idx+self.subseq_len-1]
        
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



class CALVIN_GoalConditioned_Diversity_Dataset(CALVIN_GoalConditionedDataset):
    """
    """

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        
        self.buffer_prev = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= int(1e5))
        
        if self.rollout_method == "rollout":
            # 0~11 : 1 skill
            # 12~  : 1skill per timestep
            # total 100 epsiode planning
            # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
            self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 19, max_size= 1024)

        else:
            # 0~20 : 2 skill
            # 12~  : 1skill per timestep
            # total 50 epsiode planning
            self.buffer_now = Offline_Buffer(state_dim= self.state_dim, action_dim= self.action_dim, trajectory_length = 28, max_size= 1024)


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
        start_idx, goal_idx = self.sample_indices(seq_skill['obs'])

        goal_idx = min(start_idx + 100, goal_idx)

        assert start_idx < goal_idx, "Invalid"

        # trajectory
        states = seq_skill['obs'][start_idx : start_idx+self.subseq_len][:, :self.n_obj + self.n_env]
        actions = seq_skill['actions'][start_idx:start_idx+self.subseq_len-1]
        
        # goal final
        G = deepcopy(seq_skill['obs'][goal_idx])[self.n_obj :self.n_obj + self.n_goal]


        output = edict(
            states=states,
            actions=actions,
            G = G,
            rollout = True
        )

        return output

    def __skill_learning_with_buffer__(self):

        if np.random.rand() < self.mixin_ratio:
            # T, state_dim + action_dim
            # states, actions = self.buffer_prev.sample()
            states, actions = self.buffer_now.sample()


            # G = deepcopy(states[-1])[:self.n_obj + self.n_env]
            G = deepcopy(states[-1])[self.n_obj :self.n_obj + self.n_goal]


            # trajectory
            output = edict(
                states=states[:self.subseq_len],
                actions=actions[:self.subseq_len-1],
                G=G,
                rollout = False,
            )

            return output
        else:
            return self.__skill_learning__()
        


class CALVIN_AEDataset(CALVIN_Dataset):
    """
    D4RL Goal Relabeling & subgoal dataset
    """
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.scale = 5e-3

        states = deepcopy(self.seqs[0]['obs'])
        for seq in self.seqs[1:]:
            states = np.concatenate((states, deepcopy(seq['obs'])), axis=  0)
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
