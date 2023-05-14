from copy import deepcopy
import numpy as np
from easydict import EasyDict as edict

import random
import math
# from proposed.collector.storage import Offline_Buffer
from ...collector.storage import Offline_Buffer

from glob import glob

import h5py
from torch.utils.data import Dataset
from ...contrib.spirl.pytorch_utils import RepeatedDataLoader
import pickle
from torch.utils.data.dataloader import DataLoader, SequentialSampler
import torch
import pickle 

def parse_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class CARLA_Dataset(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        with open("./LVD/data/carla/carla_dataset.pkl", mode ="rb") as f:
            self.seqs = pickle.load(f)

        self.n_seqs = len(self.seqs)
        self.phase = phase
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.device = "cuda"

        for k, v in data_conf.dataset_spec.items():
            setattr(self, k, v) 

        for k, v in kwargs.items():
            setattr(self, k, v)    

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

        self.num = 0

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

        # 적절한 planning을 위해 relabeled 

        return start_idx, goal_index

    def __getitem__(self, idx):


        seq = self.seqs[idx]
        states = seq['obs']
        actions = seq['actions']

        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = states[goal_idx][:2] # ? position이 어딘지 모름 

        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]


        data = {
            'states': states,
            'actions': actions,
            'G' : G,
        }


        return data
    
    def __len__(self):
        if self.phase == "train":
            # return 20000
            return  int(self.SPLIT[self.phase] * self.n_seqs)
        else:
            return int(self.SPLIT[self.phase] * self.n_seqs)

    def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong

        dataloader= RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            n_repeat=n_repeat,
            pin_memory=True, # self.device == 'cuda'
            # pin_memory= False, # self.device == 'cuda'
            # collate_fn = self.collate_fn,
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
            )        
        return dataloader

class CARLA_Dataset_Diversity(CARLA_Dataset):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

        if self.visual == "visual_feature":
            self.buffer_dim = self.state_dim
        else:
            self.buffer_dim = self.state_dim #+ 1024

        # 10 step 이후에 skill dynamics로 추론해 error 누적 최소화 
        self.buffer_prev = Offline_Buffer(state_dim= self.buffer_dim, action_dim= self.action_dim, trajectory_length = 19, max_size= 1024)

        
        # rollout method
        skill_length = self.subseq_len - 1
        rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        self.buffer_now = Offline_Buffer(state_dim= self.buffer_dim, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)


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
        return self.__getitem_methods__[self.mode](index)
        
    def __skill_learning__(self, index):

        seq = self.seqs[index]
        # states = deepcopy(seq['states'])
        states = deepcopy(seq['obs'])
        actions = seq['actions']
        
        start_idx, goal_idx = self.sample_indices(states)
        assert start_idx < goal_idx, "Invalid"

        G = states[goal_idx][:2] # ? 
        states = states[start_idx : start_idx + self.subseq_len]
        actions = actions[start_idx : start_idx + self.subseq_len -1]

        data = {
            'states': states,
            'actions': actions,
            'G' : G,
            'rollout' : True
        }

        return data

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            states_images, actions = self.buffer_now.sample()
            
            output = edict(
                states = states_images[:self.subseq_len],
                actions = actions[:self.subseq_len-1],
                G = deepcopy(states_images[-1][:self.n_obj] ),
                rollout = False
                # rollout = True if start_idx < 280 - self.plan_H else False
            )

            return output
        else:
            return self.__skill_learning__(index)
