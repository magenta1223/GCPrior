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

def parse_h5(file_path):
    f = h5py.File(file_path)
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
    return edict( 
        states = np.array(f.get("states")),
        actions = np.array(f.get("actions")),
        agent_centric_view = np.array(f.get("images")),
    )

def parse_pkl(file_path):
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

class Maze_StateConditioned(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        # super().__init__(data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1)


        # self.file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze/**/*.h5")        
        # self.seqs = [parse_h5(file_path) for file_path in self.file_paths]

        file_name = '/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze.pkl'

        # 파일 로드
        with open(file_name, 'rb') as f:
            self.seqs = pickle.load(f)



        self.n_seqs = len(self.seqs)

        self.n_obs = sum([ seq.states.shape[0]  for seq in self.seqs])

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
        return int(self.SPLIT[self.phase] * self.n_obs / self.subseq_len)


    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)

        output = dict(
            states = seq.states[start_idx : start_idx + self.subseq_len],
            actions = seq.actions[start_idx : start_idx + self.subseq_len-1]
        )
        return output
    
    def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
        print(n_repeat)

        return RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            drop_last=False,
            n_repeat=n_repeat,
            pin_memory=self.device == 'cuda', # self.device == 'cuda'
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
            )



# class Maze_AgentCentric_StateConditioned(Dataset):
#     SPLIT = edict(train=0.99, val=0.01, test=0.0)
#     def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
#         # super().__init__(data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1)

#         self.file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/*.h5")        
#         # self.seqs = [parse_h5(file_path) for file_path in self.file_paths]

#         self.n_seqs = len(self.file_paths)
#         self.phase = phase
#         self.dataset_size = dataset_size
#         self.shuffle = shuffle
#         self.device = "cuda"

#         for k, v in data_conf.dataset_spec.items():
#             setattr(self, k, v) 

#         for k, v in kwargs.items():
#             setattr(self, k, v)    

#         if self.phase == "train":
#             self.start = 0
#             self.end = int(self.SPLIT.train * self.n_seqs)
#         elif self.phase == "val":
#             self.start = int(self.SPLIT.train * self.n_seqs)
#             self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
#         else:
#             self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
#             self.end = self.n_seqs

#         self.num = 0

#     def _sample_seq(self, index = False):
#         # try:
#         #     if index:
#         #         print(f"index {index}")
#         #         seq_idx = self.n_seqs - 1
#         #     else:
#         #         # return np.random.choice(self.seqs[self.start:self.end])
#         #         seq_idx = np.random.choice( range(self.start, self.end) )
#         #         # return seq_idx = np.random.choice( range(self.start, self.end) )
#         # except:
#         #     seq_idx = self.n_seqs - 1
#         # return parse_h5(self.file_paths[seq_idx])
#         return parse_h5(self.file_paths[index])


#     def __len__(self):
#         if self.dataset_size != -1:
#             return self.dataset_size
#         # return int(self.SPLIT[self.phase] * len(self.file_paths))
#         return 20000 # superslow


#     def __getitem__(self, index):
#         # sample start index in data range
#         seq = self._sample_seq(index)
#         # self.num += 1
#         # # print(self.num)
#         # if self.num == self.n_seqs:
#         #     self.num = 0
#         #     print("RESET!!")

#         start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)

#         states = seq.states[start_idx : start_idx + self.subseq_len]
#         actions = seq.actions[start_idx : start_idx + self.subseq_len-1]
#         imgs = seq.agent_centric_view[start_idx : start_idx + self.subseq_len]

#         output = dict(
#             states = states,
#             actions = actions,
#             bianry_image = imgs
#         )
#         return output
    
#     def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
#         print('len {} dataset {}'.format(self.phase, len(self)))
#         assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong

#         dataloader= RepeatedDataLoader(
#             self,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             drop_last=False,
#             n_repeat=1,
#             pin_memory=True, # self.device == 'cuda'
#             # pin_memory= False, # self.device == 'cuda'
#             worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
#             )
#         dataloader.set_sampler(SequentialSampler(self))
        
#         return dataloader


class Maze_AgentCentric_StateConditioned(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        # super().__init__(data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1)
        prefix = kwargs.get("prefix", "/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep")
        self.file_paths = glob(f"{prefix}/*.h5")        
        # self.seqs = [parse_h5(file_path) for file_path in self.file_paths]

        self.n_seqs = len(self.file_paths)
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

    def __getitem__(self, idx):
        # return self.file_paths[idx]

        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as f:

            states = np.array(f['states'])

            start_idx = np.random.randint(0, states.shape[0] - self.subseq_len - 1)

            data = {
                'states': states[start_idx : start_idx + self.subseq_len],
                'actions': np.array(f['actions'])[start_idx : start_idx + self.subseq_len -1],
                'images': np.array(f['images'])[start_idx : start_idx + self.subseq_len]
            }


        return data
    
    def __len__(self):
        # return len(self.file_paths)
        # return int(self.SPLIT[self.phase] * len(self.file_paths))

        # return  int(self.SPLIT[self.phase] * (len(self.file_paths) //2  ))
        if self.phase == "train":
            # return 20000
            return  40000
        else:
            return 5000


    def load_data(self, file_path):

        with h5py.File(file_path, 'r') as f:

            states = np.array(f['states'])

            start_idx = np.random.randint(0, states.shape[0] - self.subseq_len - 1)
            states = states[start_idx : start_idx + self.subseq_len]
            images = np.array(f['images'])[start_idx : start_idx + self.subseq_len].reshape(self.subseq_len, -1)

            data = {
                'states': np.concatenate((states, images), axis = -1),
                'actions': np.array(f['actions'])[start_idx : start_idx + self.subseq_len -1],
            }



        return data
    
    def collate_fn(self, batch):
        data = [self.load_data(path) for path in batch]
        states = [d['states'] for d in data]
        actions = [d['actions'] for d in data]
        # images = [d['images'] for d in data]
        return {
            'states': torch.from_numpy(np.stack(states)),
            'actions': torch.from_numpy(np.stack(actions)),
            # 'images': torch.from_numpy(np.stack(images))
        }
    
    def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong

        dataloader= RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            n_repeat=1,
            pin_memory=True, # self.device == 'cuda'
            # pin_memory= False, # self.device == 'cuda'
            # collate_fn = self.collate_fn,
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x)
            )
        # dataloader.set_sampler(SequentialSampler(self))
        
        return dataloader


class Maze_AgentCentric_GoalConditioned_Diversity(Maze_AgentCentric_StateConditioned):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)
        self.__mode__ = "skill_learning"

        self.__getitem_methods__ = {
            "skill_learning" : self.__skill_learning__,
            "with_buffer" : self.__skill_learning_with_buffer__,
        }

                
        # 10 step 이후에 skill dynamics로 추론해 error 누적 최소화 
        self.buffer_prev = Offline_Buffer(state_dim= self.state_dim + 32 * 32, action_dim= self.action_dim, trajectory_length = 19, max_size= 1024)
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= int(1e5))
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 23, max_size= 1024)
        
        # rollout method
        skill_length = self.subseq_len - 1
        # 0~11 : 1 skill
        # 12~  : 1skill per timestep
        # total 100 epsiode planning
        # self.buffer_now = Offline_Buffer(state_dim= 30, action_dim= 9, trajectory_length = 19, max_size= 1024)
        
        rollout_length = skill_length + ((self.plan_H - skill_length) // skill_length)
        self.buffer_now = Offline_Buffer(state_dim= self.state_dim + 32 * 32, action_dim= self.action_dim, trajectory_length = rollout_length, max_size= 1024)


    def set_mode(self, mode):
        assert mode in ['skill_learning', 'with_buffer']
        self.__mode__ = mode 
        print(f"MODE : {self.mode}")
    
    @property
    def mode(self):
        return self.__mode__

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

        file_path = self.file_paths[index]

        with h5py.File(file_path, 'r') as f:

            states = np.array(f['states'])
            actions = np.array(f['actions'])
            images = np.array(f['images'])

        start_idx, goal_idx = self.sample_indices(states)

        assert start_idx < goal_idx, "Invalid"


        # hindsight relabeling 
        G = deepcopy(states[goal_idx])
        # trajectory
        # states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.n_obj + self.n_env]
        states = states[start_idx : start_idx+self.subseq_len, :self.n_obj]
        actions = actions[start_idx:start_idx+self.subseq_len-1]
        images = images[start_idx : start_idx + self.subseq_len].reshape(self.subseq_len, -1)
        # if self.only_proprioceptive:
        #     states = states[:, :self.n_obj]
        
        # G = deepcopy(seq_skill.states[goal_idx])[self.n_obj:self.n_obj + self.n_goal]


        output = edict(
            states= np.concatenate((states, images), axis = -1),
            actions=actions,
            G = G,
            rollout = True
            # rollout = True if start_idx < 280 - self.plan_H else False
        )

        return output

    def __skill_learning_with_buffer__(self, index):

        if np.random.rand() < self.mixin_ratio:
            # hindsight relabeling 
            # trajectory
            # states = seq_skill.states[start_idx : start_idx+self.subseq_len, :self.n_obj + self.n_env]
            states_images, actions = self.buffer_now.sample()
            # images = np.array(f['images'])[: self.subseq_len]
            # if self.only_proprioceptive:
            #     states = states[:, :self.n_obj]
            # states = states_images[:, :self.n_obj]
            # images = states_images[:, self.n_obj:].reshape(-1, 32,32)

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



class Maze_AEDataset(Maze_AgentCentric_StateConditioned):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

    # def __getitem__(self, index):
    #     sample = super().__getitem__(index)
    #     return dict(  states = sample['bianry_image']  )
    
    def load_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            images = np.array(f['images'])
            start_idx = np.random.randint(0, images.shape[0] - self.subseq_len - 1)

            data = {
                'images': images[start_idx : start_idx + self.subseq_len]
            }

        return data
    
    def collate_fn(self, batch):
        data = [self.load_data(path)['images'] for path in batch]
        return {
            'states': torch.from_numpy(np.stack(data))
        }
    
    def __len__(self):
        # return len(self.file_paths)
        # return int(self.SPLIT[self.phase] * len(self.file_paths))
        if self.phase == "train":
            return  40000
        else:
            return 5000    
        # return 5000
    
    def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong

        # return DataLoader(
        #     self,
        #     batch_size=batch_size,
        #     shuffle= False,
        #     num_workers=4,
        #     drop_last=False,
        #     pin_memory= True, # self.device == 'cuda'
        #     # pin_memory= False, # self.device == 'cuda'
        #     worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
        #     )
    
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=14,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory= True,
            sampler = SequentialSampler(self)
        )




