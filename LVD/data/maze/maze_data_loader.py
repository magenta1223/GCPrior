from copy import deepcopy
import numpy as np
from ...contrib.spirl.maze_data_loader import MazeStateSequenceDataset
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

def parse_h5(file_path):
    f = h5py.File(file_path)
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
    traj = f.get("traj0")
    return edict( 
        states = np.array(traj.get("states")),
        actions = np.array(traj.get("actions")),
        agent_centric_view = np.array(traj.get("images")) / 255,
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






class Maze_AgentCentric_StateConditioned(Dataset):
    SPLIT = edict(train=0.99, val=0.01, test=0.0)
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        # super().__init__(data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1)

        color_dict = {
            "wall" : np.array([0.87, 0.62, 0.38]),
            "agent" : np.array([0.32, 0.65, 0.32]),
            "ground_color1" : np.array([0.2, 0.3, 0.4]),
            "ground_color2" : np.array([0.1, 0.2, 0.3]),
        }

        self.file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/**/*.pkl")        
        # self.seqs = [parse_h5(file_path) for file_path in self.file_paths]

        file_name = '/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze.pkl' # from spirl

        # 파일 로드
        with open(file_name, 'rb') as f:
            seqs = pickle.load(f)

        self.n_seqs = len(seqs)
        self.n_obs = sum([ seq.states.shape[0]  for seq in seqs])

        self.phase = phase
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.device = "cuda"

        self.wall = np.full((32, 32, 3), color_dict['wall'])
        self.ground1 = np.full((32, 32, 3), color_dict['ground_color1'])
        self.ground2 = np.full((32, 32, 3), color_dict['ground_color2'])

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
                seq_idx = self.n_seqs - 1
            else:
                # return np.random.choice(self.seqs[self.start:self.end])
                seq_idx = np.random.choice( range(self.start, self.end) )
        except:
            seq_idx = self.n_seqs - 1
    

        return parse_pkl(self.file_paths[seq_idx])
    
    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.n_obs / self.subseq_len)


    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)

        states = seq.states[start_idx : start_idx + self.subseq_len]
        actions = seq.actions[start_idx : start_idx + self.subseq_len-1],
        imgs = seq.agent_centric_view[start_idx : start_idx + self.subseq_len - 1]



        output = dict(
            states = states,
            actions = actions,
            bianry_image = imgs
        )
        return output
    
    def get_data_loader(self, batch_size, n_repeat, num_workers = 8):
        print('len {} dataset {}'.format(self.phase, len(self)))
        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong

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
    

class Maze_AEDataset(Maze_AgentCentric_StateConditioned):
    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, *args, **kwargs):
        super().__init__(data_dir, data_conf, phase, resolution, shuffle, dataset_size, *args, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return dict(  states = sample['bianry_image']  )
