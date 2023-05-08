import pickle
import h5py
from easydict import EasyDict as edict
import numpy as np
from glob import glob
# from tqdm import notebook

from tqdm import tqdm
import os
import multiprocessing as mp
from tqdm import tqdm

import argparse

import torch
import sys

sys.path.append("/home/magenta1223/skill-based/SiMPL/proposed")

def parse_h5(file_path):
    f = h5py.File(file_path)
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
    traj = f.get("traj0")
    return edict( 
        states = np.array(traj.get("states")),
        actions = np.array(traj.get("actions")),
        agent_centric_view = np.array(traj.get("images")) / 255,
    )


def process_file(file_path, prep_path_prefix, wall, ground1, ground2):
    seq = parse_h5(file_path)
    img = seq['agent_centric_view']

    walls = np.abs(img - wall).mean(axis=-1)
    grounds = np.minimum(np.abs(img - ground1).mean(axis=-1), np.abs(img - ground2).mean(axis=-1))
    seq['agent_centric_view'] = np.stack((walls, grounds), axis=-1).argmax(axis=-1)

    split = file_path.split("/")
    batch, fname = split[-2], split[-1]
    new_path = f'{prep_path_prefix}{batch}_{fname}'

    with h5py.File(new_path, 'w') as f:
        f.create_dataset("states", data=seq['states'])
        f.create_dataset("actions", data=seq['actions'])
        f.create_dataset("images", data=seq['agent_centric_view'])


def main(args):

    prefix = args.prefix
    load = torch.load("./weights/maze/wae/log21_end.bin")['model'].eval()
    visual_encoder = load.state_encoder 
    

    new_path_prefix = f"{prefix}/maze_prep_visual_feature/"
    os.makedirs(new_path_prefix, exist_ok= True)
    file_paths = glob(f"{prefix}/maze_prep/*.h5")


    for file_path in tqdm(file_paths):
        f = h5py.File(file_path)
        # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
        states = np.array(f.get("states"))
        actions = np.array(f.get("actions"))
        images = np.array(f.get("images"))


        visual_feature = visual_encoder(torch.from_numpy(images).view(-1, 1024).float().cuda()).detach().cpu().numpy()

        fname = file_path.split("/")[-1]
        new_path = f'{new_path_prefix}{fname}'

        with h5py.File(new_path, 'w') as f:
            f.create_dataset("states", data=states)
            f.create_dataset("actions", data=actions)
            f.create_dataset("visual_feature", data=visual_feature)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default = "/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze")
    
    args = parser.parse_args()
    main(args)

