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


def main():
    prep_path_prefix = "/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/"
    os.makedirs(prep_path_prefix, exist_ok= True)
    file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze/**/*.h5")

    color_dict = {
        "wall" : np.array([0.87, 0.62, 0.38]),
        "agent" : np.array([0.32, 0.65, 0.32]),
        "ground_color1" : np.array([0.2, 0.3, 0.4]),
        "ground_color2" : np.array([0.1, 0.2, 0.3]),
    }

    wall = np.full((32, 32, 3), color_dict['wall'])
    ground1 = np.full((32, 32, 3), color_dict['ground_color1'])
    ground2 = np.full((32, 32, 3), color_dict['ground_color2'])


    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    results = []

    for i, file_path in enumerate(tqdm(file_paths, desc="Processing files")):
        results.append(pool.apply_async(process_file, args=(file_path, prep_path_prefix, wall, ground1, ground2)))

    for result in tqdm(results, desc="Waiting for results", total=len(results)):
        result.wait()


if __name__ == '__main__':
    main()
