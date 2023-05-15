import os 
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm

def get_creation_time(file_path):
    stat = os.stat(file_path)
    creation_time = stat.st_ctime
    return creation_time

def get_latest_version():
    datasets = glob("./carla_data/*")
    latest_version = np.argmax([ get_creation_time(file_path) for file_path in datasets])
    return datasets[latest_version]

if __name__ == "__main__":
    dataset_dir = get_latest_version()
    print(f"Loading dataset from {dataset_dir}")
    trajectories = glob(f"{dataset_dir}/*.pkl")
    dataset = []
    for trajectory in tqdm(trajectories):
        with open(trajectory, mode = "rb") as f:
            traj = pickle.load(f)

        obs = traj['observations']['sensor']
        actions = traj['actions']



        obs_z_removed = obs[:, :3]
        for i in range(1, 7):
            obs_z_removed = np.concatenate((obs_z_removed, obs[:, 3*i : 3*i + 2]), axis = -1).astype(np.float32)

        throttle_or_brake  = actions[:,0] + actions[:,1]
        steer = actions[:, 2]
        actions = np.stack((throttle_or_brake, steer), axis = -1)


        dataset.append(
            dict(
                obs = obs_z_removed.astype(np.float32),
                actions = actions.astype(np.float32)
            )
        )

    with open("./LVD/data/carla/carla_dataset.pkl", mode = "wb") as f:
        pickle.dump(dataset, f)

    print(f"CARLA DATASET from {dataset_dir} -> ./LVD/data/carla/carla_dataset.pkl")