import os 
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
from LVD.utils import state_process_carla
import argparse

SENSOR_SCALE = {
    "control": (1, slice(0,3)),
    "acceleration": (10, slice(0,3)),
    "angular_velocity": (10, slice(0,3)),
    "location": (100, slice(0,2)),
    "rotation": (10, slice(1,2)),
    "forward_vector": (1, slice(0,0)),
    "velocity": (10, slice(0,3)),
    "target_location": (100, slice(0,0)),
}
SENSORS = ["control", "acceleration", "velocity", "angular_velocity", "location", "rotation", "forward_vector", "target_location"]

def get_creation_time(file_path):
    stat = os.stat(file_path)
    creation_time = stat.st_ctime
    return creation_time

def get_latest_version():
    datasets = glob("./carla_data/*")
    latest_version = np.argmax([ get_creation_time(file_path) for file_path in datasets])
    return datasets[latest_version]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action= "store_true", default= False)
        
    args = parser.parse_args()

        

    dataset_dir = get_latest_version()
    print(f"Loading dataset from {dataset_dir}")
    trajectories = glob(f"{dataset_dir}/*.pkl")
    dataset = []
    

    for trajectory in tqdm(trajectories):
        with open(trajectory, mode = "rb") as f:
            traj = pickle.load(f)

        obs = traj['observations']['sensor']
        actions = traj['actions']
        
        # obs = np.concatenate((obs[:, :14], obs[:, 15:-6]), axis = -1)
        
        obs = state_process_carla(obs, args.normalize)

        # obs_dict = { key : obs[:, i*3 : (i+1)*3 ]   for i, key in enumerate(SENSORS)}
        # prep_obs_dict = {}
    
        # for k, (scale, idx) in SENSOR_SCALE.items():
        #     # raw_obs = obs_dict[k][idx] / scale
        #     # if raw_obs.
        #     prep_obs_dict[k] = obs_dict[k][:, idx] / scale
        #     # contorl : all
        #     # acceleration : all
        #     # vel : all
        #     # angular vel : all
        #     # loc : all
        #     # rot : only y
            

        # # xy = obs[:, 12:14] 
        # # obs = np.concatenate((obs[:, :12], xy, obs[:, 15:-6]), axis = -1)


        # obs = np.concatenate( [v for k, v in prep_obs_dict.items() if v.any()], axis = 1)


        throttle_or_brake  = actions[:,0] + actions[:,1]
        steer = actions[:, 2]
        actions_intregrated = np.stack((throttle_or_brake, steer), axis = -1)


        dataset.append(
            dict(
                obs = obs.astype(np.float32),
                actions = actions_intregrated.astype(np.float32)
            )
        )

    if args.normalize:
        path = "./LVD/data/carla/carla_dataset_normalized.pkl"
    else:
        path = "./LVD/data/carla/carla_dataset.pkl"


    with open(path, mode = "wb") as f:
        pickle.dump(dataset, f)

    print(f"CARLA DATASET from {dataset_dir} -> {path}")