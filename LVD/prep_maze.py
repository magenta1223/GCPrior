import pickle
import h5py
from easydict import EasyDict as edict
import numpy as np
from glob import glob
# from tqdm import notebook

from tqdm import tqdm
import os

def parse_h5(file_path):
    f = h5py.File(file_path)
    # return {k : np.array(f.get("traj0").get(k))   for k in list(f.get("traj0").keys()) if k != "images"}
    traj = f.get("traj0")
    return edict( 
        states = np.array(traj.get("states")),
        actions = np.array(traj.get("actions")),
        agent_centric_view = np.array(traj.get("images")),
    )

def main():
    os.makedirs("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/", exist_ok= True)

    for i in range(1, 18):
        os.makedirs(f"/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/batch_{i}", exist_ok= True)


    file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze/**/*.h5")


    # file_paths = glob("/home/magenta1223/skill-based/SiMPL/proposed/LVD/data/maze/maze_prep/**/*.h5")


    color_dict = {
        "wall" : np.array([0.87, 0.62, 0.38]),
        "agent" : np.array([0.32, 0.65, 0.32]),
        "ground_color1" : np.array([0.2, 0.3, 0.4]),
        "ground_color2" : np.array([0.1, 0.2, 0.3]),
    }

    wall = np.full((32, 32, 3), color_dict['wall'])
    ground1 = np.full((32, 32, 3), color_dict['ground_color1'])
    ground2 = np.full((32, 32, 3), color_dict['ground_color2'])



    for file_path in tqdm(file_paths):        
        prep_path = file_path.replace("maze/maze/", "maze/maze_prep/")
    
        os.rename(prep_path, prep_path.replace(".h5", ".pkl"))

        # if os.path.exists(prep_path):
        #     continue

        # seq = parse_h5(file_path)
        # img = seq['agent_centric_view']

        # # 색상 비교해서 가장 가까운 색상으로 변경. 

        # walls = np.abs(img - wall).mean(axis = -1)
        # grounds = np.minimum(np.abs(img - ground1).mean(axis = -1), np.abs(img - ground2).mean(axis = -1))
        # seq['agent_centric_view'] = np.stack((walls, grounds), axis = -1).argmax(axis = -1)
        


        # with open(prep_path, mode = "wb") as f:
        #     pickle.dump(seq, f)



if __name__ == "__main__":
    main()