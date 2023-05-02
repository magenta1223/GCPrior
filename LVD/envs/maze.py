from contextlib import contextmanager

# from d4rl.pointmaze import MazeEnv
import gym
import mujoco_py
import numpy as np


# from  .maze_layout import rand_layout
from ..contrib.simpl.env.maze_layout import rand_layout
from ..contrib.simpl.env.maze import MazeEnv, MazeTask, AgentCentricMazeEnv

import torch
import random

init_loc_noise = 0.1
complete_threshold = 1.0

color_dict = {
    "wall" : np.array([0.87, 0.62, 0.38]),
    "agent" : np.array([0.32, 0.65, 0.32]),
    "ground_color1" : np.array([0.2, 0.3, 0.4]),
    "ground_color2" : np.array([0.1, 0.2, 0.3]),
}

WALL = np.full((32, 32, 3), color_dict['wall'])
G1 = np.full((32, 32, 3), color_dict['ground_color1'])
G2 = np.full((32, 32, 3), color_dict['ground_color2'])


class MazeTask_Custom:
    def __init__(self, init_loc, goal_loc):
        self.init_loc = np.array(init_loc, dtype=np.float32)
        self.goal_loc = np.array(goal_loc, dtype=np.float32)

    def __repr__(self):
        return f'MTMazeTask(start:{self.init_loc}+-{init_loc_noise}, end: {self.goal_loc})'

class Maze_GC(MazeEnv):

    def __init__(self, size, seed, reward_type, done_on_completed, visual_encoder = None):
        if reward_type not in self.reward_types:
            raise f'reward_type should be one of {self.reward_types}, but {reward_type} is given'
        self.visual_encoder = visual_encoder
        super().__init__(size, seed, reward_type, done_on_completed)
        self.agent_centric_res = 32
        self.render_width = 32
        self.render_height = 32
        # self.maze_size = size
        # self.maze_spec = rand_layout(size=size, seed=seed)
        
        # for initialization
        self.task = MazeTask_Custom([0, 0], [0, 0])
        # self.done_on_completed = False
        # self.task = None
        # self.done_on_completed = done_on_completed
        # gym.utils.EzPickle.__init__(self, size, seed, reward_type, done_on_completed)
        # print(self.reset_locations)
        # 이 중에 적당히 멀리있는 goal을 뽑아야 함.
        # size=20이면 10,10에서 출발 L1거리가 최대 20이니, 10이상인걸로 
        # L1 distance가 size // 2 이상인 그런 지점을 거르고
        # 거리순으로 정렬, 3개마다 샘플링

        # print(np.array(self.reset_locations))

        # center = np.array([self.maze_size // 2, self.maze_size // 2])
        
        # dists = np.abs(goal_candidates-center).sum(axis = 1)
        # # goal_candidates = goal_candidates[ (self.maze_size // 10) * 6 > dists > (self.maze_size // 10) * 3]

        # cond1 = dists < (self.maze_size // 10) * 6
        # cond2 = dists > (self.maze_size // 10) * 3

        # goal_candidates = goal_candidates[np.where( cond1 & cond2 )[0]]
        
        # indices = random.sample(range(len(goal_candidates)), k = 10)

        # print(goal_candidates[indices])
        # assert 1==0, ""
        

        # Goal 후보를 찾아야.. 

        self.viewer = self._get_viewer(mode = "rgb_array")
        self.viewer_setup()

    def set_visual_encoder(self, visual_encoder):
        self.visual_encoder = visual_encoder


    @contextmanager
    def set_task(self, task):
        if type(task) != MazeTask_Custom:
            raise TypeError(f'task should be MazeTask but {type(task)} is given')

        prev_task = self.task
        self.task = task
        self.set_target(task.goal_loc)
        yield
        self.task = prev_task

    def reset_model(self):
        if self.task is None:
            raise RuntimeError('task is not set')
        init_loc = self.task.init_loc
        qpos = init_loc + self.np_random.uniform(low=-init_loc_noise, high=init_loc_noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        if self.visual_encoder is not None:
            with self.agent_centric_render():
                img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device) / 255


            # img = self.render(mode = "rgb_array")
            walls = np.abs(img - WALL).mean(axis=-1)
            grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
            img = np.stack((walls, grounds), axis=-1).argmax(axis=-1)
            img = torch.from_numpy(img).view(-1).unsqueeze(0).cuda().float()


            visual_embedidng = self.visual_encoder(img).detach().cpu()[0].numpy()
            states = np.concatenate((np.tile(self._get_obs(), 10), visual_embedidng, np.array(self._target) ), axis = 0 )
            return states

        else:
            return np.concatenate((self._get_obs(), np.array(self._target)), axis = 0)

    @contextmanager
    def agent_centric_render(self):
        prev_type = self.viewer.cam.type
        prev_distance = self.viewer.cam.distance
        
        self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
        self.viewer.cam.distance = 5.0
        
        yield
        
        self.viewer.cam.type = prev_type
        self.viewer.cam.distance = prev_distance
        


    def step(self, action):



        if self.task is None:
            raise RuntimeError('task is not set')
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()

        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()

        goal_dist = np.linalg.norm(ob[0:2] - self._target)
        completed = (goal_dist <= complete_threshold)
        done = self.done_on_completed and completed

        if self.visual_encoder is not None:
            with self.agent_centric_render():
                img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device)


            walls = np.abs(img - WALL).mean(axis=-1)
            grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
            img = np.stack((walls, grounds), axis=-1).argmax(axis=-1)
            img = torch.from_numpy(img).view(-1).unsqueeze(0).cuda().float()
            # img = self.render(mode = "rgb_array")

            visual_embedidng = self.visual_encoder(img).detach().cpu()[0].numpy()
            ob = np.concatenate((np.tile(ob, 10), visual_embedidng, np.array(self._target) ), axis = 0 )


        else:
            ob =  np.concatenate((ob, np.array(self._target)), axis = 0)
        
        
        if self.reward_type == 'sparse':
            reward = float(completed)
        elif self.reward_type == 'dense':
            reward = np.exp(-goal_dist)
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)




        return ob, reward, done, {}

            
    
# maze_config = {
#     'size':20,
#     'seed': 0,
#     'reward_type':'sparse',
#     'done_on_completed': True,
    # 'visual_encoder' : None
# }

maze_config = {
    'size':40,
    'seed': 0,
    'reward_type':'sparse',
    'done_on_completed': True,
    'visual_encoder' : None
}





# 40

MAZE_TASKS = np.array([
    # 기존 task들은 잘 됨. 
    # [12,13],
    # [8,20],
    # [14,7],
    # [18,15],
    # [14,12],
    # [1,11],
    # [15,15],
    # [20,6],
    # [18,4],
    # [6,14],
    # [5,10],
    # [6,15],
    # [14,9],
    # [3,14],
    # [12,19],
    # [12,4],
    # [2,11],
    # [19,13],
    # [8,18],
    # [18,5]


    # maze20 hard task
    # [16,  9], 
    # [ 2, 15],
    # [ 1,  8],
    # [18, 13],
    # [ 5, 15],
    # [ 5,  3],
    # [ 4, 18],
    # [ 3,  3],
    # [18, 11],
    # [ 2, 12],



    # MAZE 40
    [[20, 20], [10, 25]],
    [[20, 20], [ 4, 23]],
    [[20, 20], [21, 34]],
    [[20, 20], [19, 33]],
    [[20, 20], [18,  5]],
    [[20, 20], [12, 12]],
    [[20, 20], [10,  8]],
    [[20, 20], [ 9, 11]],
    [[20, 20], [31, 26]],
    [[20, 20], [ 5, 14]],
    
    # [12,13],
    # [8,20],
    # [14,7],
    # [18,15],
    # [14,12],
    # [1,11],
    # [15,15],
    # [20,6],
    # [18,4],
    # [6,14],
    # [5,10],
    # [6,15],
    # [14,9],
    # [3,14],
    # [12,19],
    # [12,4],
    # [2,11],
    # [19,13],
    # [8,18],
    # [18,5]
    ])



# MAZE_TASKS = np.array([
#     # 기존 task들은 잘 됨. 
#     # [12,13],
#     # [8,20],
#     # [14,7],
#     # [18,15],
#     # [14,12],
#     # [1,11],
#     # [15,15],
#     # [20,6],
#     # [18,4],
#     # [6,14],
#     # [5,10],
#     # [6,15],
#     # [14,9],
#     # [3,14],
#     # [12,19],
#     # [12,4],
#     # [2,11],
#     # [19,13],
#     # [8,18],
#     # [18,5]


#     # maze20 hard task
#     [[10, 10], [16,  9]], 
#     [[10, 10], [ 2, 15]],
#     [[19, 13], [ 1,  8]],
#     [[10, 10], [18, 13]],
#     [[10, 10], [ 5, 15]],
#     [[10, 10], [ 5,  3]],
#     [[10, 10], [ 4, 18]],
#     [[10, 10], [ 3,  3]],
#     [[10, 10], [18, 11]],
#     [[10, 10], [ 2, 12]],

# ])



