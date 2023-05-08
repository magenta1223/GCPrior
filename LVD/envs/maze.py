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
from copy import deepcopy

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

    def __init__(self, size, seed, reward_type, done_on_completed, relative = False, visual_encoder = None):
        if reward_type not in self.reward_types:
            raise f'reward_type should be one of {self.reward_types}, but {reward_type} is given'
        # self.viewer_setup()
        self.size = size
        self.relative = relative
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
        self._viewers = {}
        self.viewer = self._get_viewer(mode = "rgb_array")
        # self.viewer = self._get_viewer(mode = "rgb_array")
        # self.viewer_setup()




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

        ob = deepcopy(self._get_obs())
        target = deepcopy(self._target)
        if self.relative:
            ob[:2] -= self.task.init_loc
            target -= self.task.init_loc

        ob = np.concatenate((ob, target), axis = 0)
       
        return ob
        # with self.agent_centric_render():
        #     img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device) / 255
        #     walls = np.abs(img - WALL).mean(axis=-1)
        #     grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
        #     img = np.stack((walls, grounds), axis=-1).argmax(axis=-1)


        # return np.concatenate((self._get_obs(), img.reshape(-1), np.array(self._target)), axis = 0)

    @contextmanager
    def agent_centric_render(self):
        prev_type = self.viewer.cam.type
        prev_distance = self.viewer.cam.distance
        
        self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
        self.viewer.cam.distance = 5.0
        
        yield
        
        self.viewer.cam.type = prev_type
        self.viewer.cam.distance = prev_distance
        


    def step(self, action, init = True):



        if self.task is None:
            raise RuntimeError('task is not set')
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()

        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = deepcopy(self._get_obs())

        goal_dist = np.linalg.norm(ob[0:2] - self._target)
        completed = (goal_dist <= complete_threshold)
        done = self.done_on_completed and completed
        
        # if not init:
        #     with self.agent_centric_render():
        #         img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device) / 255
        #         walls = np.abs(img - WALL).mean(axis=-1)
        #         grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
        #         img = np.stack((walls, grounds), axis=-1).argmax(axis=-1)
        #     ob = np.concatenate((self._get_obs(), img.reshape(-1), np.array(self._target)), axis = 0)
        # else:
        #     ob = np.concatenate((self._get_obs(), np.array(self._target)), axis = 0)


        target = deepcopy(self._target)
        if self.relative:
            ob[:2] -= self.task.init_loc
            target -= self.task.init_loc
        ob = np.concatenate((ob, target), axis = 0)

        # ob = np.concatenate((self._get_obs(), np.array(self._target)), axis = 0)

        # ob -= np.array(self.task.init_loc)

    
        
        if self.reward_type == 'sparse':
            reward = float(completed) * 100
        elif self.reward_type == 'dense':
            reward = np.exp(-goal_dist)
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)




        return ob, reward, done, {}

    def render(self, mode = "rgb_array"):
        if mode == "agent_centric":
            with self.agent_centric_render():
                img = self.sim.render(self.agent_centric_res, self.agent_centric_res, device_id=self.render_device) / 255
                walls = np.abs(img - WALL).mean(axis=-1)
                grounds = np.minimum(np.abs(img - G1).mean(axis=-1), np.abs(img - G2).mean(axis=-1))
                # img = np.stack((walls, grounds), axis=-1).argmax(axis=-1)
            return img * 255

        else:
            return super().render(mode)


    
maze_config = {
    'size':40,
    'seed': 0,
    'reward_type':'sparse',
    'done_on_completed': True,
    'visual_encoder' : None
}

# maze_config = {
#     'size':40,
#     'seed': 0,
#     'reward_type':'sparse',
#     'done_on_completed': True,
#     'visual_encoder' : None
# }





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

    # MAZE 20 HARD TASK
    # [[10, 10], [16,  9]], 
    # [[10, 10], [ 2, 15]],
    # [[19, 13], [ 1,  8]],
    # [[10, 10], [18, 13]],
    # [[10, 10], [ 5, 15]],
    # [[10, 10], [ 5,  3]],
    # [[10, 10], [ 4, 18]],
    # [[10, 10], [ 3,  3]],
    # [[10, 10], [18, 11]],
    # [[10, 10], [ 2, 12]],



    # MAZE 40
    # [[20, 22], [10, 25]],
    # [[20, 22], [ 4, 23]],
    # [[20, 22], [21, 34]],
    # [[20, 22], [19, 33]],
    # # [[20, 22], [18,  5]],
    # [[20, 22], [12, 12]],
    # [[20, 22], [10,  8]],
    # [[20, 22], [ 9, 11]],
    # [[20, 22], [31, 26]],
    # [[20, 22], [ 5, 14]],
    
    # [[10, 24], [10, 25]], # 바로 위임
    # [[10, 24], [ 4, 23]], # 바로 옆임 ? 
    # [[10, 24], [21, 34]],
    # [[10, 24], [19, 33]],
    [[10, 24], [18,  5]],
    [[10, 24], [12, 12]],
    [[10, 24], [10,  8]],
    [[10, 24], [ 9, 11]],
    [[10, 24], [31, 26]],
    [[10, 24], [ 5, 14]],

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



