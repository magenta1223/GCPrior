from copy import deepcopy
from matplotlib.style import context
from proposed.contrib.simpl.env.kitchen import KitchenEnv, KitchenTask
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, BONUS_THRESH, OBS_ELEMENT_GOALS
import numpy as np
import gym
from d4rl.kitchen.adept_envs import mujoco_env


from dm_control.mujoco import engine
import mujoco_py
import numpy as np
from contextlib import contextmanager

mujoco_env.USE_DM_CONTROL = False
ALL_TASKS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']



# goal relabeling

class KitchenEnvSingleTask(KitchenEnv):
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, *args, **kwargs):
        self.dense = False
        self.negative_rwd = 0
        self.do_render = False
        super().__init__(*args, **kwargs)



    def set_nrwd(self, value):
        return self.negative_rwd

    def non_target_reward(self, target_task, obs_dict):
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        # next_goal = self._get_task_goal(task=self.TASK_ELEMENTS)
        next_goal = self._get_task_goal(task=ALL_TASKS) # for penalize overperforming
        idx_offset = len(next_q_obs) # ? 

        nonTarget_tasks = deepcopy(ALL_TASKS)
        nonTarget_tasks.remove(target_task)
        reward = 0
        for element in nonTarget_tasks:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx]) # 
            _complete = distance < BONUS_THRESH # 거리가 treshold미만이면 완료.        
            if _complete:
                reward -= self.negative_rwd
        return reward


    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')
        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements

    @contextmanager
    def step_render(self):
        prev_render = self.do_render
        self.do_render = True
        yield
        self.do_render= prev_render


    def set_goal(self):
        self.goal = self._get_task_goal(task = self.TASK_ELEMENTS)

    def reset_model(self):
        self.tasks_to_complete = set(self.TASK_ELEMENTS)
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        # self.goal = self._get_task_goal()  #sample a new goal on reset
        self.set_goal()
        self.tasks_to_complete = list(self.TASK_ELEMENTS)

        return self._get_obs()

    def compute_reward(self, obs_dict):
        reward = 0
        
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        # next_goal = self._get_task_goal(task=self.TASK_ELEMENTS)
        next_goal = self._get_task_goal(task=ALL_TASKS) # for penalize overperforming
        idx_offset = len(next_q_obs) # ? 

        target_task = list(self.tasks_to_complete)[0]
        
        
        element_idx = OBS_ELEMENT_INDICES[target_task]
        distance = np.linalg.norm(
            next_obj_obs[..., element_idx - idx_offset] -
            next_goal[element_idx]) # 
        complete = distance < BONUS_THRESH # 거리가 treshold미만이면 완료.

        if complete:
            reward += 1
            done = True
        
            # nonTarget_tasks = deepcopy(ALL_TASKS)
            # nonTarget_tasks.remove(target_task)
            # for element in nonTarget_tasks:
            #     element_idx = OBS_ELEMENT_INDICES[element]
            #     distance = np.linalg.norm(
            #         next_obj_obs[..., element_idx - idx_offset] -
            #         next_goal[element_idx]) # 
            #     _complete = distance < BONUS_THRESH # 거리가 treshold미만이면 완료.        
            #     if _complete:
            #         nrwd -= self.negative_rwd
        
        else:
            done = False


        return reward, done

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp

        self.robot.step(self, a, step_duration=self.skip * self.model.opt.timestep)
        obs = self._get_obs()
        reward, done = self.compute_reward(self.obs_dict)
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'img' : self.render(mode = "rgb_array") if self.do_render else None
        }
        return obs, reward, done, env_info


gym.register(
    id='kitchen-single-v0',
    entry_point='proposed.envs.base:KitchenEnvSingleTask'
)