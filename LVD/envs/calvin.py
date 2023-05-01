from ..contrib.calvin.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv

import hydra
from gym import spaces
from LVD.contrib.calvin.calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
import numpy as np

from contextlib import contextmanager



from hydra import initialize, compose
import hydra
from copy import deepcopy


CALVIN_EL_INDICES = {
    'move_slider_left': np.array([0]),
    'open_drawer': np.array([1]),
    'turn_on_lightbulb': np.array([3, 4]),
    'turn_on_led': np.array([5]),
    }

CALVIN_EL_GOALS = {
    'move_slider_left': np.array([0.15]),
    'open_drawer': np.array([0.12]),
    'turn_on_lightbulb': np.array([0.08, 1]),
    'turn_on_led': np.array([1]),
    }

CALVIN_BONUS_THRESH = 0.1

tasks = np.array([
    [0,1,2,3],
    [0,1,3,2]
])

calvin_subtasks = np.array(['open_drawer', 'turn_on_lightbulb', 'move_slider_left', 'turn_on_led'])

CALVIN_TASKS = calvin_subtasks[tasks]


with initialize(config_path="../contrib/calvin/calvin_env/conf/"):
    cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
    cfg.env["use_egl"] = False
    cfg.env["show_gui"] = False
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True

cfg_calvin = {**cfg.env}
cfg_calvin["tasks"] = cfg.tasks
cfg_calvin.pop('_target_', None)
cfg_calvin.pop('_recursive_', None)


# env = CALVIN_GC_PlayTableSim_Env(**cfg_env)

class CALVIN_Task:
    def __init__(self, subtasks):
        # for subtask in subtasks:
            # if subtask not in all_tasks:
            #     raise ValueError(f'{subtask} is not valid subtask')
        self.subtasks = subtasks

    def __repr__(self):
        return f"MTCALVIN_Task({' -> '.join(self.subtasks)})"



class CALVIN_GC_PlayTableSim_Env(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 **kwargs):
        super(CALVIN_GC_PlayTableSim_Env, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,))
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)
        # task check util function임. 실제로 할 task와는 관계 없음. 
        
        self.task = CALVIN_Task(['move_slider_left'])
        self.TASK_ELEMENTS = self.task.subtasks
        self.goal_state = np.zeros(6)

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return obs
    

    @contextmanager
    def set_task(self, task):
        if type(task) != CALVIN_Task:
            raise TypeError(f'task should be CALVIN_Task but {type(task)} is given')

        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks

        goal_state = np.zeros(6)

        for subtask in self.TASK_ELEMENTS:
            goal_state[CALVIN_EL_INDICES[subtask]] = CALVIN_EL_GOALS[subtask]

        self.goal_state = goal_state
        
        yield
        self.goal_state = np.zeros(6)
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements


    # def _get_task_goal(self, task=None):
    #     if task is None:
    #         task = self.TASK_ELEMENTS
    #     new_goal = np.zeros_like(self.goal)
    #     for element in task:
    #         element_idx = OBS_ELEMENT_INDICES[element]
    #         element_goal = OBS_ELEMENT_GOALS[element]
    #         new_goal[element_idx] = element_goal
    #     return new_goal

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()#[:6]
        # robot obs는 전부 다 하고
        # scene obs는 12~17만 
        return np.concatenate((robot_obs, scene_obs, self.goal_state), axis = -1)



    def _success(self):
        """ Returns a boolean indicating if the task was performed correctly """
        current_info = self.get_info()
        # task_filter = ["move_slider_left"]
        # task_filter = self.TASK_ELEMENTS
        reward = 0
        done = False
        completed = []
        for subtask in self.tasks_to_complete:
            task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, [subtask])
            if subtask in task_info:
                reward += 1
                completed.append(subtask)
            else:
                break

        for completed_subtask in completed:
            self.tasks_to_complete.remove(completed_subtask) 

        if reward == 4:
            done = True         

        return reward, done

    def _reward(self):
        """ Returns the reward function that will be used 
        for the RL algorithm """
        reward, done = self._success()
        info = {
            'reward': reward,
            'done' : done,
            'success' : done
        }
        return reward, done, info

    # def _termination(self):
    #     """ Indicates if the robot has reached a terminal state """
    #     success = self._success()
    #     done = success
    #     d_info = {'success': success}        
    #     return done, d_info

    def step(self, action):
        """ Performing a relative action in the environment
            input:
                action: 7 tuple containing
                        Position x, y, z. 
                        Angle in rad x, y, z. 
                        Gripper action
                        each value in range (-1, 1)

                        OR
                        8 tuple containing
                        Relative Joint angles j1 - j7 (in rad)
                        Gripper action
            output:
                observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1

        # for using actions in joint space
        if len(env_action) == 8:
            env_action = {"action": env_action, "type": "joint_rel"}

        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        obs = self.get_obs()
        info = self.get_info()
        reward, done, _info = self._reward()
        info.update(_info)
        return obs, reward, done, info
    
    def set_state(self, robot_state = None, scene_state = None):
        if robot_state is not None:
            self.robot.reset(robot_state = robot_state)
        if scene_state is not None:
            diff = 24 - scene_state.shape[0]
            scene_state = np.concatenate((scene_state, np.zeros(diff)), axis = 0)
            self.scene.reset(scene_obs= scene_state)
     





# Scene obs 
# - 0 : slider position 음수면 오른쪽
# - 1 : 서랍. 음수면 넣음
# - 2 : 검은 버튼 위치. 양수면 눌림
# - 3 : 전구 켜는 레버
# - 4 : 전구 상태. discrete. 0 or 1
# - 5 : 버튼누르면 켜지는 초록 불.  discrete. 0 or 1
# - 6, 7, 8, 9, 10, 11 : red cube xyz, rad xyz
# - 12, 13, 14, 15, 16, 17 : red cube xyz, rad xyz
# - 18, 19, 20, 21, 22, 23 : red cube xyz, rad xyz


# drawer는 무조건 닫혀있고
# slider는 무조건 오른쪽임.
# 불은 꺼져있고 

# 이걸 기준으로 goal state를 짜면 됨. 
# slider는 +0.15
# drawer +0.12
# light_bulb 0, 1
# led 0, 1


# 

# 자.. goal은 어떻지? 버튼도 눌렀고, 불도 켜져 있는 상태임. 

# 불켜기는 시작 state가 0이고 끝이 1인 경우

