import numpy as np
from contextlib import contextmanager
from carla.carla_env.base import BaseCarlaEnvironment


class CARLA_Task:
    def __repr__(self):
        return "-".join([ t[0].upper() for t in self.subtasks])


class CARLA_GC(BaseCarlaEnvironment):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, *args, **kwargs):
        # self.TASK_ELEMENTS = ['top burner']  # for initialization
        super().__init__(*args, **kwargs)
        

        self.task = None
        self.TASK_ELEMENTS = None
    
    # def _get_task_goal(self, task=None):
    #     if task is None:
    #         task = self.TASK_ELEMENTS
    #     new_goal = np.zeros_like(self.goal)
    #     for element in task:
    #         element_idx = OBS_ELEMENT_INDICES[element]
    #         element_goal = OBS_ELEMENT_GOALS[element]
    #         new_goal[element_idx] = element_goal

    #     return new_goal
    
    @contextmanager
    def set_task(self, task):
        if type(task) != CARLA_GC:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks
        
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements


    def reset(self):
        return super().reset()


    def step(self, action = None, traffic_light_color = ""):
        
        rewards = []

        next_obs, done, info = None, None, None
        for _ in range(self.config.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(
                action, traffic_light_color
            )
            rewards.append(reward)

            if done:
                break

        if next_obs is None or done is None or info is None:
            raise ValueError("frame_skip >= 1")
        return next_obs, np.mean(rewards), done, info

    def _simulator_step(self, action = None, traffic_light_color = None):
        # goal 과 얼마나 가까운지?
        next_obs = self.sim.step(action)
        reward = self.goal_reaching_reward()
        done = info = None
        return  next_obs, reward, done, info


    

# for simpl
meta_train_tasks = np.array([
    # [5,6,0,3],
    # [5,0,1,3],
    # [5,1,2,4],
    # [6,0,2,4],
    # [5,0,4,1],
    # [6,1,2,3],
    # [5,6,3,0],
    # [6,2,3,0],
    # [5,6,0,1],
    # [5,6,3,4],
    # [5,0,3,1],
    # [6,0,2,1],
    # [5,6,1,2],
    # [5,6,2,4],
    # [5,0,2,3],
    # [6,0,1,2],
    # [5,2,3,4],
    # [5,0,1,4],
    # [6,0,3,4],
    # [0,1,3,2],
    # [5,6,2,3],
    # [6,0,1,4],
    # [0,1,2,3]
])


tasks = np.array([
    # # Well-aligned, Not missing
    # [5,6,0,3], # MKBS
    # [5,0,1,3],  # MBTS
    # # Mis-aligned, Not missing
    # [6,0,2,4],  # KBLH
    # [5,1,2,4],  # MTLH
    # # Well-algined, Missing
    # [5,6,0,1],
    # [5,6,0,2],
    # # Mis-algined, Missing 
    # [6,1,2,4],  # KTLH
    # [5,1,3,4],  # MTSH
    # # [5,6,0,2],  # MKBL
])


# kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])
# KITCHEN_TASKS = kitchen_subtasks[tasks]
# KITCHEN_META_TASKS = kitchen_subtasks[meta_train_tasks]
