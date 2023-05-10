import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env

mujoco_env.USE_DM_CONTROL = False
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']

from ..contrib.simpl.env.kitchen import KitchenTask, KitchenEnv

class KitchenEnv_GC(KitchenEnv):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = ['top burner']  # for initialization
        super().__init__(*args, **kwargs)
        

        self.task = None
        self.TASK_ELEMENTS = None
    
    def _get_task_goal(self, task=None):
        if task is None:
            task = self.TASK_ELEMENTS
        new_goal = np.zeros_like(self.goal)
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal
    

# simpl meta train task
# tasks = np.array(
#     [[5,6,0,3],
#     [5,0,1,3],
#     [5,1,2,4],
#     [6,0,2,4],
#     [5,0,4,1],
#     [6,1,2,3],
#     [5,6,3,0],
#     [6,2,3,0],
#     [5,6,0,1],
#     [5,6,3,4],
#     [5,0,3,1],
#     [6,0,2,1],
#     [5,6,1,2],
#     [5,6,2,4],
#     [5,0,2,3],
#     [6,0,1,2],
#     [5,2,3,4],
#     [5,0,1,4],
#     [6,0,3,4],
#     [0,1,3,2],
#     [5,6,2,3],
#     [6,0,1,4],
#     [0,1,2,3]]
# )


tasks = np.array(
    
    [
        # Well-aligned, Not missing
        [5,6,0,3], # MKBS
        [5,0,1,3],  # MBTS
        # Mis-aligned, Not missing
        [6,0,2,4],  # KBLH
        [5,1,2,4],  # MTLH
        # Well-algined, Missing
        [5,0,3,4],  # MBSH
        [6,0,2,3],  # KBLS
        # Mis-algined, Missing 
        [6,1,2,4],  # KTLH
        [5,1,3,4],  # MTSH
        # [5,6,0,2],  # MKBL
    ]
)


kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])
KITCHEN_TASKS = kitchen_subtasks[tasks]
