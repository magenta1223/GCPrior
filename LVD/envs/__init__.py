from .kitchen import *
from .carla import *
from .maze import *



ENV_TASK = {
    "kitchen" : {
        "env_cls" : KitchenEnv_GC,
        "task_cls" : KitchenTask_GC,
        "tasks" : KITCHEN_TASKS,
        "cfg" : None 
    },
    "carla" : {
        "env_cls" : CARLA_GC,
        "task_cls" : CARLA_Task,
        "tasks"  : CARLA_TASKS,
        "cfg" : carla_config 
    },
    "maze" : {
        "env_cls" : Maze_GC,
        "task_cls" : MazeTask_Custom, 
        "tasks"  : MAZE_TASKS,
        "cfg" : maze_config 
    },
}