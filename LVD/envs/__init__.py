from .kitchen import *
from .carla import *
from .maze import *



ENV_TASK = {
    "kitchen" : {
        "env_cls" : KitchenEnv_GC,
        "task_cls" : KitchenTask,
        "tasks" : KITCHEN_TASKS,
        "cfg" : None 
    },
    "carla" : {
        "env_cls" : KitchenEnv_GC,
        "task_cls" : KitchenTask,
        "tasks"  : None,
        "cfg" : None 
    },
    "maze" : {
        "env_cls" : Maze_GC,
        "task_cls" : MazeTask_Custom, 
        "tasks"  : MAZE_TASKS,
        "cfg" : maze_config 
    },
}