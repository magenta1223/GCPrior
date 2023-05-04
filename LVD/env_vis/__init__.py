from .kitchen import *
from .maze import *


RENDER_FUNCS = {
    "kitchen" : {
        "scene" : kitchen_render,
        "imaginary_trajectory" : kitchen_imaginary_trajectory,
    },
    "maze" : {
        "scene" : maze_render,
        "imaginary_trajectory" : maze_imaginary_trajectory
    }
}