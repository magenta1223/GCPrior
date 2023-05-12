import matplotlib.pyplot as plt
import numpy as np

from d4rl.pointmaze.maze_model import WALL
from copy import deepcopy

def draw_maze(ax, env, episodes):
    img = np.rot90(env.maze_arr != WALL)
    extent = [
        -0.5, env.maze_arr.shape[0]-0.5,
        -0.5, env.maze_arr.shape[1]-0.5
    ]
    
    ax.imshow((1-img)/5, extent=extent, cmap='Reds', alpha=0.2)
    ax.scatter(*env.task.init_loc, marker='o', c='green', s=200, zorder=10, linewidths=4)
    ax.scatter(*env.task.goal_loc, marker='x', c='red', s=200, zorder=10, linewidths=4)
    ax.set_xlim(0, env.maze_size+1)
    ax.set_ylim(0, env.maze_size+1)

    
    for episode in episodes:
        states = deepcopy(np.array(episode.states))
        if env.relative:
            states[:, :2] += env.task.init_loc
        ax.plot(*states[:, :2].T  , color='royalblue', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
