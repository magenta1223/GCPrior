import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os 
from glob import glob


def visualize(imgs, task_name, prefix = None):
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(imgs)):
        frames.append([plt.imshow(imgs[i], cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    
    os.makedirs(f"./results/{task_name}", exist_ok= True)
    
    # fs = os.listdir(f"./{task_name}")
    
    fs = glob(f"./results/{task_name}/*.mp4")
    if len(fs):
        n =  max([ int(f.replace(f"./results/{task_name}/", "").replace("success", "").replace(".mp4", ""))  for f in fs]) + 1
    else:
        n = 0
    if prefix is not None:
        n = prefix + str(n)
    ani.save(f'./results/{task_name}/{n}.mp4')
