
import os
import random
import time
from datetime import datetime
import torch
from torch.nn import functional as F
import torch.distributions as torch_dist
from torch.distributions.kl import register_kl
import numpy as np
from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES, BONUS_THRESH
from ..contrib.dists import TanhNormal
import cv2
from ..envs import ENV_TASK
from ..utils import *
from .env_vis import *


def render_state(env, env_name, state, qv = None):
    if env_name == "kitchen":
        env.set_state(state[:30], qv)
        return env.render(mode = "rgb_array")

    elif env_name == "calvin":
        robot_state = state[:15]
        scene_state = state[15:]
        env.set_state(robot_state = robot_state, scene_state = scene_state)
        return env.render(mode = "rgb_array")
    else:
        return

def render_action(env, env_name, action):

    env.step(action)
    return env.render(mode = "rgb_array")

    # if env_name == "kitchen":
    #     env.step(action)
    #     return env.render(mode = "rgb_array")

    # elif env_name == "calvin":
    #     return
    # else:
    #     return  




def render_kitchen(env, states, actions, mode):
    """
    rollout을 통해 만들어낸 trajectory의
    -state sequence를 강제로 세팅
    -초기 state를 세팅하고, actino을 환경상에서 수행
    두 개를 비교
    """
    Hsteps = 10
    qv = env.init_qvel[:].copy()

    imgs = []

    video_len = states.shape[0]

    if mode == "state":
        for i in range(video_len):
            imgs.append(render_state(env, "kitchen", states[i], qv))

    else:
        imgs.append(render_state(env, "kitchen", states[i], qv))

        # action을 수행. 그러나 data 수집 당시의 qv와 달라서 약간 달라짐. 강제로 교정 후 render
        env.step(actions[0].detach().cpu().numpy())
        now_qv = env.sim.get_state().qvel
        env.set_state(states[1][:30], now_qv)
        imgs.append(env.render(mode = "rgb_array"))
        flat_d_len = 10
        for i in range(flat_d_len -1):
            # render 
            img = render_action(env, "kitchen", actions[i].detach().cpu().numpy())
            imgs.append(img)

    
        for i in range(Hsteps + 1, video_len):
        # for i in range(self.Hsteps + 1):
            imgs.append(img)

    return imgs


def render_compare(env, env_name, states, actions, num, mp4_path, c = 0, outputs = None):
    
    Hsteps = 10
    task_cls, tasks = ENV_TASK[env_name]['env_cls'], ENV_TASK[env_name]['env_cls']
    task_obj = task_cls(tasks[0])


    with env.set_task(task_obj):
        env.reset()
        imgs_state = render_kitchen(states, actions, mode = "state")
        env.reset()
        imgs_action = render_kitchen(states, actions, mode = "action")

        # imgs_action = render_video_compare(self.loss_dict['states_novel'][0], self.loss_dict['actions_novel'][0], mode = "action")

    # mp4_path = f"./imgs/gcid_stitch/video/video_{num}.mp4"
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (1200,400))

    
    goal = states[-1].detach().cpu().numpy()
    task = GOAL_CHECKERS[env_name](GOAL_TRANSFORM[env_name](goal))

    for i in range(len(imgs_state)):
        # writing to a image array
        img_s = imgs_state[i].astype(np.uint8)
        img_a = imgs_action[i].astype(np.uint8)
        img = np.concatenate((img_s,img_a, np.abs(img_s - img_a)), axis = 1)
        text = f"{task} S-A now {i} c {c}" if c != 0 else f"S-A now {i}"
        cv2.putText(img = img, text = text, color = (255,0,0),  org = (400 // 2, 400 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release() 

    if env_name == "kitchen":
        qv = env.init_qvel[:].copy()

        img_GT = render_state(outputs['states'][i, -1][:30], qv)
        img_D = render_state(outputs['subgoal_recon_D'][i, -1][:30], qv)
        img_F = render_state(outputs['subgoal_recon_f'][i, -1][:30], qv)
        img_F_D = render_state(outputs['subgoal_recon_D_f'][i, -1][:30], qv)


        # skill = outputs['z_sub']
        # dec_input = dec_input(outputs['states'], skill, Hsteps)
        # N, T, _ = dec_input.shape
        # raw_actions = self.skill_decoder(dec_input.view(N * T, -1)).view(N, T, -1)[i]

        

        with env.set_task(task_obj):
            env.reset()
            env.set_state(outputs['states'][i, 0][:30], qv)
            for idx in range(Hsteps):
                env.step(outputs['raw_actions'][idx].detach().cpu().numpy())

        img_sub_skill = env.render(mode = "rgb_array")
        task = GOAL_CHECKERS['kitchen'](outputs['subgoal_recon_f'][i].detach().cpu().numpy()[:30])

        img = np.concatenate((img_GT, img_D, img_F, img_F_D, img_sub_skill), axis= 1)
        cv2.putText(img = img,    text = task, color = (255,0,0),  org = (400 // 2, 400 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)

        cv2.imwrite(f"./imgs/gcid_stitch/img/img_{num}.png", img)
        print(mp4_path)






def render_calvin(env, states, actions, mode):
    """
    rollout을 통해 만들어낸 trajectory의
    -state sequence를 강제로 세팅
    -초기 state를 세팅하고, actino을 환경상에서 수행
    두 개를 비교
    """
    Hsteps = 10

    imgs = []

    video_len = states.shape[0]

    if mode == "state":
        for i in range(video_len):
            imgs.append(render_state(env, "calvin", states[i]))

    else:
        imgs.append(render_state(env, "calvin", states[0]))


        for i in range(Hsteps - 1):
            # render 
            img = render_action(env, "calvin", actions[i].detach().cpu().numpy())
            imgs.append(img)

    
        for i in range(Hsteps + 1, video_len):
        # for i in range(self.Hsteps + 1):
            imgs.append(img)

    return imgs



def render_compare_calvin(env, states, actions, mp4_path):

    # state를 강제로 setting했을 때
    env.reset()
    imgs_state = render_calvin(states, actions, mode = "state")
    # action을 수행했을 때 
    env.reset()
    imgs_action = render_calvin(states, actions, mode = "action")

    
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (1200,400))


    for i in range(len(imgs_state)):
        # writing to a image array
        img_s = imgs_state[i].astype(np.uint8)
        img_a = imgs_action[i].astype(np.uint8)
        img = np.concatenate((img_s,img_a, np.abs(img_s - img_a)), axis = 1)
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release() 