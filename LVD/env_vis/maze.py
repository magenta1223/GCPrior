"""
Docs
State Sequence, Action sequence를 받아서 Video로 렌더링
"""

import numpy as np
import cv2
from ..envs import ENV_TASK
import torch

QPO_DIM = 2
INIT_QV = np.array([0, 0])
Hsteps = 10

def maze_render_function(env, states, actions, mode):
    """
    """
    env.reset()
    imgs = []

    video_len = states.shape[0]

    if isinstance(actions, torch.tensor):
        actions = actions.detach().cpu().numpy()


    if mode == "state":
        for i in range(video_len):
        # for i in range(self.Hsteps + 1):
            env.set_state(states[i][:QPO_DIM], INIT_QV)
            imgs.append(env.render(mode = "rgb_array"))

    else:
        env.set_state(states[0][:QPO_DIM], INIT_QV)
        imgs.append(env.render(mode = "rgb_array"))

        # action을 수행. 그러나 data 수집 당시의 qv와 달라서 약간 달라짐. 강제로 교정 후 render
        env.step(actions[0].detach().cpu().numpy())
        now_qv = env.sim.get_state().qvel
        env.set_state(states[1][:QPO_DIM], now_qv)
        
        # flat_d_len = 10 if self.rollout_method == "rollout" else 20
        flat_d_len = 10
        for i in range(flat_d_len -1):
            # render 
            imgs.append(env.render(mode = "rgb_array"))
            state, reward, done, info = env.step(actions[i].detach().cpu().numpy())

        last_img = env.render(mode = "rgb_array")
        imgs.append(last_img)
                
        for i in range(Hsteps + 1, video_len):
        # for i in range(self.Hsteps + 1):
            imgs.append(last_img)

    return imgs



def maze_imaginary_trajectory(env, states, actions, c, path):
    """
    생성된 trajectory의 state와 action이 일관적인지 비교하기 위함.
    근데 dataset이 만들어진 환경과 내 환경이 달라서 kitchen과 같은 방식으로는 불가능
    그냥 생성된 image가 맞는지 정도만 비교 
    """
    task = ENV_TASK['maze']['task_cls']([10, 10], [15, 15])


    pos = states[:, :2].detach().cpu().numpy()
    imgs_state = states[:, 4:].reshape(-1, 32, 32) * 255
    imgs_state = np.tile(imgs_state[:,:,:, None], (1,1,1,3))
    imgs_state[:, 15:17, 15:17] = [255, 0,0]
    imgs_state = imgs_state.astype(np.uint8)

    

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (200, 200))
    for i in range(len(imgs_state)):
        # writing to a image array
        img = cv2.resize(imgs_state[i], (200, 200))
        text = f"Now {i} C {c}" if c != 0 else f"S-A now {i}"
        cv2.putText(img = img, text = text, color = (255,0,0),  org = (0 // 2, 200 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, lineType= cv2.LINE_AA)
        text = f"{pos[i][0]:.2f} {pos[i][1]:.2f}"
        cv2.putText(img = img, text = text, color = (255,0,0),  org = (0 // 2, 300 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.6, lineType= cv2.LINE_AA)

        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release() 

def maze_render(env, states):
    if isinstance(states,  torch.Tensor):
        states = states.detach().cpu().numpy()
    # env.set_state(states[:QPO_DIM], INIT_QV)
    # img = env.render(mode = "agent_centric")

    imgs_state = states[4:].reshape(-1, 32, 32) * 255
    imgs_state = np.tile(imgs_state[:,:,:, None], (1,1,1,3))
    imgs_state[:, 15:17, 15:17] = [127, 127,127]
    imgs_state = imgs_state.astype(np.uint8)

    return imgs_state