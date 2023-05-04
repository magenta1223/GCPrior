"""
Docs
State Sequence, Action sequence를 받아서 Video로 렌더링
"""

from ..envs import ENV_TASK
import numpy as np
import cv2
import torch 

QPO_DIM = 30
INIT_QV = np.zeros((29))
Hsteps = 10



def kitchen_video(env, states, actions, mode):
    """
    rollout을 통해 만들어낸 trajectory의
    -state sequence를 강제로 세팅
    -초기 state를 세팅하고, actino을 환경상에서 수행
    두 개를 비교
    """
    env.reset()
    imgs = []
    
    if isinstance(actions, torch.tensor):
        actions = actions.detach().cpu().numpy()

    video_len = states.shape[0]

    if mode == "state":
        for i in range(video_len):
        # for i in range(self.Hsteps + 1):
            env.set_state(states[i][:QPO_DIM], INIT_QV)
            imgs.append(env.render(mode = "rgb_array"))

    else:
        env.set_state(states[0][:QPO_DIM], INIT_QV)
        imgs.append(env.render(mode = "rgb_array"))

        # action을 수행. 그러나 data 수집 당시의 qv와 달라서 약간 달라짐. 강제로 교정 후 render
        env.step(actions[0])
        now_qv = env.sim.get_state().qvel
        env.set_state(states[1][:QPO_DIM], now_qv)
        
        # flat_d_len = 10 if self.rollout_method == "rollout" else 20
        flat_d_len = 10
        for i in range(flat_d_len -1):
            # render 
            imgs.append(env.render(mode = "rgb_array"))
            state, reward, done, info = env.step(actions[i])

        last_img = env.render(mode = "rgb_array")
        imgs.append(last_img)
                
        for i in range(Hsteps + 1, video_len):
        # for i in range(self.Hsteps + 1):
            imgs.append(last_img)

    return imgs


def kitchen_imaginary_trajectory(env, states, actions, c, path):
    """
    생성된 trajectory의 state와 action이 일관적인지 비교. 
    """
    task = ENV_TASK['kitchen']['task_cls'](['kettle'])

    with env.set_task(task):
        imgs_state = kitchen_video(env, states, actions, mode = "state")
        imgs_action = kitchen_video(env, states, actions, mode = "action")

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (1200,400))
    for i in range(len(imgs_state)):
        # writing to a image array
        img_s = imgs_state[i].astype(np.uint8)
        img_a = imgs_action[i].astype(np.uint8)
        img = np.concatenate((img_s,img_a, np.abs(img_s - img_a)), axis = 1)
        text = f"S-A now {i} c {c}" if c != 0 else f"S-A now {i}"
        cv2.putText(img = img,    text = text, color = (255,0,0),  org = (400 // 2, 400 // 2), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 2, lineType= cv2.LINE_AA)
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release() 

def kitchen_render(env, states):
    env.set_state(states[:QPO_DIM], INIT_QV)
    img = env.render(mode = "rgb_array")
    return img