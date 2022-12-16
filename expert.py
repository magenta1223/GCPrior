import matplotlib as plt
import numpy as np
import cv2
import pickle as pkl

import pandas as pd

OBS_SITE = {
    'microwave': 'microhandle_site',
    'bottom burner': 'knob2_site',
    'top burner': 'knob4_site',
    'light switch': 'light_site',
    'slide cabinet': 'slide_site',
    'hinge cabinet': 'hinge_site2',
    'kettle': 'kettle_site',
}

# from offline_baseliens_jax import SAC
from d4rl.kitchen.kitchen_envs import *
import matplotlib as plt
import numpy as np
import cv2
from easydict import EasyDict as edict
import argparse

from proposed.rl.vis import visualize


class KitchenAllTasksV0(KitchenBase):
    TASK_ELEMENTS = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']
    ENFORCE_TASK_ORDER = False

option_to_task = {
    'bb': 'bottom burner',
    'tb': 'top burner',
    'ls': 'light switch',
    'sc': 'slide cabinet',
    'hc': 'hinge cabinet',
    'mw': 'microwave',
    'kt': 'kettle'
}


OBS_SITE = {
    'microwave': 'microhandle_site',
    ## TODO ##
    'bottom burner': 'knob2_site',
    'top burner': 'knob4_site',
    'light switch': 'light_site',
    # ------- #
    'slide cabinet': 'slide_site',
    'hinge cabinet': 'hinge_site2',
    'kettle': 'kettle_site',
}

task_params = edict()

task_params['microwave'] = edict(
    name=OBS_SITE["microwave"],
    obj_pos=np.array([0, -0.01, 0]),
    obj_front_pos=np.array([0, -0.1, 0]),
    p0=edict(
        pos_scale=0.05,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p1=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p2=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p3=edict(
        pos_scale=0.01,
        pos_bias=np.array([-0.2, -1, 0]),
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    touch=-0.05,
    hold=-0.2
)


task_params['bottom burner'] = edict(
    name=OBS_SITE["bottom burner"],
    obj_pos=np.array([0, -0.015, 0]),
    obj_front_pos=np.array([0, - 0.1, 0]),
    p0=edict(
        pos_scale=0.05,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p1=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p2=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p3=edict(
        pos_scale=0.01,
        pos_bias=np.array([0, 0, 0]),
        quat_bias=np.array([-1.5, 0, -0.8, 0])  # np.array([-1.5, 0.01, 0, -0.0]),
    ),
    touch=-0.1,
    hold=-0.2
)

task_params['top burner'] = edict(
    name=OBS_SITE["top burner"],
    obj_pos=np.array([0, -0.025, 0]),
    obj_front_pos=np.array([0, - 0.1, 0]),
    p0=edict(
        pos_scale=0.05,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p1=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p2=edict(
        pos_scale=0.02,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p3=edict(
        pos_scale=0.01,  # 0.01,
        pos_bias=np.array([0, 0, 0]),
        quat_bias=np.array([-1.5, 0, -0.6, 0])  # np.array([-1.5, 0.01, 0, -0.0]),
    ),
    touch=-0.1,
    hold=-0.2
)

task_params['light switch'] = edict(
    name=OBS_SITE["light switch"],
    obj_pos=np.array([0, -0.025, 0]),
    obj_front_pos=np.array([0, - 0.12, 0]),
    p0=edict(
        pos_scale=0.05,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p1=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p2=edict(
        pos_scale=0.01,
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    p3=edict(
        pos_scale=0.01,
        pos_bias=np.array([-1, 0, 0]),
        quat_bias=np.array([-1.5, 0, 0, -0.0]),
    ),
    touch=-0.05,
    hold=-0.2
)


task_params['slide cabinet'] = edict(
    name = OBS_SITE["slide cabinet"],
    obj_pos = np.array([0, -0.025, 0]),
    obj_front_pos = np.array([0, -0.1, 0]),
    p0 = edict(
        pos_scale = 0.05,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p1 = edict(
        pos_scale = 0.01,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p2 = edict(
        pos_scale = 0.01,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p3 = edict(
        pos_scale = 0.01,
        pos_bias = np.array([1, 0, 0]),
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    touch = -0.05,
    hold = -0.2
)


task_params['hinge cabinet'] = edict(
    name = OBS_SITE["hinge cabinet"],
    obj_pos = np.array([0, -0.025, 0]),
    obj_front_pos = np.array([0, -0.08 , 0]),
    p0 = edict(
        pos_scale = 0.05,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p1 = edict(
        pos_scale = 0.01,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p2 = edict(
        pos_scale = 0.008,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p3 = edict(
        pos_scale = 0.01,
        pos_bias = np.array([0.8, -0.9, 0]),
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    touch = -1,
    hold = -2
)

task_params['kettle'] = edict(
    name = OBS_SITE["kettle"],
    obj_pos = np.array([-0.08, -0.025, -0.05]),
    obj_front_pos = np.array([-0.08, -0.08 , -0.05]),
    p0 = edict(
        pos_scale = 0.05,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p1 = edict(
        pos_scale = 0.01,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p2 = edict(
        pos_scale = 0.008,
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    p3 = edict(
        pos_scale = 0.01,
        pos_bias = np.array([0, 1, 0]),
        quat_bias = np.array([-1.5, 0, 0, -0.0]),
    ),
    touch = -0.01,
    hold = -1
)

class MicroWaveRuleBased(object):
    def __init__(self):
        with open('./microwave_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.8, 0.09], [0.5, 0.09], [0.3, 0.05], [0.3, 0.05], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.7
            action[8] = -0.7

        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action

class TopBurnerRuleBased(object):
    def __init__(self):
        with open('./top_burner_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.7, 0.09], [1., 0.09], [1.5, 0.05], [5.0, 0.03], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)
        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.7
            action[8] = -0.7
            action[6] = 1
            action[5] = -0.2
            action[4] = 0.3
            action[3] = 0.15

        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            if self.phase==4:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action


class BottomBurnerRuleBased(object):
    def __init__(self):
        with open('./bottom_burner_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.7, 0.09], [1., 0.09], [1.5, 0.05], [5.0, 0.03], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)
        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.7
            action[8] = -0.7
            action[6] = 1
            action[5] = -0.2
            action[4] = 0.3
            action[3] = 0.15

        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][idx][:9]
            if self.phase==4:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action

class SlideCabinetRuleBased(object):
    def __init__(self):
        with open('./slide_cabinet_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.8, 0.09], [0.5, 0.09], [1., 0.05], [1., 0.05], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)
        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.7
            action[8] = -0.7

        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][idx][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action

class KettleRuleBased(object):
    def __init__(self):
        with open('./kettle_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.8, 0.09], [0.5, 0.09], [0.3, 0.05], [1., 0.45], [1., 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)
        # print(np.linalg.norm(self.target_obs - obs[:9]))
        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][idx][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action

class LightSwitchRuleBased(object):
    def __init__(self):
        with open('./light_switch_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.8, 0.09], [0.5, 0.09], [1., 0.05], [1., 0.05], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)

        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.3
            action[8] = -0.3
        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action


class HingeCabinetRuleBased(object):
    def __init__(self):
        with open('./hinge_cabinet_obs.pkl', 'rb') as f:
            self.obs_history = pkl.load(f)
            self.phase = 1
            self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.option = [[0.8, 0.09], [0.5, 0.09], [0.5, 0.05], [1., 0.05], [0.5, 0.09]]
            self.grep = [0, 0, 0, 1, 0]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]

    def reset(self):
        self.phase = 1
        self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
        self.scale = self.option[self.phase - 1][0]
        self.threshold = self.option[self.phase - 1][1]

    def act(self, obs):
        # print(self.phase)

        action = np.array(self.target_obs - obs[:9]) / np.linalg.norm(self.target_obs - obs[:9]) * self.scale
        if self.grep[self.phase - 1]:
            action[7] = -0.9
            action[8] = -0.9
            action[5] = -0.2

        if np.linalg.norm(self.target_obs - obs[:9]) <= self.threshold:
            self.phase += 1
            if self.phase == 2:
                self.phase += 1
            if self.phase == 6:
                return None
            if self.phase == 5:
                idx = np.random.randint(6, 12)
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][idx][:9]
            else:
                self.target_obs = self.obs_history['phase_{}'.format(self.phase)][-1][:9]
            self.scale = self.option[self.phase - 1][0]
            self.threshold = self.option[self.phase - 1][1]
        return action


if __name__ == '__main__':
    import time



    env = KitchenAllTasksV0()
    SKILL_DICT = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'microwave', 'kettle', 'hinge cabinet']
    policies = {'bottom burner': BottomBurnerRuleBased,
                'top burner': TopBurnerRuleBased,
                'light switch': LightSwitchRuleBased,
                'slide cabinet': SlideCabinetRuleBased,
                'microwave': MicroWaveRuleBased,
                'kettle': KettleRuleBased,
                'hinge cabinet': HingeCabinetRuleBased}
    total_datasets = 0
    start_time = time.time()

    
    df = {}

    n_traj = 50

    for task in SKILL_DICT:
        steps = []

        for j in range(n_traj): # trajectory 개수 

            # np.random.shuffle(SKILL_DICT)
            env.TASK_ELEMENTS= [task]  # SKILL_DICT[:4] # 하나만 
            # print('{}/1000: {}'.format(j, time.time() - start_time), 'total_datasets', total_datasets, 'skills', SKILL_DICT[:4])
            # start_time = time.time()
            env.ENFORCE_TASK_ORDER = True
            obs = env.reset()
            timesteps = 0
            reward = 0
            skill_first_state = 0
            # for i in range(4): # task를 순서대로 수행. 난 싱글태스크니까 반복문 삭제
            
            policy = policies[task]()
            policy.reset()

            for _ in range(250):
                action = policy.act(obs)
                timesteps += 1
                if action is None:
                    break
                obs, rew, done, info = env.step(action)


                if done:
                    steps.append(timesteps)
                    break
        
        if len(steps) < n_traj:
            steps.extend([0] * (n_traj - len(steps)))

        df[task] = steps
    

    pd.DataFrame(df).to_csv("rulebased_result.csv", index = False)

    


    #
    #         img = env.render('rgb_array')
    #         BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #         video.write(BGR)
    # cv2.destroyAllWindows()
    # video.release()
    #
    # with open('./microwave_obs.pkl', 'wb') as f:
    #     pkl.dump(policy.obs_history, f)