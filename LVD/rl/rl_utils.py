
# envs & utils
import numpy as np

# simpl contribs
from ..contrib.simpl.torch_utils import itemize

# models
from ..utils import *

seed_everything()


# # Generating single task policy with SAC 




def render_task(env, env_name, policy, low_actor, tanh = False):
    imgs = []
    state = env.reset()


    processor = StateProcessor(env_name = env_name)


    G = processor.state2goal(state)
    state = processor.state_process(state)

    low_actor.eval()
    policy.eval()

    done = False
    time_step = 0
    time_limit = 280
    print("rendering..")
    count = 0

    while not done and time_step < time_limit: 
        if time_step % 10 == 0:
            if tanh:
                high_action_normal, high_action, loc, scale = policy.act(state, G)
            else:
                high_action, loc, scale = policy.act(state, G)

        with low_actor.condition(high_action), low_actor.expl():
            low_action = low_actor.act(state)
        
        state, reward, done, info = env.step(low_action)
        state = processor.state_process(state)
        img = env.render(mode = "rgb_array")
        imgs.append(img)
        time_step += 1
    print("done!")
    return imgs

