from easydict import EasyDict as edict 


wae_config = edict(
    tanh = False,
    skill_concat = False,
    subgoal_loss = "prior",
    rollout_method = "rollout",
    workers = 14,
    sample_interval = 1,
    last = False,
    dynamics = True,
    dropout = 0
)