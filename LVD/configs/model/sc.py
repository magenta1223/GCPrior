from easydict import EasyDict as edict 


sc_config = edict(
    tanh = True,
    skill_concat = False,
    subgoal_loss = "prior",
    rollout_method = "rollout",
    workers = 14,
    sample_interval = 1,
    last = False,
    dynamics = False,
    dropout = 0
)