from easydict import EasyDict as edict 


simpl_config = edict(
    tanh = False,
    skill_concat = False,
    subgoal_loss = "prior",
    rollout_method = "rollout",
    workers = 14,
    sample_interval = 1,
    last = False,
    dynamics = False,
    dropout = 0
)