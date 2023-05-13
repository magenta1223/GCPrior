from easydict import EasyDict as edict 

skimo_config = edict(
    tanh = True,
    skill_concat = True,
    subgoal_loss = "prior",
    rollout_method = None,
    workers = 14,
    sample_interval = 1,
    last = False,
    dynamics = True,
    dropout = 0,
    latent_state_dim = 32,
)