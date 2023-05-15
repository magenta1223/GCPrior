from ..base_config import BaseDataConfig
from easydict import EasyDict as edict
# from ...data.carla.carla_data_loader import *
from ...data import *


MODE_DICT = {
    "sc" : CARLA_Dataset,
    "sc_dreamer" : CARLA_Dataset,
    "simpl" : CARLA_Dataset,
    "skimo" : CARLA_Dataset,
    "gc_skimo" : CARLA_Dataset,
    "gc_div_joint" : CARLA_Dataset_Diversity,
}


class CARLAEnvConfig(BaseDataConfig):

    def __init__(self, structure = "sc"):
        super().__init__()

        config = edict(
            # DATA 
            dataset_class= MODE_DICT[structure],   
            action_dim=2, # 3
            state_dim=17, # 24
            n_obj = 17, # ?
            n_env = 0, # ? 
            n_goal = 2, # ? 
            env_name="carla", 
            env_name_offline="", # 
            subseq_len = 11, 
            only_proprioceptive = False,
            crop_rand_subseq=True,
            max_seq_len = 3000, 
            data_dir = ".", 
            epoch_cycles_train = 50,
            batch_size  = 1024,

            # Train Schedule 
            mixin_start = 30, 
            mixin_ratio = 0.05,
            plan_H = 100, 
            epochs = 70, 
            warmup_steps = 30,

            # Architecture
            latent_dim = 10,
            latent_state_dim  = 32,
            n_Layers = 5,
            hidden_dim = 128,
            reg_beta = 0.0005,
            prior_state_dim = 30,

            # RL
            time_limit = 3000, # orig 2000
            n_hidden = 5,
            target_kl_start  = 2, # orig 1 
            target_kl_end = 2, # orig 1 
            init_alpha = 0.1,
            only_increase = False,
            auto_alpha = True,
            reuse_rate = 512,
            q_warmup = 5000,
            q_weight = 1, 
            precollect = 20,
            early_stop_threshold = 80,
            use_hidden= True,
            finetune = True,
            n_episode = 300,
            consistency_lr = 1e-8,
            # policy_lr = 1e-8,
            policy_lr = 3e-6,
            gcprior = False,
            relative = False,
            robotics = True,

            mode = None,

            max_reward = 100,


            # etc.
            res=128,

        )

        if config['mode'] is not None:
            config['action_dim'] = 3
            
        self.set_attrs(config)
        self.name = "CARLA Environment Configuration"
