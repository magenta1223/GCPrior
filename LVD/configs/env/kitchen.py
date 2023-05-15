from ..base_config import BaseDataConfig
from easydict import EasyDict as edict
# from ...data.kitchen.kitchen_data_loader import *
from ...data import *


MODE_DICT = {
    "ae" : Kitchen_AEDataset,
    "sc" : D4RL_StateConditionedDataset,
    "sc_dreamer" : D4RL_StateConditionedDataset,
    "simpl" : D4RL_StateConditionedDataset,
    "sc_div" : D4RL_StateConditioned_Diversity_Dataset,
    "gc" : D4RL_GoalConditionedDataset,
    "gc_div" : D4RL_GoalConditioned_Diversity_Dataset,
    "gc_div_joint" : D4RL_GoalConditioned_Diversity_Dataset,
    "gc_div_joint_gp" : D4RL_GoalConditioned_Diversity_Dataset,
    "skimo" : D4RL_StateConditionedDataset,
    "gc_skimo" : D4RL_StateConditionedDataset,

}


class KitchenEnvConfig(BaseDataConfig):

    def __init__(self, structure = "sc"):
        super().__init__()

        config = edict(
            # DATA 
            dataset_class= MODE_DICT[structure],   
            action_dim=9,
            state_dim=30,
            n_obj = 9,
            n_env = 21,
            n_goal = 30,
            env_name="kitchen",
            env_name_offline="kitchen-mixed-v0",
            subseq_len = 11, 
            only_proprioceptive = False,
            crop_rand_subseq=True,
            max_seq_len = 280,
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
            time_limit = 280, # orig 2000
            n_hidden = 5,
            target_kl_start  = 50, # orig 1 
            target_kl_end = 15, # orig 1 
            init_alpha = 0.005,
            only_increase = True,
            auto_alpha = False,
            reuse_rate = 512,
            q_warmup = 5000,
            q_weight = 1, 
            precollect = 20,
            early_stop_threshold = 3.6,
            use_hidden= True,
            finetune = True,
            n_episode = 300,
            consistency_lr = 1e-8,
            # policy_lr = 1e-8,
            policy_lr = 3e-6,
            gcprior = False,
            relative = False,
            robotics = True,

            max_reward = 4,


            # etc.
            res=128,

        )
            
        self.set_attrs(config)
        self.name = "Kitchen Environment Configuration"
