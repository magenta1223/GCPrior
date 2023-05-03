from ..base_config import BaseDataConfig
from easydict import EasyDict as edict
from ...data.maze.maze_data_loader import *


MODE_DICT = {
    # "sc" : Maze_StateConditioned,
    "sc" : Maze_AgentCentric_StateConditioned,
    "gc_div_joint" : Maze_AgentCentric_GoalConditioned_Diversity,
    "wae" : Maze_AEDataset
}


class MazeEnvConfig(BaseDataConfig):

    def __init__(self, structure = "sc"):
        super().__init__()

        # datset spec
        config = edict(
            # dataset_class= MODE_DICT[structure],   
            # action_dim=2,
            # state_dim=4,
            # n_obj = 4,
            # n_env = 0,
            # n_goal = 4,
            # env_name="maze",
            # res=128,
            # crop_rand_subseq=True,
            # max_seq_len = 300,
            # subseq_len = 11,

            # DATA 
            dataset_class= MODE_DICT[structure],   
            action_dim=2,
            state_dim=4,
            n_obj = 4,
            n_env = 0,
            n_goal = 4,
            env_name="maze",
            subseq_len = 11, 
            only_proprioceptive = False,
            crop_rand_subseq=True,
            max_seq_len = 300,
            data_dir = "./LVD/data/maze/maze.pkl",
            epoch_cycles_train = 1,
            batch_size  = 64,

            # Train Schedule 
            mixin_start = 30,
            mixin_ratio = 0.05,
            plan_H = 100, 
            epochs = 70, 
            warmup_steps = 30,

            # Architecture
            latent_dim = 10,
            latent_env_dim = 32,
            latent_state_dim  = 32,
            n_Layers = 5,
            hidden_dim = 128,
            # reg_beta = 0.01,
            reg_beta = 0.01,

            


            # RL
            time_limit = 3000, # orig 2000
            n_hidden = 5,
            target_kl_start  = 2, # orig 1 
            target_kl_end = 2, # orig 1 
            init_alpha = 0.05,
            only_increase = False,
            auto_alpha = True,
            reuse_rate = 256,
            policy_lr = 3e-4,
            q_warmup = 5000,
            q_weight = 1, 
            precollect = 10,
            early_stop_threshold = 0.8,
            # prior_state_dim = 4,
            # policy_state_dim = 6, # 기존 방법론의 경우는 다 붙여서 넣으니까

            # prior_state_dim = 32 + 4 * 10, # latent_env_dim + 10 * prior_state_dim
            # policy_state_dim = 32 + 4 * 10 + 2, # 기존 방법론의 경우는 다 붙여서 넣으니까
            prior_state_dim = 32 + 4, # latent_env_dim + 10 * prior_state_dim
            policy_state_dim = 32 + 4 + 2, # 기존 방법론의 경우는 다 붙여서 넣으니까

            # etc.
            res=128,


        )
            
        self.set_attrs(config)
        self.name = "Maze Environment Configuration"
