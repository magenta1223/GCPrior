from ..base_config import BaseDataConfig
from easydict import EasyDict as edict
# from ...data.kitchen.kitchen_data_loader import *
from ...data.calvin.calvin_data_loader import *


MODE_DICT = {
    "sc" : CALVIN_Dataset,
    "sc_div" : CALVIN_Diversity_Dataset,
    "gc" : CALVIN_GoalConditionedDataset,
    "gc_div" : CALVIN_GoalConditioned_Diversity_Dataset,
    "gc_div_joint" : CALVIN_GoalConditioned_Diversity_Dataset,
    "ae" : CALVIN_AEDataset,
}


class CALVINEnvConfig(BaseDataConfig):

    def __init__(self, structure = "sc"):
        super().__init__()

        config = edict(
            dataset_class= MODE_DICT[structure],   
            # n_actions=7,
            action_dim=7,
            # state_dim=21,
            # n_obj = 15,
            # n_env = 6, 
            state_dim=21,
            n_obj = 15,
            n_env = 6, 
            n_goal = 6,
            env_name="CALVIN",
            res=128,
            crop_rand_subseq=True,
            max_seq_len = 500,
            subseq_len = 11
        )
            
        self.set_attrs(config)
        self.name = "CALVIN Environment Configuration"
