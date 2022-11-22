from ..base_config import BaseDataConfig
from easydict import EasyDict as edict
from ...data.kitchen.src.kitchen_data_loader import *

import d4rl

MODE_DICT = {
    "default" : D4RLSequenceSplitDataset,
    "gc" : D4RLGoalRelabelingDataset,
    "did" : D4RLDIDDataset,
    "gcid" : D4RLGCIDDataset,
    "vic" : D4RLGCIDDataset,

}


class KitchenEnvConfig(BaseDataConfig):

    def __init__(self, mode = "default"):
        super().__init__()




        config = edict(
            dataset_class= MODE_DICT[mode],   
            n_actions=9,
            state_dim=60,
            env_name="kitchen-mixed-v0",
            res=128,
            crop_rand_subseq=True,
            max_seq_len = 280
        )
            
        self.set_attrs(config)
        self.name = "Kitchen Environment Configuration"
