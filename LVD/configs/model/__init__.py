from .wae import wae_config

from .sc import sc_config
from .skimo import skimo_config
from .simpl import simpl_config
from .gc_div_joint import gc_div_joint_config


MODEL_CONFIGS = dict(
    sc = sc_config,
    simpl = simpl_config,
    gc_div_joint = gc_div_joint_config,
    skimo = skimo_config,
    wae = wae_config
)