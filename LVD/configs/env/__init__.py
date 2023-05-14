
from .kitchen import KitchenEnvConfig
from .calvin import CALVINEnvConfig
from .maze import MazeEnvConfig
from .carla import CARLAEnvConfig


ENV_CONFIGS = dict(
    kitchen = KitchenEnvConfig,
    calvin = CALVINEnvConfig,
    maze = MazeEnvConfig,
    carla = CARLAEnvConfig
)