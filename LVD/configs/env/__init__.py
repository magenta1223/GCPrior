
from .kitchen import KitchenEnvConfig
from .calvin import CALVINEnvConfig
from .maze import MazeEnvConfig


ENV_CONFIGS = dict(
    kitchen = KitchenEnvConfig,
    calvin = CALVINEnvConfig,
    maze = MazeEnvConfig
)