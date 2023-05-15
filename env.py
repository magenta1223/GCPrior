import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import fire
# import flax
# import flax.linen as nn
# from flax.struct import dataclass, field
from typing_extensions import Concatenate, ParamSpec, override

# from carla_env.behavior_cloning import behavior_cloning
from carla_env.collect_data import collect_data
from carla_env.utils.config import parse_config
from carla_env.utils.logger import Logging

P = ParamSpec("P")
R = TypeVar("R")
# Params = flax.core.FrozenDict[str, Any]


# @dataclass
# class Model(Generic[P, R]):
#     step: int
#     params: Params
#     apply_fn: Callable[Concatenate[Params, P], R] = field(pytree_node=False)

#     @classmethod
#     def create(cls, model_def: nn.Module, params: Params) -> "Model":
#         return cls(step=1, apply_fn=model_def.apply, params=params)

#     def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
#         # pylint: disable=not-callable
#         return self.apply_fn(self.params, *args, **kwargs)


class Program:
    def __init__(self, cfg: str) -> None:
        config = parse_config(cfg)

        logging_path = (config.data_path or Path.cwd()) / "outputs.log"
        print("Logging to", logging_path)
        Logging.setup(
            filepath=logging_path,
            level=logging.DEBUG,
            formatter="(%(asctime)s) [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # self.bc = partial(behavior_cloning, config=config)
        self.dc = partial(collect_data, config=config)


if __name__ == "__main__":
    fire.Fire(Program)
