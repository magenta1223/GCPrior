from typing import Callable

import carla

from carla_env.simulator.simulator import Simulator
from carla_env.utils.config import ExperimentConfigs


class Visualizer:
    def __init__(self, simulator: Simulator, config: ExperimentConfigs):
        self.__simulator = simulator
        self.__config = config

        if self.__config.visual.draw_path:
            self.enable(self.draw_path)
        if self.__config.visual.draw_velocity:
            self.enable(self.draw_velocity)

    def enable(self, method: Callable[[carla.WorldSnapshot], None]):
        return self.simulator.world.on_tick(method)

    def disable(self, method_id: int):
        self.simulator.world.remove_on_tick(method_id)

    def draw_path(self, _):
        self.simulator.world.draw_point(
            self.simulator.ego_vehicle.location + carla.Location(z=0.1),
            life_time=10,
        )

    def draw_velocity(self, _):
        vel = self.simulator.ego_vehicle.velocity.length() + 1e-6
        vec_len = 5
        self.simulator.world.draw_arrow(
            self.simulator.ego_vehicle.location + carla.Location(z=0.1),
            self.simulator.ego_vehicle.location + carla.Location(
                x=self.simulator.ego_vehicle.velocity.x * vec_len / vel,
                y=self.simulator.ego_vehicle.velocity.y * vec_len / vel,
                z=0.1,
            ),
        )

    @property
    def simulator(self):
        return self.__simulator
    
    @property
    def config(self):
        return self.__config
