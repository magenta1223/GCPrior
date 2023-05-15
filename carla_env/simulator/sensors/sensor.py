from typing import Callable, Generic, TypeVar

import carla

from carla_env.simulator.actor import Actor

T = TypeVar("T", bound=carla.SensorData)


class Sensor(Actor[carla.Sensor], Generic[T]):
    def listen(self, callback: Callable[[T], None]):
        self.carla.listen(callback)  # type: ignore
