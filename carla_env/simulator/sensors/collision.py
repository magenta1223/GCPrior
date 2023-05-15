import math
from dataclasses import dataclass
from threading import Lock
from typing import List, cast

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator
from carla_env.utils.lock import lock_release_after
from carla_env.utils.logger import Logging

logger = Logging.get_logger(__name__)


@dataclass
class CollisionEvent:
    """Collision event data class."""

    actor_id: int
    """The id of the actor that collided with the ego vehicle."""

    frame: int
    """The frame in which the collision occurred."""

    intensity: float
    """The intensity of the collision."""


class CollisionSensor(Sensor[carla.CollisionEvent]):
    def init(self, max_queue: int = 4000):
        self.__collided = False
        self.__collision_history: List[CollisionEvent] = []
        self.__max_queue = max_queue
        self.__lock = Lock()

        self.listen(self._callback__on_collision)

    @classmethod
    @override
    def spawn(
        cls,
        simulator: Simulator,
        parent: Actor,
        max_queue: int = 4000,
    ):
        blueprint_library = simulator.world.blueprint_library
        blueprint = blueprint_library.find("sensor.other.collision")

        return super().spawn(
            simulator=simulator,
            blueprint=blueprint,
            attach_to=parent,
            max_queue=max_queue,
        )

    def _callback__on_collision(self, data: carla.SensorData):
        if self.__lock.locked():
            return
        self.__lock.acquire()

        event = cast(carla.CollisionEvent, data)

        if len(self.__collision_history) >= self.__max_queue:
            self.__collision_history.pop(0)

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

        self.__collided = True
        self.__collision_history.append(
            CollisionEvent(
                actor_id=event.other_actor.id,
                frame=event.frame,
                intensity=intensity,
            )
        )

        self.__lock.release()

    def reset(self) -> None:
        while self.__lock.locked():
            pass
        self.__lock.acquire()
        self.__collided = False
        self.__collision_history.clear()
        lock_release_after(self.__lock, 0.1)

    @property
    def has_collided(self) -> bool:
        """Whether the ego vehicle collided with other vehicles or obstacles."""
        return self.__collided

    @property
    def object(self) -> Actor:
        """The object that the ego vehicle collided with."""
        return self.simulator.world.get_actor(self.__collision_history[-1].actor_id)

    @property
    def objects(self) -> List[Actor]:
        """The objects that the ego vehicle collided with."""
        ids = set(history.actor_id for history in self.__collision_history)
        return self.simulator.world.get_actors(ids)
