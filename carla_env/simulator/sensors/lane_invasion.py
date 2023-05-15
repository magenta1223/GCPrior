from threading import Lock
from typing import Set, cast

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.simulator import Simulator
from carla_env.utils.lock import lock_release_after


class LaneInvasionSensor(Sensor[carla.LaneInvasionEvent]):
    @override
    def init(self):
        self.__lane_types: Set[carla.LaneMarkingType] = set()
        self.__lock = Lock()

        self.listen(self._callback__on_invasion)

    @classmethod
    @override
    def spawn(
        cls,
        simulator: Simulator,
        parent: Actor,
    ):
        blueprint_library = simulator.world.blueprint_library
        blueprint = blueprint_library.find("sensor.other.lane_invasion")

        return super().spawn(
            simulator=simulator,
            blueprint=blueprint,
            attach_to=parent,
        )

    def reset(self):
        while self.__lock.locked():
            pass
        self.__lock.acquire()
        self.__lane_types.clear()
        lock_release_after(self.__lock, 0.1)

    @override
    def destroy(self) -> None:
        self.reset()
        super().destroy()

    def _callback__on_invasion(self, data: carla.SensorData):
        if self.__lock.locked():
            return

        self.__lock.acquire()

        event = cast(carla.LaneInvasionEvent, data)
        self.__lane_types = set(marking.type for marking in event.crossed_lane_markings)

        self.__lock.release()

    @property
    def lane_types(self) -> Set[carla.LaneMarkingType]:
        return self.__lane_types
