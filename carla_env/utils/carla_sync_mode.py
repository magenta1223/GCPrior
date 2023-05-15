import queue
from typing import List, Union

import carla

from carla_env.simulator.sensors.sensor import Sensor
from carla_env.simulator.world import World


class CarlaSyncMode:
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world: World, *sensors: Sensor, fps: float = 20):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues: List[Union[
            queue.Queue[carla.WorldSnapshot],
            queue.Queue[carla.SensorData],
        ]] = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout: float):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        if self._settings is not None:
            self.world.apply_settings(self._settings)

    def _retrieve_data(
        self,
        sensor_queue: Union[
            "queue.Queue[carla.WorldSnapshot]", "queue.Queue[carla.SensorData]"
        ],
        timeout: float,
    ):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
