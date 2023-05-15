from typing import Callable, Iterable, Optional, Tuple, Union

import carla

from carla_env.simulator.carla_wrapper import CarlaWrapper
from carla_env.simulator.simulator import Simulator
from carla_env.utils.logger import Logging

logger = Logging.get_logger(__name__)


class World(CarlaWrapper[carla.World]):
    """The world of the simulator. This class is a wrapper of the carla.World class.

    Args:
        world (carla.World): The world of the simulator.

    """

    def __init__(self, world: carla.World, simulator: Simulator):
        super().__init__(world)

        self.__simulator = simulator

        self.carla.tick()
        self.__removing_old_actors()

    def __removing_old_actors(self):
        actors = self.carla.get_actors()
        for vehicle in actors.filter("*vehicle*"):
            logger.warning("Destroying old vehicle %s", vehicle.id)
            vehicle.destroy()
        for sensor in actors.filter("*sensor*"):
            logger.warning("Destroying old sensor %s", sensor.id)
            sensor.destroy()

    def get_spectator(self):
        from carla_env.simulator.spectator import Spectator

        return Spectator(self.carla.get_spectator())

    @property
    def spectator(self):
        """The spectator of the simulator."""
        return self.get_spectator()

    def get_vehicles(self):
        from carla_env.simulator.vehicles.vehicle import Vehicle

        return [
            Vehicle(self.__simulator, vehicle)
            for vehicle in self.carla.get_actors().filter("*vehicle*")
            if isinstance(vehicle, carla.Vehicle)
        ]

    def get_traffic_lights(self):
        from carla_env.simulator.actor import Actor

        return [
            Actor(self.__simulator, traffic_light)
            for traffic_light in self.carla.get_actors().filter("*traffic_light*")
            if isinstance(traffic_light, carla.TrafficLight)
        ]

    def get_actor(self, actor_id: int):
        from carla_env.simulator.actor import Actor

        return Actor(self.__simulator, self.carla.get_actor(actor_id))

    def get_actors(self, actor_ids: Optional[Iterable[int]] = None):
        from carla_env.simulator.actor import Actor

        if actor_ids:
            actors = self.carla.get_actors(list(actor_ids))
        else:
            actors = self.carla.get_actors()
        return [Actor(self.__simulator, actor) for actor in actors]

    def tick(self, timeout: float = 10.0):
        """Tick the world."""
        return self.carla.tick(timeout)

    def wait_for_tick(self, timeout: float = 10.0):
        return self.carla.wait_for_tick(timeout)

    def on_tick(self, callback: Callable[[carla.WorldSnapshot], None]):
        """Register a callback to be called every tick."""
        return self.carla.on_tick(callback)

    def remove_on_tick(self, callback_id: int):
        """Remove a callback from being called every tick."""
        self.carla.remove_on_tick(callback_id)

    def get_settings(self):
        """Get the settings of the world."""
        return self.carla.get_settings()

    def apply_settings(self, settings: carla.WorldSettings):
        """Apply the settings to the world."""
        self.carla.apply_settings(settings)

    def draw_arrow(
        self,
        begin: carla.Location,
        end: carla.Location,
        thickness: float = 0.1,
        arrow_size: float = 0.1,
        life_time: Optional[float] = None,
    ):
        """Draw an arrow in the world."""
        if life_time is None:
            life_time = self.carla.get_settings().fixed_delta_seconds * 1.1
        self.carla.debug.draw_arrow(
            begin, end, thickness=thickness, arrow_size=arrow_size, life_time=life_time
        )

    def draw_point(
        self,
        location: carla.Location,
        size: float = 0.1,
        color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (255, 0, 0),
        life_time: Optional[float] = None,
    ):
        """Draw a point in the world."""
        if life_time is None:
            life_time = self.carla.get_settings().fixed_delta_seconds * 1.1
        self.carla.debug.draw_point(
            location, size=size, color=carla.Color(*color), life_time=life_time
        )

    @property
    def map(self):
        """The map of the world."""
        return self.carla.get_map()

    @property
    def weather(self):
        """The weather of the world."""
        return self.carla.get_weather()

    @weather.setter
    def weather(self, weather: carla.WeatherParameters):
        """The weather of the world."""
        return self.carla.set_weather(weather)

    @property
    def blueprint_library(self):
        """The blueprint library of the world."""
        return self.carla.get_blueprint_library()
