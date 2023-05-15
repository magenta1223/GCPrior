import carla

from carla_env.simulator.simulator import Simulator
from carla_env.simulator.traffic_manager import TrafficManager
from carla_env.simulator.world import World


class Client(carla.Client):
    """The client of the simulator. This class is a wrapper of the carla.Client class.

    Args:
        host (str): The host of the simulator.

        port (int): The port of the simulator.

    """

    def __init__(self, simulator: Simulator, host: str, port: int) -> None:
        super().__init__(host, port)
        self.__world = World(self.get_world(), simulator)
        self.set_timeout(10.0)

    def get_trafficmanager(self, client_connection: int = 8000) -> TrafficManager:
        return TrafficManager(super().get_trafficmanager(client_connection))

    @property
    def world(self) -> World:
        """The world of the simulator."""
        return self.__world

    @property
    def traffic_manager(self) -> TrafficManager:
        """The traffic manager of the simulator."""
        return self.get_trafficmanager()
