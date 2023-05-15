from typing import List, Optional

import carla

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from carla_env.simulator.route_selector import RouteSelector
from carla_env.simulator.vehicles.vehicle import Vehicle
from carla_env.simulator.world import World
from carla_env.utils.config import ExperimentConfigs
from carla_env.utils.route_planner import CustomGlobalRoutePlanner


class RouteManager:
    """Route Manager for Carla Simulator. This class is responsible for generating
    waypoints for the ego vehicle.

    Args:
        world (World): Carla world.

        sampling_resolution (float, optional): Sampling resolution for the route
            planner. Defaults to 0.1.

        route_list (Optional[List[Tuple[int, int]]], optional): List of route
            candidates. Defaults to None.

        random_route (bool, optional): Whether to use random route. Defaults to False.

    """

    def __init__(
        self,
        world: World,
        config: ExperimentConfigs,
        sampling_resolution: float = 0.1,
    ):
        self.__world = world
        self.__sampling_resolution = sampling_resolution
        self.__dao = GlobalRoutePlannerDAO(self.map, sampling_resolution)

        self.__route_planner = CustomGlobalRoutePlanner(self.__dao)
        self.__route_planner.setup()

        self.__route_selector = RouteSelector(
            self.__world, config.routes, config.random_route
        )
        if self.map.name == "Town04":
            self.__get_wayoints = self.__town04__get_waypoints
        else:
            self.__get_wayoints = self.__default_get_waypoints

        self.__initial_transform: Optional[carla.Transform] = None
        self.__target_transform: Optional[carla.Transform] = None
        self.__waypoints: List[carla.Waypoint] = []

    @staticmethod
    def __town04__get_waypoints(*_) -> List[carla.Waypoint]:
        """Get the waypoints of the route from the initial transform to the target
        transform.

        Args:
            initial_transform (carla.Transform): The initial transform of the route.

            target_transform (carla.Transform): The target transform of the route.

        Returns:
            List[carla.Waypoint]: The waypoints of the route.

        """
        return []

    def __default_get_waypoints(
        self,
        initial_transform: carla.Transform,
        target_transform: carla.Transform,
    ) -> List[carla.Waypoint]:
        """Get the waypoints of the route from the initial transform to the target
        transform.

        Args:
            initial_transform (carla.Transform): The initial transform of the route.

            target_transform (carla.Transform): The target transform of the route.

        Returns:
            List[carla.Waypoint]: The waypoints of the route.

        """
        return self.__route_planner.trace_route(
            initial_transform.location, target_transform.location
        )

    def select_route(self):
        """Select a new route.

        The selected route is stored in the following properties:
            - initial_transform
            - target_transform
            - waypoints

        """
        self.__initial_transform, self.__target_transform = self.__route_selector.next()
        self.__waypoints = self.__get_wayoints(
            self.initial_transform, self.target_transform
        )

    def compute_direction_velocities(
        self, vehicle: Vehicle, target_location: carla.Location
    ):
        return self.__route_planner.compute_direction_velocities(
            vehicle.location, vehicle.velocity, target_location
        )

    @property
    def world(self):
        """The world of the simulator."""
        return self.__world

    @property
    def map(self):
        """The map of the simulator."""
        return self.world.map

    @property
    def sampling_resolution(self):
        """The sampling resolution of the simulator."""
        return self.__sampling_resolution

    @property
    def waypoints(self):
        """The waypoints of the current route."""
        return self.__waypoints

    @property
    def initial_transform(self):
        """The initial transform of the current route."""
        if self.__initial_transform is None:
            raise ValueError("No route is selected. Please call select_route() first.")
        return self.__initial_transform

    @property
    def target_transform(self):
        """The target transform of the current route."""
        if self.__target_transform is None:
            raise ValueError("No route is selected. Please call select_route() first.")
        return self.__target_transform
