import random

import carla
import numpy as np
from typing_extensions import override

from carla_env.simulator.simulator import Simulator
from carla_env.simulator.vehicles.vehicle import Vehicle


class AutoVehicle(Vehicle):
    _LANE_ID_CANDIDATES = [-1, -2, -3, -4]

    @override
    def init(self):
        super().init()
        self.autopilot()

    @classmethod
    @override
    def spawn(cls, *args, simulator: Simulator, **kwargs):
        blueprint = cls.__get_blueprint(simulator)
        spawn_points = simulator.world.map.get_spawn_points()

        def town04_get_initial_transform() -> carla.Transform:
            road_id = 47
            road_length = 117.0
            initial_transform = simulator.world.map.get_waypoint_xodr(
                road_id=road_id,
                lane_id=random.choice(cls._LANE_ID_CANDIDATES),
                s=np.random.random.uniform(road_length),
            ).transform
            return initial_transform

        def default_get_initial_transform() -> carla.Transform:
            return random.choice(spawn_points)

        if simulator.world.map.name == "Town04":
            initial_transform = town04_get_initial_transform()
        else:
            initial_transform = default_get_initial_transform()

        vehicle = None
        while not vehicle:
            vehicle = super().spawn(
                *args,
                simulator,
                blueprint,
                initial_transform,
                **kwargs,
            )
        return vehicle

    @staticmethod
    def __blueprint_filter(blueprint: carla.ActorBlueprint) -> bool:
        """Filter the blueprints of the vehicles."""
        return int(blueprint.get_attribute("number_of_wheels")) == 4

    @classmethod
    def __get_blueprint(cls, simulator: Simulator) -> carla.ActorBlueprint:
        """Get a random blueprint of the vehicle."""
        blueprints = list(
            filter(
                cls.__blueprint_filter,
                simulator.world.blueprint_library.filter("vehicle.*"),
            )
        )
        blueprint = random.choice(blueprints)

        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        blueprint.set_attribute("role_name", "autopilot")

        return blueprint
