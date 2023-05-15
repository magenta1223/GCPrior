import math
from enum import Enum

import carla

from carla_env.simulator.actor import Actor
from carla_env.simulator.carla_wrapper import CarlaWrapper


class Spectator(CarlaWrapper[carla.Actor]):
    class FollowMode(Enum):
        BEHIND = 0
        ABOVE = 1
        INSIDE = 2

    def follow(
        self, actor: Actor, mode: FollowMode = FollowMode.ABOVE, cascade: bool = True
    ):
        world = self.carla.get_world()

        height = actor.carla.bounding_box.extent.z * 2

        if mode == Spectator.FollowMode.BEHIND:
            pitch_rad = math.radians(10)
            distance = actor.carla.bounding_box.extent.x * 5
            distance_z = distance * math.sin(pitch_rad)
            distance_xy = distance * math.cos(pitch_rad)

            def transform():
                degree = math.radians(actor.transform.rotation.yaw)
                delta = carla.Location(
                    x=-distance_xy * math.cos(degree),
                    y=-distance_xy * math.sin(degree),
                    z=height + distance_z,
                )
                return carla.Transform(
                    actor.transform.location + delta,
                    carla.Rotation(pitch=-10, yaw=actor.transform.rotation.yaw),
                )

        elif mode == Spectator.FollowMode.ABOVE:

            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90),
                )

        elif mode == Spectator.FollowMode.INSIDE:

            def transform():
                return carla.Transform(
                    actor.transform.location + carla.Location(z=height * 0.8),
                    carla.Rotation(pitch=-5, yaw=actor.transform.rotation.yaw),
                )

        callback_id = world.on_tick(lambda _: self.carla.set_transform(transform()))

        if cascade:
            actor.on_destroy(lambda: world.remove_on_tick(callback_id))
