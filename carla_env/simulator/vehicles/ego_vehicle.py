from typing import Optional

import carla
from typing_extensions import override

from carla_env.simulator.actor import ActorInitializeError
from carla_env.simulator.sensors.camera import CameraSensor
from carla_env.simulator.sensors.collision import CollisionSensor
from carla_env.simulator.sensors.lane_invasion import LaneInvasionSensor
from carla_env.simulator.sensors.lidar import LidarSensor
from carla_env.simulator.simulator import Simulator
from carla_env.simulator.spectator import Spectator
from carla_env.simulator.vehicles.vehicle import Vehicle
from carla_env.utils.config import ExperimentConfigs
from carla_env.utils.logger import Logging
from carla_env.utils.vector import to_array

logger = Logging.get_logger(__name__)


class EgoVehicle(Vehicle):
    @override
    def init(self, config: ExperimentConfigs):
        super().init()

        self.__vehicle_type = config.vehicle_type

        lidar_sensor = self.add_sensor(LidarSensor, config=config)
        camera = self.add_sensor(CameraSensor, config=config)
        collision_sensor = self.add_sensor(CollisionSensor)
        lane_invasion_sensor = self.add_sensor(LaneInvasionSensor)

        if (
            lidar_sensor is None
            or camera is None
            or collision_sensor is None
            or lane_invasion_sensor is None
        ):
            raise ActorInitializeError("Failed to spawn sensors of ego vehicle")

        self.__lidar_sensor = lidar_sensor
        self.__camera = camera
        self.__collision_sensor = collision_sensor
        self.__lane_invasion_sensor = lane_invasion_sensor

        self.velocity = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        self.angular_velocity = carla.Vector3D(x=0.0, y=0.0, z=0.0)

        self.simulator.world.spectator.follow(self, Spectator.FollowMode.ABOVE)

    @classmethod
    @override
    def spawn(
        cls,
        simulator: Simulator,
        config: ExperimentConfigs,
        initial_transform: Optional[carla.Transform] = None,
    ):
        blueprint = simulator.world.blueprint_library.find(
            f"vehicle.{config.vehicle_type}"
        )
        blueprint.set_attribute("role_name", "ego")

        return super().spawn(
            config,
            simulator=simulator,
            blueprint=blueprint,
            transform=initial_transform,
        )

    def reset(self):
        self.stop()
        self.collision_sensor.reset()
        self.lane_invasion_sensor.reset()

    @property
    def lidar_sensor(self) -> LidarSensor:
        """Lidar sensor of the ego vehicle."""
        return self.__lidar_sensor

    @property
    def camera(self) -> CameraSensor:
        """Camera sensor of the ego vehicle."""
        return self.__camera

    @property
    def collision_sensor(self) -> CollisionSensor:
        """Collision sensor of the ego vehicle."""
        return self.__collision_sensor

    @property
    def lane_invasion_sensor(self) -> LaneInvasionSensor:
        """Lane invasion sensor of the ego vehicle."""
        return self.__lane_invasion_sensor

    @property
    def vehicle_type(self):
        return self.__vehicle_type

    # def apply_control(self, control: carla.VehicleControl):
    #     wetness = self.simulator.world.weather.precipitation / 100.0
    #     diff = wetness / 9.5
    #     control.throttle = min(control.throttle + diff, 1.0)
    #     control.brake = max(control.brake - diff, 0.0)
    #     return super().apply_control(control)

    def get_observation(self):
        return {
            "acceleration": to_array(self.acceleration),
            "velocity": to_array(self.velocity),
            "angular_velocity": to_array(self.angular_velocity),
            "location": to_array(self.location),
            "rotation": to_array(self.rotation),
            "forward_vector": to_array(self.rotation.get_forward_vector()),
        }
