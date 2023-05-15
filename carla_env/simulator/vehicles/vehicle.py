from typing import List, Optional, Type, TypeVar

import carla
from typing_extensions import override

from carla_env.simulator.actor import Actor
from carla_env.simulator.sensors.sensor import Sensor
from carla_env.utils.logger import Logging

logger = Logging.get_logger(__name__)

T = TypeVar("T", bound=Sensor)


class Vehicle(Actor[carla.Vehicle]):
    def init(self):
        super().init()
        self.__sensors = []

    def autopilot(self):
        self.carla.set_autopilot(True)

    def disable_autopilot(self):
        self.carla.set_autopilot(False)

    def get_control(self) -> carla.VehicleControl:
        return self.carla.get_control()

    def get_physic_control(self) -> carla.VehiclePhysicsControl:
        return self.carla.get_physics_control()

    def apply_control(self, control: carla.VehicleControl):
        self.carla.apply_control(control)

    def apply_physics_control(self, physics_control: carla.VehiclePhysicsControl):
        self.carla.apply_physics_control(physics_control)

    def stop(self):
        self.velocity = carla.Vector3D(0, 0, 0)
        self.angular_velocity = carla.Vector3D(0, 0, 0)
        self.apply_control(carla.VehicleControl())

    @property
    def velocity(self) -> carla.Vector3D:
        return super().velocity

    @velocity.setter
    def velocity(self, velocity: carla.Vector3D):
        self.carla.set_target_velocity(velocity)

    @property
    def angular_velocity(self) -> carla.Vector3D:
        return super().angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: carla.Vector3D):
        self.carla.set_target_angular_velocity(angular_velocity)

    @property
    def sensors(self) -> List[Sensor]:
        return self.__sensors

    def add_sensor(self, sensor_type: Type[T], **kwargs) -> Optional[T]:
        sensor = sensor_type.spawn(self.simulator, parent=self, **kwargs)
        if sensor is None:
            return None
        self.attach_sensor(sensor)
        return sensor

    def attach_sensor(self, sensor: Sensor):
        self.__sensors.append(sensor)

    @override
    def before_destroy(self):
        for sensor in self.__sensors:
            sensor.destroy()
        return super().before_destroy()
