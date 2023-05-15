from typing import Callable, Generic, List, Optional, Type, TypeVar, Union

import carla
from typing_extensions import TypeGuard, override

from carla_env.simulator.carla_wrapper import CarlaWrapper
from carla_env.simulator.simulator import Simulator
from carla_env.utils.logger import Logging

logger = Logging.get_logger(__name__)

T = TypeVar("T", bound=carla.Actor)


class ActorInitializeError(Exception):
    """Raised when the actor failed to initialize."""


class Actor(Generic[T], CarlaWrapper[T]):
    def __init__(self, simulator: Simulator, actor: T):
        super().__init__(actor)
        self.__simulator = simulator
        self.__on_destroy_callbacks: List[Callable[[], None]] = []
        self.__destroyed = False
        self.__is_projection = True

    def init(self) -> None:
        self.__is_projection = False

    @classmethod
    def spawn(
        cls,
        *args,
        simulator: Simulator,
        blueprint: carla.ActorBlueprint,
        transform: Optional[carla.Transform] = None,
        attach_to: Optional["Actor"] = None,
        **kwargs,
    ):
        if transform is None:
            transform = carla.Transform(
                carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0, pitch=0, roll=0)
            )
        parent = attach_to.carla if attach_to is not None else None
        actor = None
        try:
            actor = simulator.world.carla.spawn_actor(
                blueprint, transform, attach_to=parent
            )
            logger.info("Spawn %s", actor)

            actor = cls(simulator, actor)
            actor.init(*args, **kwargs)
            return actor
        except (RuntimeError, ActorInitializeError) as e:
            logger.error("Failed to spawn %s: %s", blueprint, e)
            if actor:
                actor.destroy()
            return None

    @property
    def simulator(self):
        """The simulator of the actor."""
        return self.__simulator

    @property
    def client(self):
        """The client of the actor."""
        return self.simulator.client

    @property
    def world(self):
        """The world of the actor."""
        return self.simulator.world

    @property
    def is_alive(self) -> bool:
        """Whether the actor is alive."""
        return self.carla.is_alive

    def on_destroy(self, callback: Callable[[], None]) -> None:
        """Add a callback when the actor is destroyed."""
        self.__on_destroy_callbacks.append(callback)

    def before_destroy(self):
        pass

    def after_destroy(self):
        pass

    def destroy(self):
        if self.__destroyed or self.carla and not self.is_alive:
            return

        self.__destroyed = True
        actor_desc = str(self.carla)

        self.before_destroy()

        success = self.carla.destroy()
        if success:
            logger.info("Destroy %s", actor_desc)
        else:
            logger.error("Failed to destroy %s", actor_desc)

        for callback in self.__on_destroy_callbacks:
            callback()

        self.after_destroy()

    @property
    def transform(self) -> carla.Transform:
        """The transform of the actor."""
        return self.carla.get_transform()

    @transform.setter
    def transform(self, transform: carla.Transform) -> None:
        self.carla.set_transform(transform)

    @property
    def location(self) -> carla.Location:
        """The location of the actor."""
        return self.transform.location

    @location.setter
    def location(self, location: carla.Location) -> None:
        self.carla.set_location(location)

    @property
    def rotation(self) -> carla.Rotation:
        """The rotation of the actor."""
        return self.transform.rotation

    @rotation.setter
    def rotation(self, rotation: carla.Rotation) -> None:
        self.transform = carla.Transform(self.transform.location, rotation)

    @property
    def velocity(self) -> carla.Vector3D:
        """The velocity of the actor."""
        return self.carla.get_velocity()

    @property
    def angular_velocity(self) -> carla.Vector3D:
        """The angular velocity of the actor."""
        return self.carla.get_angular_velocity()

    @property
    def acceleration(self) -> carla.Vector3D:
        """The acceleration of the actor."""
        return self.carla.get_acceleration()

    def add_force(self, force: carla.Vector3D) -> None:
        """Add a force to the actor."""
        self.carla.add_force(force)

    def add_torque(self, torque: carla.Vector3D) -> None:
        """Add a torque to the actor."""
        self.carla.add_torque(torque)

    def add_impulse(self, impulse: carla.Vector3D) -> None:
        """Add an impulse to the actor."""
        self.carla.add_impulse(impulse)

    def distance(self, other: Union["Actor", carla.Transform, carla.Location]) -> float:
        """The distance to another actor."""
        if isinstance(other, carla.Location):
            return self.location.distance(other)
        return self.location.distance(other.location)

    def distance_2d(
        self, other: Union["Actor", carla.Transform, carla.Location]
    ) -> float:
        """The distance to another actor."""
        if isinstance(other, carla.Location):
            return self.location.distance_2d(other)
        return self.location.distance_2d(other.location)

    def isinstance(self, actor_type: Type[T]) -> TypeGuard["Actor[T]"]:
        return isinstance(self.carla, actor_type)

    @override
    def __repr__(self):
        return f"{self.carla}"

    def __del__(self):
        if not self.__is_projection:
            self.destroy()
