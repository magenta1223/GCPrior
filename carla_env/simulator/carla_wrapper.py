from typing import Generic, TypeVar

T = TypeVar("T")


class CarlaWrapper(Generic[T]):
    def __init__(self, carla_obj: T):
        self.__carla_obj = carla_obj

    @property
    def carla(self) -> T:
        return self.__carla_obj
