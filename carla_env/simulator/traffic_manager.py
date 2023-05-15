import carla


class TrafficManager(carla.TrafficManager):
    def __new__(cls, traffic_manager: carla.TrafficManager):
        return traffic_manager

    def reset(self):
        self.set_global_distance_to_leading_vehicle(2.0)
        self.set_synchronous_mode(True)
        self.global_percentage_speed_difference(30.0)
