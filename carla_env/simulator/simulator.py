import random
from typing import Dict, List, Optional, Tuple

import carla
import gym
import gym.spaces
import numpy as np
from typing_extensions import override

from carla_env.utils.config import ExperimentConfigs
from carla_env.utils.lidar import generate_lidar_bin
from carla_env.utils.logger import Logging
from carla_env.utils.vector import to_array

logger = Logging.get_logger(__name__)


class Simulator(gym.Env[dict, np.ndarray]):
    """The simulator of the environment. This class is responsible for creating the
    client and the world of the simulator.

    Args:
        config (ExperimentConfigs): The experiment configurations.

    """

    def __init__(self, config: ExperimentConfigs):
        from carla_env.simulator.client import Client
        from carla_env.simulator.route_manager import RouteManager
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
        from carla_env.simulator.visualizer import Visualizer

        self.__config = config

        self.__client = Client(self, config.carla_ip, 2000 - config.num_routes * 5)

        self.__world = self.__client.world
        self.__world.weather = getattr(carla.WeatherParameters, config.weather)
        self.__fps = config.fps

        self.__route_manager = RouteManager(world=self.__world, config=config)

        self.__is_multi_agent = config.multiagent
        self.__num_auto_vehicles = config.num_vehicles

        self.__auto_vehicles: Optional[List[AutoVehicle]] = None
        self.__ego_vehicle: Optional[EgoVehicle] = None

        self.__visualizer = Visualizer(self, config)

        self.action_space = gym.spaces.Box(shape=(2,), low=-1, high=1)
        self.observation_space = gym.spaces.Dict(
            {
                "sensor": gym.spaces.Box(
                    shape=(config.lidar.num_theta_bin + 24,), low=-1, high=1
                ),
                "image": gym.spaces.Box(
                    shape=(224, 224, 3), low=0, high=255, dtype=np.uint8
                ),
            }
        )

    @override
    def reset(self):
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
        from carla_env.utils.carla_sync_mode import CarlaSyncMode

        # Destroy the auto vehicles.
        if self.__auto_vehicles is not None:
            for auto_vehicle in self.__auto_vehicles:
                auto_vehicle.destroy()

        # Spawn the ego vehicle.
        if self.__ego_vehicle is None:
            self.__ego_vehicle = None
            while self.__ego_vehicle is None:
                self.route_manager.select_route()
                self.__ego_vehicle = EgoVehicle.spawn(
                    simulator=self,
                    config=self.__config,
                    initial_transform=self.route_manager.initial_transform,
                )
            self.__sync_mode = CarlaSyncMode(
                self.world, self.ego_vehicle.lidar_sensor, fps=self.__fps
            )
        else:
            self.route_manager.select_route()
            self.ego_vehicle.reset()
            self.ego_vehicle.transform = self.route_manager.initial_transform

        logger.info("Vehicle starts at: %s", to_array(self.ego_vehicle.location))

        # Spawn the auto vehicles.
        if self.is_multi_agent:
            self.__auto_vehicles = [
                AutoVehicle.spawn(simulator=self)
                for _ in range(self.__num_auto_vehicles)
            ]

        self.__steps = 0
        self.__prev_reward = None

        return self.step()[0]

    @override
    def step(self, action: Optional[np.ndarray] = None):
        self.__steps += 1

        if action is not None:
            acc = float(action[0])
            throttle = max(acc, 0)
            brake = -min(acc, 0)
            steer = float(action[1])
            brake = brake if brake > 0.01 else 0
        else:
            throttle = 0
            brake = 0
            steer = 0

        self.ego_vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=0,
            )
        )

        _, lidar_sensor = self.__sync_mode.tick(timeout=10)
        lidar_bin = generate_lidar_bin(
            lidar_sensor,
            self.__config.lidar.num_theta_bin,
            self.__config.lidar.max_range,
        )

        reward, reward_dict, done_dict = calculate_reward(self, self.__prev_reward)
        self.__prev_reward = reward_dict
        next_observation = {
            "lidar": np.array(lidar_bin),
            "control": np.array([throttle, brake, steer]),
            **self.ego_vehicle.get_observation(),
            "target_location": to_array(self.route_manager.target_transform.location),
        }
        info = {
            "reward": reward_dict,
            "total_reward": reward,
            "done": done_dict,
            "control_repeat": self.config.frame_skip,
            "weather": self.config.weather,
            "settings_map": self.world.map.name,
            "settings_multiagent": self.config.multiagent,
            "traffic_lights_color": "UNLABELED",
        }
        done = any(done_dict.values())

        if self.steps % 50 == 0 or done:
            logger.info("Step: %s", self.steps)
            logger.info("Vehicle: %s", next_observation["location"])
            logger.info("Target: %s", next_observation["target_location"])
            logger.info("Reward: %s (%s)", reward, reward_dict)
            logger.info(
                "Done: %s",
                next(filter(lambda x: x[1], done_dict.items()))[0] if done else False,
            )

        if done_dict["reached_max_steps"]:
            logger.warning("Episode reached max steps. Terminating episode.")

        return (
            {
                "sensor": np.hstack(list(next_observation.values())),
                "image": self.ego_vehicle.camera.image,
            },
            reward,
            done,
            info,
        )

    @override
    def render(self):
        return super().render()

    def finish(self):
        """Finish the simulator."""
        if self.__ego_vehicle and self.__ego_vehicle.is_alive:
            self.__ego_vehicle.destroy()

        if self.__auto_vehicles:
            for auto_vehicle in self.__auto_vehicles:
                if auto_vehicle.is_alive:
                    auto_vehicle.destroy()

    @property
    def client(self):
        """The client of the simulator."""
        return self.__client

    @property
    def world(self):
        """The world of the simulator."""
        return self.__world

    @property
    def route_manager(self):
        """The route manager of the simulator."""
        return self.__route_manager

    @property
    def ego_vehicle(self):
        """The ego vehicle of the simulator."""
        if not self.__ego_vehicle:
            raise ValueError("Ego vehicle is not initialized. Call reset() first.")
        return self.__ego_vehicle

    @property
    def vehicle_location(self):
        """The location of the ego vehicle."""
        return self.ego_vehicle.location

    @property
    def target_location(self):
        """The target location of the ego vehicle."""
        return self.route_manager.target_transform.location

    @property
    def is_multi_agent(self):
        """Whether the simulator is multi-agent."""
        return self.__is_multi_agent

    @property
    def num_auto_vehicles(self):
        """The number of vehicles. If the simulator is not multi-agent, this value is
        0."""
        return self.__num_auto_vehicles if self.is_multi_agent else 0

    @property
    def auto_vehicles(self):
        """The auto vehicles of the simulator."""
        if self.is_multi_agent and not self.__auto_vehicles:
            raise ValueError("Auto vehicles are not initialized. Call reset() first.")
        return self.__auto_vehicles

    @property
    def fps(self):
        """The fps of the simulator."""
        return self.__fps

    @property
    def steps(self):
        """The number of steps."""
        return self.__steps

    @property
    def config(self):
        """The config of the simulator."""
        return self.__config


def calculate_reward(
    simulator: Simulator, prev_reward: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float], Dict[str, bool]]:
    def get_collision_reward_done():
        if simulator.ego_vehicle.collision_sensor.has_collided:
            return -1, True
        return 0, False

    def get_dist_reward_done(): # sprase reward
        dist = simulator.ego_vehicle.distance(simulator.target_location)
        return -dist, dist < 5

    def get_vel_forward_perp_reward():
        return simulator.route_manager.compute_direction_velocities(
            simulator.ego_vehicle, simulator.target_location
        )

    def get_lane_invasion_done():
        lane_invasion = simulator.ego_vehicle.lane_invasion_sensor.lane_types
        return (
            carla.LaneMarkingType.Solid in lane_invasion
            or carla.LaneMarkingType.SolidSolid in lane_invasion
        )

    def get_reached_map_steps():
        return simulator.steps >= simulator.config.max_steps

    collision_reward, collision_done = get_collision_reward_done()
    dist_reward, dist_done = get_dist_reward_done()
    vel_forward_reward, vel_perp_reward = get_vel_forward_perp_reward()
    lane_invasion_done = get_lane_invasion_done()
    reached_max_steps = get_reached_map_steps()

    if prev_reward:
        prev_dist = prev_reward["dist"]
    else:
        prev_dist = -simulator.route_manager.initial_transform.location.distance(
            simulator.route_manager.target_transform.location
        )
    base_reward = dist_reward - prev_dist
    vel_reward = simulator.ego_vehicle.velocity.length() - 20
    total_reward = base_reward + vel_reward + 100 * collision_reward

    reward_dict = {
        "base_reward": base_reward,
        "collision": collision_reward,
        "dist": dist_reward,
        "vel": vel_reward,
        "vel_forward": vel_forward_reward,
        "vel_perp": vel_perp_reward,
    }

    done_dict = {
        "dist_done": dist_done,
        "collided_done": collision_done,
        "lane_collided_done": lane_invasion_done,
        "reached_max_steps": reached_max_steps,
    }

    return total_reward, reward_dict, done_dict
