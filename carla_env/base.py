import abc
import asyncio
import math
import time
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import carla
import gym
import gym.spaces
import numpy as np
import pygame
from typing_extensions import Literal

from agents.tools.misc import is_within_distance_ahead
from carla_env.dataset import Dataset, load_datasets
from carla_env.simulator.actor import Actor
from carla_env.simulator.simulator import Simulator
from carla_env.utils.carla_sync_mode import CarlaSyncMode
from carla_env.utils.config import ExperimentConfigs
from carla_env.utils.roaming_agent import RoamingAgent


class BaseCarlaEnvironment(abc.ABC, gym.Env[dict, np.ndarray]):
    """Base Carla Environment.

    This class is the base class for all Carla environments. It provides the basic
    functionality to connect to a Carla server, spawn a vehicle, and control it. It also
    provides the basic functionality to record the data from the sensors.

    Args:
        config: Experiment configs.
        image_model: Image model to be used for image processing.
        weather: Weather to be used in the environment.
        carla_ip: IP address of the Carla server.
        carla_port: Port of the Carla server.
    """

    OBS_IDX = {
        "control": np.array([0, 1, 2]),
        "acceleration": np.array([3, 4, 5]),
        "angular_velocity": np.array([6, 7, 8]),
        "location": np.array([9, 10, 11]),
        "rotation": np.array([12, 13, 14]),
        "forward_vector": np.array([15, 16, 17]),
        "veloctiy": np.array([18, 19, 20]),
        "target_location": np.array([21, 22, 23]),
    }

    def __init__(self, config: ExperimentConfigs):
        self.config = config

        self.sim = Simulator(config)

        #  dataset
        self.data_path = config.data_path

        ## Collision detection
        self._proximity_threshold = 10.0
        self._traffic_light_threshold = 5.0

        self.reset_simulator()

        #  sync mode
        self.sync_mode = CarlaSyncMode(
            self.sim.world, self.sim.ego_vehicle.lidar_sensor, fps=20
        )

        self.actor_list = self.sim.world.get_actors()
        self.vehicle_list = self.sim.world.get_vehicles()
        self.lights_list = self.sim.world.get_traffic_lights()

    def reset_simulator(self):
        self.sim.reset()

    def reset(self):
        self.reset_simulator()

        # self.weather.tick()
        self.agent = RoamingAgent(
            self.sim.ego_vehicle.carla,
            follow_traffic_lights=self.config.lights,
        )
        # pylint: disable=protected-access
        self.agent._local_planner.set_global_plan(
            self.sim.route_manager.waypoints
        )

        self.count = 0

        obs, _, _, _ = self.step()
        return obs

    def seed(self, seed: int):
        return seed

    def compute_action(
        self,
    ) -> Tuple[
        carla.VehicleControl,
        Union[Tuple[Literal[True], carla.Actor], Tuple[Literal[False], None]],
    ]:
        return self.agent.run_step()

    def step(
        self,
        action: Optional[np.ndarray] = None,
        traffic_light_color: Optional[str] = "",
    ) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        rewards: List[np.ndarray] = []
        next_obs, done, info = None, None, None
        for _ in range(self.config.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(
                action, traffic_light_color
            )
            rewards.append(reward)

            if done:
                break

        if next_obs is None or done is None or info is None:
            raise ValueError("frame_skip >= 1")
        return next_obs, np.mean(rewards), done, info

    def _is_map_hazard(self) -> Union[
        Tuple[Literal[True], float, Actor], Tuple[Literal[False], float, None]
    ]:
        """
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        if self.sim.ego_vehicle.collision_sensor.has_collided:
            return True, -1.0, self.sim.ego_vehicle.collision_sensor.object

        return False, 0.0, None

    def _get_trafficlight_trigger_location(
        self, traffic_light: Actor[carla.TrafficLight]
    ):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point: carla.Vector3D, radians: float):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.transform
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.carla.trigger_volume.location)
        area_ext = traffic_light.carla.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)

    def _is_light_red(self):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.
        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self.sim.ego_vehicle.location
        ego_vehicle_waypoint = self.sim.world.map.get_waypoint(ego_vehicle_location)

        for traffic_light in self.lights_list:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.sim.world.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(
                object_waypoint.transform,
                self.sim.ego_vehicle.transform,
                self._traffic_light_threshold,
            ):
                if traffic_light.carla.state == carla.TrafficLightState.Red:
                    return True, -0.1, traffic_light

        return False, 0.0, None

    def _get_traffic_light_reward(self):
        traffic_light_hazard, _, _ = self._is_light_red()
        return traffic_light_hazard, 0.0

    def _get_collision_reward(self):
        map_hazard, reward, _ = self._is_map_hazard()
        return map_hazard, reward

    def get_distance_vehicle_target(self):
        vehicle_location = self.sim.vehicle_location
        target_location = self.sim.target_location
        return np.linalg.norm(
            np.array(
                [
                    vehicle_location.x - target_location.x,
                    vehicle_location.y - target_location.y,
                    vehicle_location.z - target_location.z,
                ]
            )
        )

    def goal_reaching_reward(self):
        # Now we will write goal_reaching_rewards
        target_location = self.sim.target_location

        # This is the distance computation
        """
        dist = self.route_planner.compute_distance(vehicle_location, target_location)

        base_reward = -1.0 * dist
        collided_done, collision_reward = self._get_collision_reward(vehicle)
        traffic_light_done, traffic_light_reward = self._get_traffic_light_reward(vehicle)
        object_collided_done, object_collided_reward = self._get_object_collided_reward(vehicle)
        total_reward = base_reward + 100 * collision_reward + 100 * traffic_light_reward + 100.0 * object_collided_reward
        """
        # dist = self.route_planner.compute_distance(vehicle_location, target_location)
        vel_forward, vel_perp = self.sim.route_manager.compute_direction_velocities(
            self.sim.ego_vehicle, target_location
        )

        # print('[GoalReachReward] VehLoc: %s Target: %s Dist: %s VelF:%s' % (str(vehicle_location), str(target_location), str(dist), str(vel_forward)))
        # base_reward = -1.0 * (dist / 100.0) + 5.0
        base_reward = vel_forward

        # Redifine base reward to be the L2 distance between the vehicle and the target
        base_reward = self.get_distance_vehicle_target()

        _, traffic_light_reward = self._get_traffic_light_reward()
        collided_done, collision_reward = self._get_collision_reward()
        total_reward: np.ndarray = (
            base_reward + 100 * collision_reward
        )  # + 100 * traffic_light_reward + 100.0 * object_collided_reward

        lane_invasion = self.sim.ego_vehicle.lane_invasion_sensor.lane_types
        lane_done = (
            carla.LaneMarkingType.Solid in lane_invasion
            or carla.LaneMarkingType.SolidSolid in lane_invasion
        )

        reward_dict = {
            "collision": collision_reward,
            "traffic_light": traffic_light_reward,
            "base_reward": base_reward,
            "vel_forward": vel_forward,
            "vel_perp": vel_perp,
        }

        done_dict = {
            "collided_done": collided_done,
            "lane_collided_done": lane_done,
            "traffic_light_done": False,
            "reached_max_steps": self.count >= self.config.max_steps,
        }

        return total_reward, reward_dict, done_dict

    def _simulator_step(
        self,
        action: Optional[np.ndarray] = None,
        traffic_light_color: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, bool, Dict[str, Any]]:
        raise NotImplementedError

    def finish(self):
        self.sim.finish()
        pygame.quit()

    def get_dataset(self) -> List[Dataset]:
        if self.data_path is None or not self.data_path.exists():
            return []
        return list(load_datasets(self.data_path))
