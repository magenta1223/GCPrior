import numpy as np
from contextlib import contextmanager
from carla_env.simulator.simulator import *
import carla
from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
from carla_env.utils.config import parse_config
from carla_env.utils.config import ExperimentConfigs



class Location(carla.Location):
    def __init__(self, transform : carla.Transform, name = None, *args, **kwargs):
        self.__init_data__ = np.array([transform.location.x, transform.location.y, transform.location.z])
        self.name = name
        super().__init__( *self.__init_data__  )
        
    def __array__(self):
        loc = np.array([self.x, self.y, self.z])    
        # if np.array_equal(loc, self.__init_data__):
            # print("SMAE !! ")
        return loc
    def __repr__(self):

        if self.name is not None:
            return f"{self.name} {np.array(self).round(2)}"
        else:
            return f"CARLA LOCS {np.array(self).round(2)}"

class EgoVehicle_XY(EgoVehicle):
    def get_observation(self):
        return {
            "acceleration": to_array(self.acceleration)[:2],
            "angular_velocity": to_array(self.angular_velocity)[2],
            "location": to_array(self.location)[:2],
            "rotation": to_array(self.rotation)[2],
            # "forward_vector": to_array(self.rotation.get_forward_vector())[:2],
            "velocity": to_array(self.velocity)[:2],
        }


class CARLA_Task:
    def __init__(self, coordinates) -> None:
        self.target_location = Location(*coordinates, 0)


class CARLA_GC(Simulator):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, config: ExperimentConfigs):
        super().__init__(config)
        self.task = CARLA_Task([0,0])

    @override
    def reset(self):
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
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
                # observes only x and y
                self.__ego_vehicle = EgoVehicle_XY.spawn(
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
    
    @contextmanager
    def set_task(self, task):
        if type(task) != CARLA_Task:
            raise TypeError(f'task should be CARLA_Task but {type(task)} is given')

        prev_task = self.task

        self.task = task
        yield
        self.task = prev_task


    def reset(self):
        return super().reset()


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


        reward, reward_dict, done_dict = calculate_reward(self, self.__prev_reward)
        self.__prev_reward = reward_dict
        next_observation = {
            "control": np.array([throttle, brake, steer]),
            **self.ego_vehicle.get_observation(),
            # "target_location": to_array(self.route_manager.target_transform.location),
            "target_location": to_array(self.target_location),

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

        return np.hstack(list(next_observation.values())), reward, done, info


    @property
    def target_location(self):
        return self.task.target_location


carla_config = parse_config("./configs/data_collecting.yaml")



# for simpl
meta_train_tasks = np.array([
    # [5,6,0,3],
    # [5,0,1,3],
    # [5,1,2,4],
    # [6,0,2,4],
    # [5,0,4,1],
    # [6,1,2,3],
    # [5,6,3,0],
    # [6,2,3,0],
    # [5,6,0,1],
    # [5,6,3,4],
    # [5,0,3,1],
    # [6,0,2,1],
    # [5,6,1,2],
    # [5,6,2,4],
    # [5,0,2,3],
    # [6,0,1,2],
    # [5,2,3,4],
    # [5,0,1,4],
    # [6,0,3,4],
    # [0,1,3,2],
    # [5,6,2,3],
    # [6,0,1,4],
    # [0,1,2,3]
])


tasks = np.array([115, 144, 105, 76, 32, 30, 28, 137, 108, 106, 120])


# kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])
# KITCHEN_TASKS = kitchen_subtasks[tasks]
# KITCHEN_META_TASKS = kitchen_subtasks[meta_train_tasks]

CARLA_TASKS = None
