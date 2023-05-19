import numpy as np
from contextlib import contextmanager
from carla_env.simulator.simulator import *
import carla
from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
from carla_env.utils.config import parse_config
from carla_env.utils.config import ExperimentConfigs



# class Location(carla.Location):
#     def __init__(self, position = None, transform = None, name = None, *args, **kwargs):
        
#         assert position is not None or transform is not None, "No input"
#         assert position is None or transform is None, "Input duplicated"
        
#         if position is not None:
#             self.__init_data__ = position
#             self.name = name
#         else:
#             # here?
#             self.__init_data__ = np.array([transform.location.x, transform.location.y, transform.location.z])
#             self.name = name
#         super().__init__( *self.__init_data__  )
        
#     def __array__(self):
#         loc = np.array([self.x, self.y, self.z])    
#         # if np.array_equal(loc, self.__init_data__):
#             # print("SMAE !! ")
#         return loc
#     def __repr__(self):

#         if self.name is not None:
#             return f"{self.name} {np.array(self).round(2)}"
#         else:
#             return f"CARLA LOCS {np.array(self).round(2)}"

# class EgoVehicle_XY(EgoVehicle):
#     def get_observation(self):
#         return {
#             "acceleration": to_array(self.acceleration)[:2],
#             "angular_velocity": to_array(self.angular_velocity)[2],
#             "location": to_array(self.location)[:2],
#             "rotation": to_array(self.rotation)[2],
#             # "forward_vector": to_array(self.rotation.get_forward_vector())[:2],
#             "velocity": to_array(self.velocity)[:2],
#         }


class CARLA_Task:
    def __init__(self, index):
        self.target_location = index
    
    def __repr__(self):
        return f"CARLA Task Index : {self.target_location}"


class CARLA_GC(Simulator):
    """
    Goal Conditioned Environment
    """
    render_width = 400
    render_height = 400
    render_device = -1

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __init__(self, config: ExperimentConfigs):
        from carla_env.simulator.client import Client
        from carla_env.simulator.route_manager import RouteManager
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle

        # super().__init__(config)

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

        self.__visualizer = None

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


        self.spawn_points = self.__world.map.get_spawn_points()

        self.init_transform = self.spawn_points[134]

        self.task = CARLA_Task(to_carla_transform([0. ,0. ,0. ,0. ,0. ,0. ]))
        # self.initial_loc = carla.Location(*[0,0,0])

        # init_point = np.array([  32.04544449,   13.27302933,    0.59999996, 0.0, -179.84078979,    0.0 ])
        # self.init_transform = to_carla_transform(init_point)

    @contextmanager
    def set_task(self, task):
        if type(task) != CARLA_Task:
            raise TypeError(f'task should be CARLA_Task but {type(task)} is given')

        prev_task = self.task
        # prev_init_loc = self.init_transform

        self.task = task
        # self.initial_loc =  carla.Transform()

        yield

        # self.initial_loc = prev_init_loc
        self.task = prev_task


    def reset(self):
        from carla_env.simulator.vehicles.auto_vehicle import AutoVehicle
        from carla_env.simulator.vehicles.ego_vehicle import EgoVehicle
        from carla_env.utils.carla_sync_mode import CarlaSyncMode
        from carla_env.simulator.visualizer import Visualizer

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
                    # initial_transform=self.init_transform,

                )
            self.__sync_mode = CarlaSyncMode(
                self.world, self.ego_vehicle.lidar_sensor, fps=self.__fps
            )
            self.__visualizer = Visualizer(self, self.config)
        else:
            # print(self.route_manager.initial_transform)
            self.route_manager.select_route()
            self.ego_vehicle.reset()
            self.ego_vehicle.transform = self.route_manager.initial_transform
            # self.ego_vehicle.transform = self.init_transform
            # self.ego_vehicle.location = self.initial_loc.location

            # self.ego_vehicle.set_transform(self.initial_loc)

        
        # print("Vehicle starts at: %s", to_array(self.ego_vehicle.location))
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

    def step(self, action: Optional[np.ndarray] = None):
        # print(self.target_location)
        self.__steps += 1

        # if action is not None:
        #     acc = float(action[0])
        #     throttle = max(acc, 0)
        #     brake = -min(acc, 0)
        #     steer = float(action[1])
        #     brake = brake if brake > 0.01 else 0
        # else:
        #     throttle = 0
        #     brake = 0
        #     steer = 0

        if action is not None and len(action) == 2:
            acc = float(action[0])
            throttle = max(acc, 0)
            brake = -min(acc, 0)
            steer = float(action[1])
            brake = brake if brake > 0.01 else 0
        elif action is not None and len(action) == 3:
            acc = float(action[0])
            throttle, brake, steer = np.array(action).tolist()
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


        vehicle_obs = self.ego_vehicle.get_observation()

        reward, reward_dict, done_dict = calculate_reward(self, self.__prev_reward)
        self.__prev_reward = reward_dict
        next_observation = {
            "control": np.array([throttle, brake, steer]),
            **vehicle_obs,
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
        # done = any(done_dict.values())

        del done_dict['lane_collided_done']
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

        
        if done_dict['dist_done']:
            reward = 100
        else:
            reward = 0

        # dist = self.ego_vehicle.distance(self.target_location)


        # print(to_array(self.ego_vehicle.location).round(2), to_array(self.ego_vehicle.location).round(2))
    
        
        # vehicle_loc = to_array(self.ego_vehicle.location).round(2)
        # target_loc = to_array(self.target_location).round(2)
        
        info['vehicle_loc'] = to_array(self.ego_vehicle.location).round(2)
        info['target_loc'] = to_array(self.target_location).round(2)
        info['vel'] = np.linalg.norm(vehicle_obs['velocity'])



        return np.hstack(list(next_observation.values())), reward, done, info


    @property
    def target_location(self):
        # return self.task.target_location
        return self.route_manager.target_transform

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
        # return self.route_manager.target_transform.location
        return self.spawn_points[self.task.target_location].location

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



carla_config = parse_config("./configs/learning.yaml")



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


# tasks = np.array([115, 144, 105, 76, 32, 30, 28, 137, 108, 106, 120])


# tasks = np.array([86, 79, 2, 120, 4])

tasks = np.array([
    138, # circulate 
    86, # corner  
    8, # right-left (not in data)  
    70, # 70 
    ])


# tasks = [99]


# spawn_points = np.load("./LVD/data/carla/spawn_points.npy")

# kitchen_subtasks = np.array(['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle'])
# KITCHEN_TASKS = kitchen_subtasks[tasks]
# KITCHEN_META_TASKS = kitchen_subtasks[meta_train_tasks]



def to_carla_transform(p):
    loc, rot = carla.Location(*p[:3]), carla.Rotation(*p[3:])
    return carla.Transform(loc, rot)


# CARLA_TASKS = [  CARLA_Task( Location(position = t))   for t in spawn_points[tasks]]
# CARLA_META_TASKS = CARLA_TASKS = [  to_carla_transform(p)  for p in spawn_points[tasks]]


CARLA_META_TASKS = CARLA_TASKS = tasks