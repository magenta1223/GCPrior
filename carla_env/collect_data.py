"""Collect data from Carla simulator.

example call:
./PythonAPI/util/config.py --map Town01 --delta-seconds 0.05
python PythonAPI/carla/agents/navigation/data_collection_agent.py \
    --vision_size 256 --vision_fov 90 --steps 10000 --weather --lights
"""

import datetime
import time
from pathlib import Path
from typing import Any, List, Optional

# import flax
import numpy as np
from matplotlib.pyplot import step

from carla_env.dataset import Dataset, dump_dataset
from carla_env.simulator import Simulator
from carla_env.utils.config import ExperimentConfigs
from carla_env.utils.logger import Logging
from carla_env.utils.roaming_agent import RoamingAgent

logger = Logging.get_logger(__name__)
# Params = flax.core.FrozenDict[str, Any]


def collect_data(config: ExperimentConfigs):
    """Collect data from CARLA simulator.
    
    Args:
        config (ExperimentConfigs): Experiment configs.
        
    Raises:
        ValueError: Raises an error if carla_ip is None.
    """
    if config.carla_ip is None:
        print("Please pass your carla IP address")
        return

    env = Simulator(config)
    env.reset()

    record_dir = create_record_dirpath(env, config.data_path)
    record_path = record_dir / "record"
    record_path.mkdir(parents=True, exist_ok=True)

    record_dirname_per_weather = record_dir / "record" / config.weather
    record_dirname_per_weather.mkdir(parents=True, exist_ok=True)

    try:
        total_step = 0
        for j in range(12000):
            observations_sensor: List[np.ndarray] = []
            observations_image: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            rewards: List[float] = []
            terminals: List[bool] = []
            infos: List[dict] = []

            logger.info("EPISODE: %s (%s/1,000,000)", j, format(total_step, ","))

            env.reset()
            obs, _, _, _ = env.step()
            observations_sensor.append(obs["sensor"][config.lidar.num_theta_bin:])
            # observations_image.append(obs["image"].copy())

            agent = RoamingAgent(
                env.ego_vehicle.carla,
                follow_traffic_lights=config.lights,
            )
            # pylint: disable=protected-access
            agent._local_planner.set_global_plan(env.route_manager.waypoints)

            done = False
            sum_step_fps = 0
            while not done:
                t = time.time()
                control, _ = agent.run_step()
                action = np.array([control.throttle - control.brake, control.steer])
                next_obs, reward, done, info = env.step(action)

                control = np.array([control.throttle, control.steer, control.brake])
                observations_sensor.append(next_obs["sensor"][config.lidar.num_theta_bin:])
                # observations_image.append(next_obs["image"].copy())
                actions.append(control.copy())
                rewards.append(reward)
                terminals.append(done)
                infos.append(info)

                sum_step_fps += 1 / (time.time() - t)
                if env.steps % 50 == 0:
                    print(f"FPS: {sum_step_fps / 50:.2f}")
                    sum_step_fps = 0

            env.ego_vehicle.stop()

            total_step += env.steps

            if total_step > config.max_total_steps:
                logger.info("Finished collecting data")
                break

            dataset: Dataset = {
                "observations": {
                    "sensor": np.array(observations_sensor),
                    "image": np.array([]),
                },
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "terminals": np.array(terminals),
                "infos": infos,
                "lidar_bin": config.lidar.num_theta_bin,
            }

            if infos[-1]["done"]["dist_done"]:
                filename = record_dir / f"episode_{j}.pkl"
            else:
                filename = record_dir / f"episode_{j}_failed.pkl"
            dump_dataset(dataset, filename)
    finally:
        env.finish()


def create_record_dirpath(sim: Simulator, base_dir: Optional[Path] = None):
    """Create a directory path to save the collected data.
    
    Returns:
        Path: Path to the directory to save the collected data.
        
    Example:
        carla_data/carla-town01-224x224-fov90-1k-2020-05-20-15-00-00
    """
    if base_dir is None:
        base_dir = Path.cwd() / "carla_data"
    now = datetime.datetime.now()

    params = (
        "carla",
        sim.world.map.name.lower().split("/")[-1],
        f"{sim.config.vision_size}x{sim.config.vision_size}",
        f"fov{sim.config.vision_fov}",
        f"{sim.config.frame_skip}" if sim.config.frame_skip > 1 else "",
        "multiagent" if sim.config.multiagent else "",
        "lights" if sim.config.lights else "",
        f"{sim.config.max_steps // 1000}k",
        now.strftime("%Y-%m-%d-%H-%M-%S"),
    )
    return base_dir / "-".join(x for x in params if x)
