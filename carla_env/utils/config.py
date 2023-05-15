from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml


@dataclass
class LidarConfigs:
    upper_fov: float = 5.
    lower_fov: float = -30.
    rotation_frequency: float = 20.
    max_range: float = 20.
    num_theta_bin: int = 80
    dropoff_general_rate: float = .1
    dropoff_intensity_limit: float = .2
    dropoff_zero_intensity: float = .2
    points_per_second: float = 120_000


@dataclass
class VisualizationConfigs:
    """Arguments for visualization."""

    draw_path: bool = False
    """Whether to draw the path."""

    draw_velocity: bool = False
    """Whether to draw the velocity."""

    draw_fps: bool = False
    """Whether to draw the FPS."""


@dataclass
class ExperimentConfigs:
    """Arguments for running env.py."""

    vision_size: int = 224
    """Size of the vision sensor."""

    vision_fov: int = 90
    """Field of view of the vision sensor."""

    weather: str = "ClearNoon"
    """Weather to use. Example: ClearNoon"""

    frame_skip: int = 1
    """Number of frames to skip."""

    multiagent: bool = False
    """Whether to use multi-agent."""

    num_vehicles: int = 0
    """Number of vehicles. Only used when multiagent is True."""

    lane: int = 0
    lights: bool = False
    mode: str = "ours"

    num_routes: int = 0
    """Number of routes to use."""

    routes: List[Tuple[int, int]] = field(default_factory=list)
    """List of routes to use"""

    vehicle_type: str = "audi.a2"
    """Type of vehicle to use. Example: audi.a2"""

    random_route: bool = False
    """Whether to use random route."""

    max_steps: int = 3000
    """Maximum number of steps per episode."""

    max_total_steps: int = 1_000_000
    """Maximum number of total steps."""

    carla_ip: str = "localhost"
    """IP address of the carla server."""

    data_path: Optional[Path] = None
    """Path to the data directory to save the episode data and logs."""

    lidar: LidarConfigs = field(default_factory=LidarConfigs)
    """Lidar configurations."""

    fps: int = 30
    """FPS of the simulator."""

    visual: VisualizationConfigs = field(default_factory=VisualizationConfigs)
    """Visualization configurations."""


def check_route_list(routes: Any) -> List[Tuple[int, int]]:
    """Check if the route list is valid.

    Args:
        route_str (str): String of routes

    Returns:
        List[List[int]]: List of routes

    Raises:
        ValueError: If the route string is not valid
    """
    if not isinstance(routes, list):
        raise ValueError("Invalid route list. Must be a list of lists or tuples.")
    for route in routes:
        if not isinstance(route, (list, tuple)):
            raise ValueError("Invalid route list. Must be a list of lists or tuples.")
        if len(route) != 2:
            raise ValueError(
                "Invalid route list. Each element must be a list or tuple of length 2."
            )
        for route_id in route:
            if not isinstance(route_id, int):
                raise ValueError("Invalid route list. Each element must be an integer.")
    return routes


def check_is_ip(ip: str) -> str:
    """Check if the ip is valid.

    Args:
        ip (str): IP address

    Returns:
        str: Returns the ip string if the ip is valid, raises exception otherwise

    Raises:
        ValueError: If the ip is not valid
    """
    def checker(ip: str) -> bool:
        if ip == "localhost":
            return True
        ip_split = ip.split(".")
        if len(ip_split) != 4:
            return False
        for i in ip_split:
            if not i.isdigit():
                return False
            i = int(i)
            if i < 0 or i > 255:
                return False
        return True

    if not checker(ip):
        raise ValueError("Invalid IP address")
    return ip


def parse_config(filename: Union[str, Path]):
    """Parse arguments.

    Return:
        EnvArguments: Arguments for running env.py
    """
    with open(filename, "r") as f:
        config = yaml.safe_load(f)

        # Check if the config is valid
        if "carla_ip" in config:
            if config["carla_ip"] is None:
                config["carla_ip"] = "localhost"
            else:
                config["carla_ip"] = check_is_ip(config["carla_ip"])
        if "data_path" in config and config["data_path"] is not None:
            config["data_path"] = Path(config["data_path"])
        if "routes" in config:
            config["routes"] = check_route_list(config["routes"])

        if "lidar" in config:
            config["lidar"] = LidarConfigs(**config["lidar"])
        if "visual" in config:
            config["visual"] = VisualizationConfigs(**config["visual"])

        return ExperimentConfigs(**config)
