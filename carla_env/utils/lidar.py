from typing import List

import numpy as np

from carla_env.utils.coordinates import cart2pol


def generate_lidar_bin(lidar_sensor, num_theta_bin: float, env_range: float):
    # Format rl lidar
    lidar = np.frombuffer(lidar_sensor.raw_data, dtype=np.float32).reshape((-1, 4))

    # (x,y,z) to (min_dist,theta,z)
    lidar_x = lidar[:, 0]
    lidar_y = lidar[:, 1]
    lidar_z = lidar[:, 2]

    lidar_x, lidar_y = cart2pol(lidar_x, lidar_y)
    lidar_cylinder: np.ndarray = np.vstack((lidar_x, lidar_y, lidar_z)).T

    lidar_bin: List[np.ndarray] = []
    empty_cnt = 0
    # discretize theta
    for i in range(-1 * int(num_theta_bin / 2), int(num_theta_bin / 2)):
        low_deg = 2 * i * np.pi / num_theta_bin
        high_deg = 2 * (i + 1) * np.pi / num_theta_bin
        points = lidar_cylinder[
            (lidar_cylinder[:, 1] > low_deg) * (lidar_cylinder[:, 1] < high_deg)
        ][:, 0]

        if not points.any():
            # print(f'{i} ~ {i+1} bin is empty')
            empty_cnt += 1
            lidar_bin.append(np.array([env_range]))
        else:
            max_idx = points.argmax()  # standard (x,y) or (x,y,z)
            lidar_bin.append(
                lidar_cylinder[
                    (lidar_cylinder[:, 1] > low_deg)
                    * (lidar_cylinder[:, 1] < high_deg)
                ][max_idx][0]
            )

    return lidar_bin
