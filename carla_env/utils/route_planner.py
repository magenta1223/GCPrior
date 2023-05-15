import carla
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner


class CustomGlobalRoutePlanner(GlobalRoutePlanner):
    def compute_direction_velocities(
        self,
        origin: carla.Location,
        velocity: carla.Vector3D,
        destination: carla.Location,
    ):
        if self._graph is None:
            raise RuntimeError("Graph is not initialized yet.")

        node_list = super()._path_search(
            origin=origin, destination=destination
        )

        origin_xy = np.array([origin.x, origin.y])
        velocity_xy = np.array([velocity.x, velocity.y])

        first_node_xy = np.array(self._graph.nodes[node_list[1]]["vertex"][:2])

        target_direction_vector = np.subtract(first_node_xy, origin_xy)
        target_unit_vector = np.array(target_direction_vector) / np.linalg.norm(
            target_direction_vector
        )

        vel_s = np.dot(velocity_xy, target_unit_vector)

        unit_velocity = velocity_xy / (np.linalg.norm(velocity_xy) + 1e-8)
        angle = np.arccos(np.clip(np.dot(unit_velocity, target_unit_vector), -1.0, 1.0))
        vel_perp = np.linalg.norm(velocity_xy) * np.sin(angle)
        return vel_s, vel_perp

    def compute_distance(self, origin, destination):
        if self._graph is None:
            raise RuntimeError("Graph is not initialized yet.")

        node_list = super()._path_search(origin, destination)
        first_node_xy = self._graph.nodes[node_list[0]]["vertex"]

        distances = [
            super()._distance_heuristic(prev_node, next_node)
            for prev_node, next_node in zip(node_list[:-1], node_list[1:])
        ]
        distances = [
            np.linalg.norm(
                np.subtract(
                    np.array([origin.x, origin.y, 0.0]), np.array(first_node_xy)
                )
            )
        ] + distances
        return np.sum(distances)
