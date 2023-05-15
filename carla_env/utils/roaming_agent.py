import carla

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.tools.misc import compute_magnitude_angle, is_within_distance_ahead


class LocalPlannerModified(LocalPlanner):
    def __del__(self):
        pass  # otherwise it deletes our vehicle object

    def run_step(self):
        return super().run_step(
            debug=False
        )  # otherwise by default shows waypoints, that interfere with our camera


class RoamingAgent(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, follow_traffic_lights=True):
        """
        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgent, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._local_planner = LocalPlannerModified(self._vehicle)
        self._follow_traffic_lights = follow_traffic_lights

    def run_step(self):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, _ = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        traffic_light_color = self._is_light_red(lights_list)
        if traffic_light_color == "RED" and self._follow_traffic_lights:
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()

        return control, traffic_light_color

    # override case class
    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.
        Only suitable for Towns 03 -- 07.
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        traffic_light_color = "NONE"  # default, if no traffic lights are seen

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if (
                object_waypoint.road_id != ego_vehicle_waypoint.road_id
                or object_waypoint.lane_id != ego_vehicle_waypoint.lane_id
            ):
                continue

            if is_within_distance_ahead(
                traffic_light.get_transform(),
                self._vehicle.get_transform(),
                self._proximity_threshold,
            ):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return "RED"
                elif traffic_light.state == carla.TrafficLightState.Yellow:
                    traffic_light_color = "YELLOW"
                elif traffic_light.state == carla.TrafficLightState.Green:
                    if traffic_light_color is not "YELLOW":  # (more severe)
                        traffic_light_color = "GREEN"
                else:
                    import pdb

                    pdb.set_trace()
                    # investigate https://carla.readthedocs.io/en/latest/python_api/#carlatrafficlightstate

        return traffic_light_color

    # override case class
    def _is_light_red_us_style(self, lights_list, debug=False):
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        traffic_light_color = "NONE"  # default, if no traffic lights are seen

        if ego_vehicle_waypoint.is_junction:
            # It is too late. Do not block the intersection! Keep going!
            return "JUNCTION"

        if self._local_planner.target_waypoint is not None:
            if self._local_planner.target_waypoint.is_junction:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(
                        loc,
                        ego_vehicle_location,
                        self._vehicle.get_transform().rotation.yaw,
                    )
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print(
                            "=== Magnitude = {} | Angle = {} | ID = {}".format(
                                sel_magnitude, min_angle, sel_traffic_light.id
                            )
                        )

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return "RED"
                    elif (
                        self._last_traffic_light.state == carla.TrafficLightState.Yellow
                    ):
                        traffic_light_color = "YELLOW"
                    elif (
                        self._last_traffic_light.state == carla.TrafficLightState.Green
                    ):
                        if traffic_light_color is not "YELLOW":  # (more severe)
                            traffic_light_color = "GREEN"
                    else:
                        import pdb

                        pdb.set_trace()
                        # investigate https://carla.readthedocs.io/en/latest/python_api/#carlatrafficlightstate
                else:
                    self._last_traffic_light = None

        return traffic_light_color
