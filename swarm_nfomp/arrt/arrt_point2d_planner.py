import dataclasses
from typing import Tuple

import numpy as np

from swarm_nfomp.arrt.a_star import AStar
from swarm_nfomp.arrt.rrt_planner import RRTParameters
from swarm_nfomp.arrt.rrt_point2d_planner import RRTPoint2DPlanner
from swarm_nfomp.utils.point2d import Point2D


@dataclasses.dataclass
class ARRTPoint2DPlannerParameters(RRTParameters):
    a_star_iterations: int
    a_star_side_count: int


class ARRTPoint2DPlanner(RRTPoint2DPlanner):
    def __init__(self, parameters: ARRTPoint2DPlannerParameters):
        super().__init__(parameters)
        self._parameters = parameters
        self._random_target_point = None
        self._random_plane_point = None
        self._start_point = None
        self._x_astar_side_size = None
        self._y_astar_side_size = None

    def plan(self):
        for i in range(self._parameters.iterations):
            self.step()
        return self._calculate_path()

    def step(self):
        self._random_target_point = self._sample_random_point_with_goal_point()
        self._random_plane_point = self._sample_random_point()
        nearest_node = self.tree.nearest_node(self._random_target_point)
        self._start_point = nearest_node.point
        self.transform_random_plane_point()
        self.calculate_side_sizes()
        node_dict = {(0, 0): nearest_node}
        a_star_planner = AStar((0, 0), (self._x_astar_side_size, 0), self.a_star_collision_function)
        a_star_planner.step()
        for i in range(self._parameters.a_star_iterations):
            result = a_star_planner.step()
            if result is None:
                break
            current_state, parent_state = result
            parent_node = node_dict[parent_state]
            node = self.tree.add_point(self._point2d_from_location(current_state), parent_node)
            node_dict[current_state] = node

    def transform_random_plane_point(self):
        delta = self._random_target_point.as_numpy() - self._start_point.as_numpy()
        delta_plane = self._random_plane_point.as_numpy() - self._start_point.as_numpy()
        projection = delta * np.sum(delta * delta_plane) / (np.linalg.norm(delta) ** 2 + 1e-6)
        point = self._start_point.as_numpy() + delta_plane - projection
        self._random_plane_point = Point2D(point[0], point[1])

    def a_star_collision_function(self, point_location1: Tuple[int, int], point_location2: Tuple[int, int]):
        point1 = self._point2d_from_location(point_location1)
        point2 = self._point2d_from_location(point_location2)
        return not self._is_path_free(point1, point2)

    def _point2d_from_location(self, current_state):
        x = self._start_point.x + (self._random_target_point.x - self._start_point.x) * current_state[
            0] / self._x_astar_side_size + (self._random_plane_point.x - self._start_point.x) * \
            current_state[1] / self._y_astar_side_size
        y = self._start_point.y + (self._random_target_point.y - self._start_point.y) * current_state[
            0] / self._x_astar_side_size + (self._random_plane_point.y - self._start_point.y) * \
            current_state[1] / self._y_astar_side_size
        return Point2D(x, y)

    def calculate_side_sizes(self):
        delta = self._random_target_point.as_numpy() - self._start_point.as_numpy()
        delta_plane = self._random_plane_point.as_numpy() - self._start_point.as_numpy()
        self._x_astar_side_size = int(np.linalg.norm(delta) / self._parameters.steer_distance) + 1
        self._y_astar_side_size = int(np.linalg.norm(delta_plane) / self._parameters.steer_distance) + 1
