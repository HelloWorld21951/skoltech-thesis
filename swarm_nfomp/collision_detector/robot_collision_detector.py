from typing import List, Dict

import numpy as np
import shapely.affinity
from shapely import Polygon, MultiPolygon

from swarm_nfomp.planner.planner import CollisionDetector
from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.position_array2d import PositionArray2D


class RobotCollisionDetector(CollisionDetector[Position2D]):
    def __init__(self, inside_rectangle_region: MultiPolygon, outside_rectangle_region: Polygon,
                 robot_shape: Polygon, collision_step: float):
        self.inside_rectangle_region = inside_rectangle_region
        self.outside_rectangle_region = outside_rectangle_region
        self.robot_shape = robot_shape
        self.collision_step = collision_step
        self.number_collision_between_checks = 0

    @staticmethod
    def affine_transform(position: Position2D):
        matrix = position.as_matrix()
        return [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]

    def transformed_robot_shape(self, robot_position) -> Polygon:
        return shapely.affinity.affine_transform(self.robot_shape, self.affine_transform(robot_position))

    def is_collision(self, robot_position: Position2D) -> bool:
        shape: Polygon = self.transformed_robot_shape(robot_position)
        if self.inside_rectangle_region.intersects(shape):
            return True
        if not self.outside_rectangle_region.contains(shape):
            return True
        return False

    def is_collision_for_array(self, robot_positions: List[Position2D]) -> np.array:
        return np.array([self.is_collision(robot_position) for robot_position in robot_positions])

    def is_collision_between(self, point1, point2, first_position_free=True):
        self.number_collision_between_checks += 1
        points_count = int(point1.distance(point2) / self.collision_step) + 1
        points = PositionArray2D.interpolate(point1, point2, points_count)
        if not first_position_free:
            return np.any(self.is_collision_for_array(points))
        return np.any(self.is_collision_for_array(points[1:]))

    def steer(self, point1, point2, distance):
        self.number_collision_between_checks += 1
        point2 = point1.steer(point2, distance)
        points_count = int(point1.distance(point2) / self.collision_step) + 1
        points = PositionArray2D.interpolate(point1, point2, points_count)
        collisions = self.is_collision_for_array(points)
        nonzero = np.nonzero(collisions)[0]
        if len(nonzero) == 0:
            index = -1
        elif nonzero[0] == 0:
            index = 0
        else:
            index = nonzero[0] - 1
        return points[index]

    @classmethod
    def from_dict(cls, data: Dict):
        inside_region = MultiPolygon([Polygon(p) for p in data["inside_polygon"]])
        outside_region = Polygon(data["outside_polygon"])
        robot_shape = Polygon(data["robot_shape"])
        return cls(inside_region, outside_region, robot_shape, data["collision_step"])
