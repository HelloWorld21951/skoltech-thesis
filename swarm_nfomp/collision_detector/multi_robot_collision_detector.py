import dataclasses
from typing import Dict
from typing import List

import numpy as np
import shapely.affinity
from shapely.geometry import Polygon, MultiPolygon

from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.position_array2d import PositionArray2D


@dataclasses.dataclass
class MultiRobotState:
    positions: PositionArray2D
    shapes: List[Polygon]


class MultiRobotCollisionDetector:
    def __init__(self, inside_rectangle_region: MultiPolygon, outside_rectangle_region: Polygon,
                 robot_shapes: List[Polygon]):
        self.inside_rectangle_region = inside_rectangle_region
        self.outside_rectangle_region = outside_rectangle_region
        self.robot_shapes = robot_shapes

    @staticmethod
    def affine_transform(position: Position2D):
        matrix = position.as_matrix()
        return [matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]

    def transformed_robot_shapes(self, robot_swarm_positions) -> List[Polygon]:
        return [shapely.affinity.affine_transform(robot_shape, self.affine_transform(robot_position))
                for robot_shape, robot_position in zip(self.robot_shapes, robot_swarm_positions)]

    def is_collision(self, robot_swarm_positions: PositionArray2D) -> bool:
        transformed_robot_shapes: List[Polygon] = self.transformed_robot_shapes(robot_swarm_positions)
        for shape in transformed_robot_shapes:
            if self.inside_rectangle_region.intersects(shape):
                return True
            if not self.outside_rectangle_region.contains(shape):
                return True
        for i in range(len(transformed_robot_shapes)):
            for j in range(i + 1, len(transformed_robot_shapes)):
                if transformed_robot_shapes[i].intersects(transformed_robot_shapes[j]):
                    return True
        return False

    def is_collision_for_each_robot(self, robot_swarm_positions: PositionArray2D) -> np.array:
        transformed_robot_shapes: List[Polygon] = self.transformed_robot_shapes(robot_swarm_positions)
        result = np.zeros(len(transformed_robot_shapes), dtype=bool)
        for i in range(len(transformed_robot_shapes)):
            shape = transformed_robot_shapes[i]
            if self.inside_rectangle_region.intersects(shape):
                result[i] = True
            elif not self.outside_rectangle_region.contains(shape):
                result[i] = True
            for j in range(len(transformed_robot_shapes)):
                if j != i and transformed_robot_shapes[i].intersects(transformed_robot_shapes[j]):
                    result[i] = True
                    break
        return np.array(result)

    def is_collision_for_list(self, robot_swarm_positions: List[PositionArray2D]) -> np.array:
        return np.array([self.is_collision(robot_swarm_position) for robot_swarm_position in robot_swarm_positions])

    def is_collision_for_each_robot_for_list(self, robot_swarm_positions: List[PositionArray2D]) -> np.array:
        return np.array(
            [self.is_collision_for_each_robot(robot_swarm_position) for robot_swarm_position in robot_swarm_positions])

    @classmethod
    def from_dict(cls, data: Dict):
        inside_region = MultiPolygon([Polygon(p) for p in data["inside_polygon"]])
        outside_region = Polygon(data["outside_polygon"])
        robot_shapes = [Polygon(p) for p in data["robot_shapes"]]
        return cls(inside_region, outside_region, robot_shapes)


