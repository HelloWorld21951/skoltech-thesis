from typing import Dict, List
from copy import deepcopy

import numpy as np

import trimesh

from swarm_nfomp.utils.position3d import Position3D
from swarm_nfomp.utils.position_array3d import PositionArray3D


class MultiRobot3DCollisionDetector:
    def __init__(self, inside_rectangle_region: List[trimesh.Trimesh], outside_rectangle_region: trimesh.path.Path3D,
                 robot_shapes: List[trimesh.Trimesh]):
        self._outside_bounds = outside_rectangle_region.bounds
        self._collision_manager = trimesh.collision.CollisionManager()
        for i in range(len(inside_rectangle_region)):
            self._collision_manager.add_object(f"inside_obstacle_{i}", inside_rectangle_region[i])
        self._robot_shapes = robot_shapes

    def transformed_robot_shapes(self, states: PositionArray3D) -> List[trimesh.Trimesh]:
        assert len(states) == len(self._robot_shapes)
        return [deepcopy(robot_shape).apply_transform(state.tf) for robot_shape, state in zip(self._robot_shapes, states)]

    def _are_robots_in_bounds(self, robots: List[trimesh.Trimesh]) -> np.ndarray:
        res = np.array([True for _ in robots])
        for i in range(len(robots)):
            if np.all(trimesh.bounds.contains(bounds=self._outside_bounds, points=robots[i].bounds)):
                res[i] = False
        return res

    def is_collision(self, states: PositionArray3D) -> bool:
        transformed_robot_shapes = self.transformed_robot_shapes(states)
        for i in range(len(transformed_robot_shapes)):
            self._collision_manager.add_object(f"robot_position_{i}", transformed_robot_shapes[i])

        is_collision = not np.all(self._are_robots_in_bounds(transformed_robot_shapes))
        if not is_collision:
            is_collision |= self._collision_manager.in_collision_internal()
        
        for i in range(len(transformed_robot_shapes)):
            self._collision_manager.remove_object(f"robot_position_{i}")

        return is_collision
    
    def is_collision_for_each_robot(self, states: PositionArray3D) -> np.ndarray:
        transformed_robot_shapes = self.transformed_robot_shapes(states)
        res = self._are_robots_in_bounds(transformed_robot_shapes)

        for i in range(len(transformed_robot_shapes)):
            self._collision_manager.add_object(f"robot_position_{i}", transformed_robot_shapes[i])

        for i in range(len(transformed_robot_shapes)):
            self._collision_manager.remove_object(f"robot_position_{i}")
            res[i] |= self._collision_manager.in_collision_single(transformed_robot_shapes[i])
            self._collision_manager.add_object(f"robot_position_{i}", transformed_robot_shapes[i])
        
        for i in range(len(transformed_robot_shapes)):
            self._collision_manager.remove_object(f"robot_position_{i}")

        return res
    
    def is_collision_for_each_robot_for_list(self, states_list: List[PositionArray3D]) -> np.array:
        return np.array([self.is_collision_for_each_robot(states) for states in states_list])

    @classmethod
    def from_dict(cls, data: Dict) -> 'MultiRobot3DCollisionDetector':
        outside_region = trimesh.primitives.Box(bounds=data["outside_polygon"]).as_outline()
        inside_region = [trimesh.creation.box(bounds=p) for p in data["inside_polygon"]]
        robot_shapes = [trimesh.creation.box(bounds=p) for p in data["robot_shapes"]]
        return cls(inside_region, outside_region, robot_shapes)


