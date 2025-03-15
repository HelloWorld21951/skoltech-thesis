from typing import List, Dict
from copy import deepcopy

import trimesh

from swarm_nfomp.planner.planner import CollisionDetector
from swarm_nfomp.utils.position3d import Position3D


class Robot3DCollisionDetector(CollisionDetector[Position3D]):
    def __init__(self, inside_rectangle_region: List[trimesh.Trimesh], outside_rectangle_region: trimesh.path.Path3D,
                 robot_shape: trimesh.Trimesh, collision_step: float):
        self._outside_bounds = outside_rectangle_region.bounds
        self._collision_manager = trimesh.collision.CollisionManager()
        for i in range(len(inside_rectangle_region)):
            self._collision_manager.add_object(f"inside_obstacle_{i}")
        self._collision_step = collision_step
        self._robot_shape = robot_shape

    def is_collision_between(self, state1: Position3D, state2: Position3D, first_position_free=True):
        number_of_positions = int(state1.distance(state2) / self._collision_step)
        positions = state1.interpolate(state2, number_of_positions)
        if first_position_free:
            positions = positions[1:]

        for position in positions:
            robot = deepcopy(self._robot_shapes[i])
            robot.apply_transform(position.tf)
            if not trimesh.bounds.contains(bounds=self._outside_bounds, points=robot.bounds) or \
                self._collision_manager.in_collision_single(robot):
                return True
        return False

    @classmethod
    def from_dict(cls, data: Dict) -> 'Robot3DCollisionDetector':
        outside_region = trimesh.primitives.Box(bounds=data["outside_polygon"]).as_outline()
        inside_region = [trimesh.creation.box(bounds=p) for p in data["inside_polygon"]]
        robot_shape = trimesh.creation.box(bounds=data["robot_shape"])
        return cls(inside_region, outside_region, robot_shape, data["collision_step"])
