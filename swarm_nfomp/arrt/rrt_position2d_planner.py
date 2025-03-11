import dataclasses
import numpy as np
from typing import List
from copy import deepcopy

from swarm_nfomp.arrt.rrt_planner import RRTPlanner
from swarm_nfomp.arrt.rrt_star_planner import RRTStarPlanner
from swarm_nfomp.collision_detector.robot_collision_detector import RobotCollisionDetector
from swarm_nfomp.planner.planner import Bounds, PlannerTask, Path, State
from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.rectangle_bounds import RectangleBounds2D
from swarm_nfomp.utils.position_array2d import PositionArray2D
from swarm_nfomp.utils.math import wrap_angles


@dataclasses.dataclass
class RectangleBoundsWithAngle2D(Bounds[Position2D]):
    max_x: float
    min_x: float
    max_y: float
    min_y: float

    def sample_random_point(self) -> Position2D:
        return Position2D(
            np.random.uniform(self.min_x, self.max_x),
            np.random.uniform(self.min_y, self.max_y),
            np.random.uniform(-np.pi, np.pi)
        )

    @classmethod
    def from_rectangle_bounds_2d(cls, bounds_2d: RectangleBounds2D):
        return cls(bounds_2d.max_x, bounds_2d.min_x, bounds_2d.max_y, bounds_2d.min_y)


@dataclasses.dataclass
class Position2DPathSegment:
    start: Position2D
    end: Position2D

    @property
    def length(self):
        return self.start.distance(self.end)


@dataclasses.dataclass
class Position2DPlannerPath(Path[Position2D]):
    _positions: List[Position2D]

    @property
    def positions(self) -> List[Position2D]:
        return self._positions

    @property
    def segments(self) -> List[Position2DPathSegment]:
        return [Position2DPathSegment(self.positions[i], self.positions[i + 1])
                for i in range(len(self.positions) - 1)]

    @property
    def length(self) -> float:
        return sum(seg.length for seg in self.segments)

    def add_intermediate_positions(self, total_positions_in_path: int) -> None:
        current_path_size = len(self._positions)
        if total_positions_in_path < current_path_size:
            raise ValueError("Current path size is bigger than desired")

        if total_positions_in_path == current_path_size:
            return

        positions_to_fill = total_positions_in_path - 1

        new_positions = []
        for seg in self.segments:
            number_of_positions = int(
                np.round(total_positions_in_path * seg.length / self.length))
            number_of_positions = min(positions_to_fill, number_of_positions)
            positions_to_fill -= number_of_positions
            positions_on_segment = np.linspace(
                seg.start.as_vec(), seg.end.as_vec(), number_of_positions, endpoint=False)
            angle_diff = wrap_angles(seg.end.angle - seg.start.angle)
            positions_on_segment[:, 2] = seg.start.angle + wrap_angles(
                np.linspace(0, angle_diff, number_of_positions, endpoint=False))
            new_positions.extend([Position2D.from_vec(pos)
                                 for pos in positions_on_segment])

        for _ in range(positions_to_fill + 1):
            new_positions.append(self._positions[-1])

        self._positions = new_positions


@dataclasses.dataclass
class Position2DPlannerTask(PlannerTask[Position2D]):
    start: Position2D
    goal: Position2D
    collision_detector: RobotCollisionDetector
    bounds: RectangleBoundsWithAngle2D


class RRTPosition2DPlanner(RRTPlanner[Position2D]):
    def _path_from_list(self, array: list) -> Position2DPlannerPath:
        return Position2DPlannerPath(array)


class RRTStarPosition2DPlanner(RRTStarPlanner[Position2D]):
    def _path_from_list(self, array: list) -> Position2DPlannerPath:
        return Position2DPlannerPath(array)
