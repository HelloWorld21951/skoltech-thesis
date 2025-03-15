import dataclasses
import numpy as np
from typing import List

from swarm_nfomp.arrt.rrt_planner import RRTPlanner
from swarm_nfomp.collision_detector.robot_3d_collision_detector import Robot3DCollisionDetector
from swarm_nfomp.planner.planner import PlannerTask, Path
from swarm_nfomp.utils.position3d import Position3D
from swarm_nfomp.utils.box_bounds_3d import BoxBounds3D
from swarm_nfomp.utils.position_array3d import PositionArray3D


@dataclasses.dataclass
class Position3DPathSegment:
    start: Position3D
    end: Position3D

    @property
    def length(self):
        return self.start.distance(self.end)


@dataclasses.dataclass
class Position3DPlannerPath(Path[Position3D]):
    _positions: PositionArray3D

    @property
    def positions(self) -> List[Position3D]:
        return self._positions.as_list()

    @property
    def segments(self) -> List[Position3DPathSegment]:
        return [Position3DPathSegment(self.positions[i], self.positions[i + 1])
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
            positions_on_segment = PositionArray3D.interpolate(seg.start, seg.end, number_of_positions).as_list()
            new_positions.extend(positions_on_segment[:-1])

        for _ in range(positions_to_fill + 1):
            new_positions.append(self._positions[-1])

        self._positions = PositionArray3D.from_list(new_positions)


@dataclasses.dataclass
class Position3DPlannerTask(PlannerTask[Position3D]):
    start: Position3D
    goal: Position3D
    collision_detector: Robot3DCollisionDetector
    bounds: BoxBounds3D


class RRTPosition3DPlanner(RRTPlanner[Position3D]):
    def _path_from_list(self, array: list) -> Position3DPlannerPath:
        return Position3DPlannerPath(array)
