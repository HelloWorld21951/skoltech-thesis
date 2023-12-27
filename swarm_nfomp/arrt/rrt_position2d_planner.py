import dataclasses

import numpy as np

from swarm_nfomp.arrt.rrt_planner import RRTPlanner
from swarm_nfomp.collision_detector.robot_collision_detector import RobotCollisionDetector
from swarm_nfomp.planner.planner import Bounds, PlannerTask, Path, State
from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.position_array2d import PositionArray2D


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


@dataclasses.dataclass
class Position2DPlannerTask(PlannerTask[Position2D]):
    start: Position2D
    goal: Position2D
    collision_detector: RobotCollisionDetector
    bounds: RectangleBoundsWithAngle2D


class RRTPosition2DPlanner(RRTPlanner[Position2D]):
    def _path_from_list(self, array: list) -> Path[State]:
        return PositionArray2D.from_list(array)
