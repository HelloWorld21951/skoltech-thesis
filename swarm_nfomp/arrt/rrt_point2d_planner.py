import dataclasses

from swarm_nfomp.arrt.rrt_planner import RRTPlanner
from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector
from swarm_nfomp.utils.point2d import Point2D
from swarm_nfomp.utils.rectangle_bounds import RectangleBounds2D


@dataclasses.dataclass
class Point2DPlannerTask:
    start: Point2D
    goal: Point2D
    collision_detector: PointArrayCollisionDetector
    bounds: RectangleBounds2D


class RRTPoint2DPlanner(RRTPlanner[Point2D]):
    pass
