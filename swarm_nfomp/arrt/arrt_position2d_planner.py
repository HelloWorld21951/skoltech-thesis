from swarm_nfomp.arrt.arrt_planner import ARRTPlanner
from swarm_nfomp.planner.planner import Path, State
from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.position_array2d import PositionArray2D


class ARRTPosition2DPlanner(ARRTPlanner[Position2D]):
    def _path_from_list(self, array: list) -> Path[State]:
        return PositionArray2D.from_list(array)

    def _state_from_vector(self, vector) -> State:
        return Position2D.from_vec(vector)
