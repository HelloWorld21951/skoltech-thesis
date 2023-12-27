from typing import Tuple

import numpy as np

from swarm_nfomp.arrt.a_star import GridPlannerConfig
from swarm_nfomp.grid_planner.eggs_grid_planner import EGGSGridPlanner
from swarm_nfomp.grid_planner.grid_planner import GridPlannerTask


class EGGSARRTAdapter(EGGSGridPlanner):
    def __init__(self, parameters: GridPlannerConfig):
        super().__init__()
        self._collision_function = None
        self._parameters = parameters
        self._offset = None
        self._grid_shape = None
        self._is_goal_reached = False

    def setup(self, start: Tuple[int, int], goal: Tuple[int, int], collision_function):
        min_x = min(start[0], goal[0]) - self._parameters.padding_x
        max_x = max(start[0], goal[0]) + self._parameters.padding_x
        min_y = min(start[1], goal[1]) - self._parameters.padding_y
        max_y = max(start[1], goal[1]) + self._parameters.padding_y
        self._offset = (min_x, min_y)
        self._grid_shape = (max_x - min_x + 1, max_y - min_y + 1)
        planner_task = GridPlannerTask(
            start_point=self.eggs_state_from_arrt_state(start),
            goal_point=self.eggs_state_from_arrt_state(goal),
            grid=np.zeros(self._grid_shape, dtype=bool),
        )
        self.set_planner_task(planner_task)
        self._collision_function = collision_function
        self._is_goal_reached = False

    def step(self):
        current_node = None
        while len(self._open_node_list) > 0:
            current_node = self._step()
            if current_node is None:
                return None
            if self._is_goal_reached:
                return None
            if current_node == self._goal:
                self._is_goal_reached = True
            parent_node = self._parents[current_node]
            if parent_node is None:
                arrt_parent_node = None
            else:
                arrt_parent_node = self.arrt_state_from_eggs_state(parent_node)
            arrt_current_node = self.arrt_state_from_eggs_state(current_node)
            return arrt_current_node, arrt_parent_node

    def eggs_state_from_arrt_state(self, arrt_state):
        return arrt_state[0] - self._offset[0], arrt_state[1] - self._offset[1]

    def arrt_state_from_eggs_state(self, eggs_state):
        return eggs_state[0] + self._offset[0], eggs_state[1] + self._offset[1]

    def _is_valid(self, x, y, parent_x, parent_y):
        is_inside = 0 <= x < self.planner_task.grid.shape[0] and 0 <= y < self.planner_task.grid.shape[1]
        if not is_inside:
            return False
        state = self.arrt_state_from_eggs_state((x, y))
        parent_state = self.arrt_state_from_eggs_state((parent_x, parent_y))
        return not self._collision_function(state, parent_state)
