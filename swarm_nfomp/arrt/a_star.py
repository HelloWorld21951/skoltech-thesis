import dataclasses
import heapq
from typing import Tuple, Optional, Dict

import numpy as np


@dataclasses.dataclass
class GridPlannerConfig:
    padding_x: int
    padding_y: int
    grid_planner_type: str


class AStar(object):
    def __init__(self, parameters: GridPlannerConfig):
        self._start = None
        self._goal = None
        self._closed = set()
        self._opened = []
        self._cost = {}
        self._parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self._collision_function = None
        self._is_goal_reached = False
        self._parameters = parameters

    def setup(self, start: Tuple[int, int], goal: Tuple[int, int], collision_function):
        self._start = start
        self._goal = goal
        self._closed = set()
        self._opened = [(0, start)]
        self._cost = {start: 0}
        self._parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        self._collision_function = collision_function
        self._is_goal_reached = False

    def step(self):
        while len(self._opened) > 0:
            current_state = heapq.heappop(self._opened)[1]
            if current_state in self._closed:
                continue
            if self._is_goal_reached:
                return None
            if current_state == self._goal:
                self._is_goal_reached = True
            # add to closed dict so that we don't expand this node again.
            self._closed.add(current_state)
            states = self._get_successors(current_state)
            for state in states:
                new_cost = self._cost[current_state] + 1
                if new_cost < self._cost.get(state, np.inf):
                    self._cost[state] = new_cost
                    self._parent[state] = current_state
                    total_cost = new_cost + self._calculate_heuristic(state)
                    heapq.heappush(self._opened, (total_cost, state))
            return current_state, self._parent[current_state]

    def _get_successors(self, state):
        successor_states = [
            (state[0] - 1, state[1]),
            (state[0], state[1] - 1),
            (state[0] + 1, state[1]),
            (state[0], state[1] + 1)
        ]
        result = []
        for successor_state in successor_states:
            if successor_state not in self._closed and self._check_bounds(state) and not self._collision_function(state,
                                                                                                                  successor_state):
                result.append(successor_state)
        return result

    def _calculate_heuristic(self, state):
        return np.abs(state[0] - self._goal[0]) + np.abs(state[1] - self._goal[1])
        # return np.sqrt((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2)

    @staticmethod
    def _check_bounds(state):
        return 0 <= state[0]
