import dataclasses
from abc import ABC
from typing import Optional, Tuple

import numpy as np


@dataclasses.dataclass
class GridPlannerTask:
    start_point: Tuple[int, int]
    goal_point: Tuple[int, int]
    grid: np.ndarray

    @classmethod
    def from_dict(cls, dict_: dict):
        return cls(
            start_point=tuple(dict_["start_point"]),
            goal_point=tuple(dict_["goal_point"]),
            grid=np.array(dict_["grid"]),
        )


class GridPlanner(ABC):
    def __init__(self):
        self.planner_task: Optional[GridPlannerTask] = None
        self.closed_set = set()

    def set_planner_task(self, planner_task: GridPlannerTask):
        self.planner_task = planner_task
        self.closed_set = set()

    def plan(self) -> np.ndarray:
        raise NotImplementedError()
