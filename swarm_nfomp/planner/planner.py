import dataclasses
from typing import TypeVar, Generic, List

import numpy as np

State = TypeVar('State')


class CollisionDetector(Generic[State]):
    def is_collision_between(self, state: State, state2: State) -> np.ndarray:
        raise NotImplementedError()


class Bounds(Generic[State]):
    def sample_random_point(self) -> State:
        raise NotImplementedError()


@dataclasses.dataclass
class PlannerTask(Generic[State]):
    start: State
    goal: State
    collision_detector: CollisionDetector[State]
    bounds: Bounds[State]


class Path(Generic[State]):
    @classmethod
    def from_list(cls, array: List[State]):
        raise NotImplementedError()


class Planner(Generic[State]):
    def __init__(self):
        self.planner_task = None
        self.is_goal_reached = False

    def set_planner_task(self, task: PlannerTask[State]):
        self.planner_task = task
        self.is_goal_reached = False
        self.setup()

    def setup(self):
        raise NotImplementedError()

    def plan(self) -> Path[State]:
        raise NotImplementedError()
