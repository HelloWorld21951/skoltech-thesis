import dataclasses
from typing import Optional, Generic, Any

import numpy as np

from swarm_nfomp.planner.planner import Planner, State, Path


@dataclasses.dataclass
class TreeNode(Generic[State]):
    point: State
    parent: Any
    index: int


class Tree(Generic[State]):
    def __init__(self, node: TreeNode[State]):
        self.root = node
        self.points = node.point.as_numpy()[None]
        self.nodes = [node]

    def add_point(self, point: State, parent: TreeNode[State]):
        node = TreeNode(point, parent, self.points.shape[0])
        self.points = np.vstack((self.points, point.as_numpy()))
        self.nodes.append(node)
        return node

    def nearest_node(self, point: State) -> TreeNode:
        distances = np.linalg.norm(self.points - point.as_numpy(), axis=1)
        index = np.argmin(distances)
        return self.nodes[index]


@dataclasses.dataclass
class RRTParameters:
    iterations: int
    steer_distance: float
    goal_point_probability: float
    goal_threshold: float


class RRTPlanner(Planner[State]):
    def __init__(self, parameters: RRTParameters):
        super().__init__()
        self.tree: Optional[Tree[State]] = None
        self._parameters = parameters

    def setup(self):
        self.tree = Tree(TreeNode(self.planner_task.start, None, 0))

    def plan(self) -> Path[State]:
        for i in range(self._parameters.iterations):
            random_point = self._sample_random_point_with_goal_point()
            nearest_node = self.tree.nearest_node(random_point)
            new_point = nearest_node.point.steer(random_point, self._parameters.steer_distance)
            if not self.planner_task.collision_detector.is_collision_between(nearest_node.point, new_point):
                self.tree.add_point(new_point, nearest_node)
                if new_point.distance(self.planner_task.goal) < self._parameters.goal_threshold:
                    self.is_goal_reached = True
        return self._calculate_path()

    def _sample_random_point_with_goal_point(self):
        if np.random.uniform() < self._parameters.goal_point_probability:
            return self.planner_task.goal
        return self._sample_random_point()

    def _sample_random_point(self):
        bounds = self.planner_task.bounds
        return bounds.sample_random_point()

    def _calculate_path(self) -> Path[State]:
        node = self.tree.nearest_node(self.planner_task.goal)
        path = [node.point]
        index = node.index
        while index != 0:
            node = node.parent
            path.append(node.point)
            index = node.index
        return self._path_from_list(path[::-1])

    def _path_from_list(self, array: list) -> Path[State]:
        raise NotImplementedError()
