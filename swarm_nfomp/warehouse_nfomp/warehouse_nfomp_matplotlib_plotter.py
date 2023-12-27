import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from swarm_nfomp.collision_detector.multi_robot_collision_detector import MultiRobotCollisionDetector
from swarm_nfomp.plotting.plotting import show_multipolygon, show_position_array2d_with_mask
from swarm_nfomp.utils.position_array2d import PositionArray2D
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp import MultiRobotResultPath, MultiRobotPathPlannerTask


class WarehouseNfompMatplotlibPlotter:
    def __init__(self):
        self._fig = plt.figure(dpi=200)

    def show(self, planner_task: MultiRobotPathPlannerTask, result: MultiRobotResultPath):
        self._fig.clear()
        ax = self._fig.gca()
        ax.set_aspect('equal', adjustable='box')
        bounds = planner_task.bounds
        ax.set_xlim(bounds.min_x, bounds.max_x)
        ax.set_ylim(bounds.min_y, bounds.max_y)
        self.show_robot_collision_detector(planner_task.collision_detector)
        self.show_robot_paths(planner_task.collision_detector.robot_shapes, result, planner_task.collision_detector)
        plt.pause(0.01)

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fig.savefig(filename)

    def show_robot_collision_detector(self, collision_detector: MultiRobotCollisionDetector):
        show_multipolygon(collision_detector.inside_rectangle_region, self._fig, color="black")

    def show_robot_paths(self, robot_shapes, result: MultiRobotResultPath, detector: MultiRobotCollisionDetector):
        robot_positions: List[PositionArray2D] = [PositionArray2D.from_vec(x) for x in result.numpy_positions]
        is_collision = np.zeros((len(robot_positions), len(robot_positions[0])), dtype=bool)
        for i in range(len(robot_positions)):
            is_collision[i] = detector.is_collision_for_each_robot(robot_positions[i])
        for is_collision, path in zip(is_collision.T, result.robot_paths):
            show_position_array2d_with_mask(path, self._fig, is_collision, color="black",
                                            color1="red", color2="green")
