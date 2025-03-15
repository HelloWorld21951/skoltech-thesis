from typing import List

import numpy as np

from swarm_nfomp.collision_detector.multi_robot_3d_collision_detector import MultiRobot3DCollisionDetector
from swarm_nfomp.utils.math import wrap_angles
from swarm_nfomp.utils.metric_manager import MetricManager
from swarm_nfomp.utils.position_array3d import PositionArray3D
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp import MultiRobot3DResultPath


class WarehouseNfompMetricCalculator:
    def __init__(self, metric_manager):
        self._metrics_manager: MetricManager = metric_manager
        self._length_truncation_threshold = 0.1

    def calculate_metrics(self, path: MultiRobot3DResultPath, detector: MultiRobot3DCollisionDetector):
        self.calculate_collision_free_states(path, detector)
        self.calculate_total_time(path)
        self.calculate_angle_difference_rmse(path)
        self.calculate_total_distance(path)
        self.calculate_truncated_aol(path)

    def calculate_collision_free_states(self, path: MultiRobot3DResultPath, detector: MultiRobot3DCollisionDetector):
        robot_positions = [PositionArray3D.from_vec(x) for x in path.numpy_positions]
        number_collision_free_states = 0
        for robot_position in robot_positions:
            is_collision = detector.is_collision(robot_position)
            if not is_collision:
                number_collision_free_states += 1
        self._metrics_manager.add_metric("Number of collision free states", number_collision_free_states)

    def calculate_total_time(self, path: MultiRobot3DResultPath):
        points = path.numpy_positions[:, :, :2]
        distances = np.linalg.norm(points[1:] - points[:-1], axis=2)
        total_time = np.sum(np.max(distances, axis=1))
        self._metrics_manager.add_metric("Total time (s)", total_time)

    def calculate_angle_difference_rmse(self, path):
        angles = path.numpy_positions[:, :, 3:]
        angle_differences = wrap_angles(angles[1:] - angles[:-1])
        self._metrics_manager.add_metric("Angle difference RMSE (rad)",
                                         np.sqrt(np.mean(angle_differences ** 2)))

    def calculate_total_distance(self, path):
        points = path.numpy_positions[:, :, :3]
        distances = np.linalg.norm(points[1:] - points[:-1], axis=2)
        total_distance = np.sum(distances)
        self._metrics_manager.add_metric("Total distance (m)", total_distance)

    def calculate_truncated_aol(self, path):
        points = path.numpy_positions[:, :, :3]
        distances = np.linalg.norm(points[1:] - points[:-1], axis=2)
        truncated_distances = np.clip(distances, self._length_truncation_threshold, None)
        angles = path.numpy_positions[:, :, 3:]
        angle_differences = np.sum(np.abs(wrap_angles(angles[1:] - angles[:-1])), axis=2)
        truncated_aol = np.mean(angle_differences / truncated_distances)
        self._metrics_manager.add_metric("Truncated Angle over Length (AOL) (rad / m)", truncated_aol)