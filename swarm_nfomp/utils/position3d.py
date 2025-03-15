import trimesh.transformations as tf
import numpy as np

from functools import cached_property
from typing import Iterable, List


class Position3D:
    def __init__(self, translation: Iterable[float], euler_angles: Iterable[float]) -> None:
        self._translation = tf.translation_matrix(translation)
        self._rotation = tf.euler_matrix(*euler_angles)

    @classmethod
    def from_tf(cls, transform: np.ndarray) -> 'Position3D':
        assert transform.shape == (4, 4)
        translation = tf.translation_from_matrix(transform)
        euler_angles = tf.euler_from_matrix(transform)
        return cls(translation, euler_angles)

    @cached_property
    def tf(self) -> np.ndarray:
        return self._translation @ self._rotation
    
    @cached_property
    def translation(self) -> np.ndarray:
        return tf.translation_from_matrix(self._translation)
    
    @cached_property
    def rotation(self) -> np.ndarray:
        return self._rotation[:3, :3]
    
    @cached_property
    def euler_angles(self) -> np.ndarray:
        return np.asarray(tf.euler_from_matrix(self.tf))
    
    def __mul__(self, other: 'Position3D') -> 'Position3D':
        return Position3D.from_tf(self.tf @ other.tf)

    def __sub__(self, other: 'Position3D') -> 'Position3D':
        translation_diff = self.translation - other.translation
        rotation_diff = self.rotation @ np.linalg.inv(other.rotation)
        transform = np.eye(4)
        transform[:3, :3] = rotation_diff
        transform[:3, 3] = translation_diff.T
        return Position3D.from_tf(transform)

    def __repr__(self) -> str:
        return f"Position3D(translation={self.translation}, rotation={self.euler_angles})"
    
    def distance(self, other: 'Position3D') -> float:
        return np.sum((other.translation - self.translation) ** 2)
    
    def steer(self, target: 'Position3D', steer_distance: float) -> 'Position3D':
        distance = self.distance(target)
        if distance <= steer_distance:
            return target
        
        direction = target - self
        return Position3D(
            translation=self.translation + direction.translation * steer_distance / distance,
            euler_angles=self.euler_angles + direction.euler_angles * steer_distance / distance,
        )
    