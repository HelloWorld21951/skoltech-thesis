from dataclasses import dataclass

import numpy as np

from swarm_nfomp.utils.point_array2d import PointArray2D


@dataclass
class Point2D:
    x: float
    y: float

    def as_numpy(self):
        return np.array([self.x, self.y])

    def distance(self, other):
        return np.linalg.norm(self.as_numpy() - other.as_numpy())

    @classmethod
    def from_vec(cls, vec):
        return cls(vec[0], vec[1])

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"])

    def steer(self, point2, steer_distance: float):
        distance = np.linalg.norm(point2.as_numpy() - self.as_numpy())
        if distance < steer_distance:
            return point2
        direction = (point2.as_numpy() - self.as_numpy()) / distance
        return Point2D(self.x + steer_distance * direction[0], self.y + steer_distance * direction[1])
