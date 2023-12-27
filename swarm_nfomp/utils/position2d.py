import numpy as np

from swarm_nfomp.utils.math import wrap_angles


class Position2D:
    def __init__(self, x, y, angle):
        self._x: np.ndarray = x
        self._y: np.ndarray = y
        self._angle: np.ndarray = angle

    @property
    def rotation(self):
        return self._angle

    @property
    def translation(self):
        return np.array([self._x, self._y]).T

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def angle(self):
        return self._angle

    @classmethod
    def from_vec(cls, vec):
        assert isinstance(vec, np.ndarray)
        assert vec.shape[-1] == 3
        assert len(vec.shape) == 1
        return cls(vec[0], vec[1], vec[2])

    def as_vec(self):
        return np.array([self._x, self._y, self._angle]).T

    def as_numpy(self):
        return self.as_vec()

    def _mul_impl(self, other):
        x1 = other.x * np.cos(self._angle) - other.y * np.sin(self._angle) + self._x
        y1 = other.x * np.sin(self._angle) + other.y * np.cos(self._angle) + self._y
        a1 = (other.rotation + self._angle + np.pi) % (2 * np.pi) - np.pi
        return x1, y1, a1

    def __mul__(self, other):
        x1, y1, a1 = self._mul_impl(other)
        return other.__class__(x1, y1, a1)

    def __sub__(self, other):
        x1 = self._x - other.x
        y1 = self._y - other.y
        a1 = wrap_angles(self._angle - other.rotation)
        return np.array([x1, y1, a1])

    def inv(self):
        x = -self.x * np.cos(self.rotation) - self.y * np.sin(self.rotation)
        y = self.x * np.sin(self.rotation) - self.y * np.cos(self.rotation)
        return self.__class__(x, y, -self.rotation)

    def apply(self, points: np.ndarray):
        assert points.shape[-1] == 2
        x = points[..., 0]
        y = points[..., 1]
        x1 = x * np.cos(self._angle) - y * np.sin(self._angle) + self._x
        y1 = x * np.sin(self._angle) + y * np.cos(self._angle) + self._y
        return np.stack([x1, y1], axis=-1)

    def __repr__(self):
        return f"Position2D(x={self._x}, y={self._y}, angle={self._angle})"

    def distance(self, position):
        angle_delta = wrap_angles(self.rotation - position.rotation)
        return np.sqrt((position.x - self.x) ** 2 + (position.y - self.y) ** 2 + angle_delta ** 2)

    def __eq__(self, other):
        return (np.all(self.x == other.x)) and (np.all(self.y == other.y)) and (np.all(self.rotation == other.rotation))

    def as_matrix(self):
        return np.array([[np.cos(self.rotation), -np.sin(self.rotation), self.x],
                         [np.sin(self.rotation), np.cos(self.rotation), self.y],
                         [0, 0, 1]])

    def steer(self, point2, steer_distance: float):
        distance = self.distance(point2)
        if distance < steer_distance:
            return point2
        direction = (point2.as_numpy() - self.as_numpy())
        direction[2] = wrap_angles(direction[2])
        direction = direction / distance
        return Position2D(self.x + steer_distance * direction[0], self.y + steer_distance * direction[1],
                          self.rotation + steer_distance * direction[2])
