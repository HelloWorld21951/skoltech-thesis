import dataclasses

import numpy as np

from swarm_nfomp.utils.math import wrap_angles
from swarm_nfomp.utils.position_array2d import PositionArray2D


@dataclasses.dataclass
class Point2Array:
    x: np.ndarray
    y: np.ndarray

    def as_numpy_array(self):
        return np.array([self.x, self.y])


def test_as_numpy_array():
    x = np.array([0, 3])
    y = np.array([0, 2])
    p = Point2Array(x, y)
    np.testing.assert_array_equal(p.as_numpy_array(), np.array([x, y]))


def test_wrap_angle():
    y = np.array([0, 6])
    result_y = wrap_angles(angles=y)
    np.testing.assert_array_equal(result_y, np.array([y[0], y[1] - 2 * np.pi]))


def test_position_array_2d():
    p = PositionArray2D(np.array([0, 1]), np.array([0, 2]), np.array([0, 4]))
    x = p.as_vec()
