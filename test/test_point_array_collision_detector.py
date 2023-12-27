import numpy as np
import pytest

from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector
from swarm_nfomp.utils.point2d import Point2D
from swarm_nfomp.utils.point_array2d import PointArray2D


@pytest.fixture
def collision_detector():
    rectangle_collision_detector_parameters = {
        "outside_rectangle_region_array": [[-10, 20, 0, 10]],
        "inside_rectangle_region_array": [[4, 6, 0, 4], [4, 6, 6, 10]]
    }
    return PointArrayCollisionDetector.from_dict(rectangle_collision_detector_parameters)


@pytest.fixture
def point():
    return Point2D(2, 2)


def test_is_collision(collision_detector, point):
    assert ~collision_detector.is_collision(point)
