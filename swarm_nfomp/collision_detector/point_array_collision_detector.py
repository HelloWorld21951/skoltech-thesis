import numpy as np

from swarm_nfomp.utils.point2d import Point2D
from swarm_nfomp.utils.point_array2d import PointArray2D


class RectangleRegionArray:
    def __init__(self, min_x: np.ndarray, max_x: np.ndarray, min_y: np.ndarray, max_y: np.ndarray):
        assert min_x.shape == max_x.shape
        assert min_x.shape == min_y.shape
        assert min_x.shape == max_y.shape
        assert np.all(min_x < max_x)
        assert np.all(min_y < max_y)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def inside(self, positions: PointArray2D) -> np.ndarray:
        result = self.min_x <= positions.x[:, None]
        result &= positions.x[:, None] <= self.max_x
        result &= self.min_y <= positions.y[:, None]
        result &= positions.y[:, None] <= self.max_y
        return np.any(result, axis=1)

    @classmethod
    def from_dict(cls, data):
        data = np.array(data)
        assert len(data.shape) == 2
        assert data.shape[1] == 4
        return cls(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    def __getitem__(self, item):
        return np.array([self.min_x[item], self.max_x[item], self.min_y[item], self.max_y[item]])

    def __len__(self):
        return len(self.min_x)


class PointArrayCollisionDetector:
    def __init__(self, inside_rectangle_region_array: RectangleRegionArray,
                 outside_rectangle_region_array: RectangleRegionArray):
        self.inside_rectangle_region_array = inside_rectangle_region_array
        self.outside_rectangle_region_array = outside_rectangle_region_array

    def is_collision(self, point: Point2D) -> bool:
        points = PointArray2D.from_vec(point.as_numpy()[None])
        return (self.inside_rectangle_region_array.inside(points) | (
            ~self.outside_rectangle_region_array.inside(points)))[0]

    def is_collision_for_array(self, points: PointArray2D) -> np.ndarray:
        return self.inside_rectangle_region_array.inside(points) | (
            ~self.outside_rectangle_region_array.inside(points))

    @classmethod
    def from_dict(cls, data):
        return PointArrayCollisionDetector(RectangleRegionArray.from_dict(data["inside_rectangle_region_array"]),
                                           RectangleRegionArray.from_dict(data["outside_rectangle_region_array"]))
