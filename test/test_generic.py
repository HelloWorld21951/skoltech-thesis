from typing import TypeVar, Generic

State = TypeVar('State')


class Bounds(Generic[State]):
    def __init__(self, vec):
        self._vec = vec

    def sample_random_point(self) -> State:
        return self.__orig_class__.__args__[0].from_vec(self._vec)


def test_generic():
    class Point2D:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        @classmethod
        def from_vec(cls, vec):
            return cls(vec[0], vec[1])

    bounds = Bounds[Point2D]([1, 2])
    bounds.sample_random_point()
