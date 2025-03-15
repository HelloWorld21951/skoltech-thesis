from typing import List, Dict, Union, Iterable

import numpy as np

from swarm_nfomp.utils.math import wrap_angles
from swarm_nfomp.utils.position3d import Position3D


class PositionArray3D:
    def __init__(self, translation_list: Iterable[Iterable[float]], euler_angles_list: Iterable[Iterable[float]]) -> None:
        assert len(translation_list) == len(euler_angles_list)
        self._positions = [Position3D(translation, euler_angles) for translation, euler_angles in zip(translation_list, euler_angles_list)]

    @classmethod
    def from_vec(cls, vec: np.ndarray) -> 'PositionArray3D':
        assert isinstance(vec, np.ndarray)
        assert vec.shape[-1] == 6
        assert len(vec.shape) == 2
        return cls(vec[:, :3], vec[:, 3:])
    
    def as_vec(self) -> np.ndarray:
        positions = np.zeros((len(self._positions), 6))
        for i in range(len(self._positions)):
            positions[i, :3] = self._positions[i].translation
            positions[i, 3:] = self._positions[i].euler_angles
        return positions
    
    @classmethod
    def from_list(cls, positions: List[Position3D]) -> 'PositionArray3D':
        translation_list = [position.translation for position in positions]
        euler_angles_list = [position.euler_angles for position in positions]
        return cls(translation_list, euler_angles_list)

    def as_list(self) -> List[Position3D]:
        return self._positions

    def __getitem__(self, item) -> Union['PositionArray3D', Position3D]:
        if isinstance(item, slice):
            return PositionArray3D.from_list(self._positions[item])
        return self._positions[item]

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        return f"PositionArray3D(positions={self._positions})"

    @classmethod
    def interpolate(cls, point1: Position3D, point2: Position3D, interpolation_count: int) -> 'PositionArray3D':
        assert interpolation_count >= 2

        diff = point2 - point1
        translation_list = []
        euler_angles_list = []
        for i in range(interpolation_count):
            translation_list.append(point1.translation + diff.translation * i / (interpolation_count - 1))
            euler_angles_list.append(point1.euler_angles + diff.euler_angles * i / (interpolation_count - 1))

        return cls(translation_list, euler_angles_list)

    @classmethod
    def from_dict(cls, parameters: Dict) -> 'PositionArray3D':
        return cls.from_vec(np.array(parameters))
