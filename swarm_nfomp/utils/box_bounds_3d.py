import dataclasses

import numpy as np

from swarm_nfomp.planner.planner import Bounds
from swarm_nfomp.utils.position3d import Position3D


@dataclasses.dataclass
class BoxBounds3D(Bounds[Position3D]):
    max_x: float
    min_x: float
    max_y: float
    min_y: float
    max_z: float
    min_z: float

    def sample_random_point(self) -> Position3D:
        translation = [
            np.random.uniform(self.min_x, self.max_x),
            np.random.uniform(self.min_y, self.max_y),
            np.random.uniform(self.min_z, self.max_z),
        ]
        euler_angles = [
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
        ]
        return Position3D(translation, euler_angles)
