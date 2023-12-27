import dataclasses
import os
from typing import List

import numpy as np
import plotly.graph_objects as go
from shapely import Polygon, MultiPolygon

from swarm_nfomp.collision_detector.multi_robot_collision_detector import MultiRobotCollisionDetector
from swarm_nfomp.utils.position_array2d import PositionArray2D
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp import MultiRobotResultPath


@dataclasses.dataclass
class CollisionDetectionResultVisualizerConfig:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


class CollisionDetectionResultVisualizer:
    def __init__(self, parameters: CollisionDetectionResultVisualizerConfig):
        self._fig = go.Figure()
        self.parameters = parameters

    def visualize(self, detector: MultiRobotCollisionDetector, robot_positions: List[PositionArray2D]):
        self.draw_initial_polygons(detector)
        self.add_frames(detector, robot_positions)
        self.update_layout()

    def draw_initial_polygons(self, detector: MultiRobotCollisionDetector):
        self._fig.add_trace(self.draw_multipolygon(detector.inside_rectangle_region))
        self._fig.add_trace(self.draw_polygon(detector.outside_rectangle_region))
        self._fig.add_trace(self.draw_polygons(detector.robot_shapes, color="green"))

    @staticmethod
    def draw_multipolygon(polygon: MultiPolygon):
        # create an empty list to hold the coordinates of the polygons
        x, y = [], []

        # iterate over the polygons in the MultiPolygon
        for poly in polygon.geoms:
            # get the exterior coordinates of the polygon
            poly_coords = poly.exterior.coords.xy
            x.extend(poly_coords[0].tolist())
            y.extend(poly_coords[1].tolist())
            x.append(None)
            y.append(None)

        # create a scatter trace with the polygon coordinates
        return go.Scatter(x=x, y=y, mode='lines', fill="toself", line=dict(color='blue'), name="inside polygon")

    @staticmethod
    def draw_polygon(polygon: Polygon):
        # get the exterior coordinates of the polygon
        poly_coords = polygon.exterior.coords.xy
        x = poly_coords[0].tolist()
        y = poly_coords[1].tolist()

        # create a scatter trace with the polygon coordinates
        return go.Scatter(x=x, y=y, mode='lines', line=dict(color='red'), name="outside polygon")

    @staticmethod
    def draw_polygons(polygons: List[Polygon], color: str):
        # create an empty list to hold the coordinates of the polygons
        x, y = [], []

        # iterate over the polygons in the MultiPolygon
        for poly in polygons:
            # get the exterior coordinates of the polygon
            poly_coords = poly.exterior.coords.xy
            x.extend(poly_coords[0].tolist())
            y.extend(poly_coords[1].tolist())
            x.append(None)
            y.append(None)

        # create a scatter trace with the polygon coordinates
        return go.Scatter(x=x, y=y, mode='lines', line=dict(color=color), fill="toself", name="robot shapes")

    def add_frames(self, detector: MultiRobotCollisionDetector, robot_positions: List[PositionArray2D]):
        frames = []
        for robot_position in robot_positions:
            is_collision = detector.is_collision(robot_position)
            robot_shapes = detector.transformed_robot_shapes(robot_position)
            color = "red" if is_collision else "green"
            frame = go.Frame(
                data=[
                    self.draw_multipolygon(detector.inside_rectangle_region),
                    self.draw_polygon(detector.outside_rectangle_region),
                    self.draw_polygons(robot_shapes, color=color),
                ],
            )
            frames.append(frame)
        self._fig.frames = frames

    def update_layout(self):
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    x=0.1,
                    y=1.2,
                    buttons=list([
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {
                                    "duration": 50,
                                    "redraw": False
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 0,
                                }
                            }]
                        ),
                    ]),
                )
            ]
        )
        self._fig.update_layout(
            yaxis={
                "scaleanchor": "x",
                "scaleratio": 1,
                "range": [self.parameters.ymin, self.parameters.ymax]
            },
            xaxis={
                "range": [self.parameters.xmin, self.parameters.xmax]
            }
        )

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fig.write_html(filename)

    @property
    def fig(self):
        return self._fig


class WarehousePathPlannerResultVisualizer(CollisionDetectionResultVisualizer):
    def visualize(self, detector: MultiRobotCollisionDetector, path: MultiRobotResultPath):
        self.draw_initial_polygons(detector)
        self.draw_paths(path)
        self.add_frames(detector, [PositionArray2D.from_vec(x) for x in path.numpy_positions])
        self.update_layout()

    def draw_initial_polygons(self, detector):
        super().draw_initial_polygons(detector)

    def draw_paths(self, path: MultiRobotResultPath):
        positions: np.ndarray = path.numpy_positions
        for i in range(positions.shape[1]):
            xs = []
            ys = []
            for position in positions[:, i]:
                xs.append(position[0])
                xs.append(position[0] + np.cos(position[2]) * 1)
                xs.append(None)
                ys.append(position[1])
                ys.append(position[1] + np.sin(position[2]) * 1)
                ys.append(None)
            trace = go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"Robot {i} orientation",
                line=dict(
                    width=2
                )
            )
            self._fig.add_trace(trace)
            trace = go.Scatter(
                x=positions[:, i, 0],
                y=positions[:, i, 1],
                mode="lines+markers",
                name=f"Robot {i}",
                line=dict(
                    width=2
                )
            )
            self._fig.add_trace(trace)
