import numpy as np
from matplotlib import pyplot as plt
from shapely import MultiPolygon, Polygon

from swarm_nfomp.arrt.rrt_planner import Tree
from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector
from swarm_nfomp.collision_detector.robot_collision_detector import RobotCollisionDetector
from swarm_nfomp.utils.point_array2d import PointArray2D
from swarm_nfomp.utils.position2d import Position2D
from swarm_nfomp.utils.position_array2d import PositionArray2D


def show_tree(tree: Tree, fig):
    ax = fig.gca()

    plt.scatter(tree.points[:, 0], tree.points[:, 1], color='black', s=1)

    for node in tree.nodes:
        if node.parent is not None:
            ax.plot([node.point.x, node.parent.point.x], [node.point.y, node.parent.point.y], color='blue')


def show_array_collision_detector(collision_detector: PointArrayCollisionDetector, fig):
    ax = fig.gca()
    for region in collision_detector.inside_rectangle_region_array:
        ax.add_patch(
            plt.Rectangle(
                (region[0], region[2]),  # (x,y)
                region[1] - region[0],  # width
                region[3] - region[2],  # height
                color='black'
            )
        )


def show_point(start_point, fig, color="green"):
    ax = fig.gca()
    ax.scatter([start_point.x], [start_point.y], color=color, s=20)


def show_position(position: Position2D, fig, color="green"):
    ax = fig.gca()
    ax.scatter([position.x], [position.y], color=color, s=20)


def show_path(path: PointArray2D, fig):
    plt.plot(path.x, path.y, color='red')


def show_multipolygon(polygon: MultiPolygon, fig, color):
    # create an empty list to hold the coordinates of the polygons
    x, y = [], []
    ax = fig.gca()
    # iterate over the polygons in the MultiPolygon
    for poly in polygon.geoms:
        # get the exterior coordinates of the polygon
        poly_coords = poly.exterior.coords.xy
        x = poly_coords[0].tolist()
        y = poly_coords[1].tolist()
        xy = np.stack([x, y], axis=1)
        ax.add_patch(plt.Polygon(xy, True, color=color))


def show_robot_collision_detector(robot_collision_detector: RobotCollisionDetector, fig):
    show_multipolygon(robot_collision_detector.inside_rectangle_region, fig, 'black')


def show_position_array2d(position_array: PositionArray2D, fig, color="green"):
    plt.plot(position_array.x, position_array.y, color=color)
    plt.quiver(position_array.x, position_array.y, np.cos(position_array.angle), np.sin(position_array.angle),
               color=color)


def show_position_array2d_with_mask(position_array: PositionArray2D, fig, mask: np.ndarray, color="black",
                                    color1="green", color2="red"):
    if np.count_nonzero(mask):
        plt.quiver(position_array.x[mask], position_array.y[mask], np.cos(position_array.angle)[mask],
                   np.sin(position_array.angle)[mask],
                   color=color1)
    mask = np.logical_not(mask)
    if np.count_nonzero(mask):
        plt.quiver(position_array.x[mask], position_array.y[mask], np.cos(position_array.angle)[mask],
               np.sin(position_array.angle)[mask],
               color=color2)
    plt.plot(position_array.x, position_array.y, color=color)


def show_transformed_robot_shapes(position_array: PositionArray2D, robot_shape: Polygon, fig, color="green"):
    poly_coords = robot_shape.exterior.coords.xy
    x = poly_coords[0].tolist()
    y = poly_coords[1].tolist()
    xy = np.stack([x, y], axis=1)
    ax = fig.gca()
    for position in position_array:
        transformed_xy = position.apply(xy)
        ax.add_patch(plt.Polygon(transformed_xy, True, fill=False, color=color))
