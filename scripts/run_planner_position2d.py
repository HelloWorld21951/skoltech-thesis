import copy
import os
from multiprocessing import Process, Queue

import yaml
from matplotlib import pyplot as plt

from swarm_nfomp.arrt.arrt_position2d_planner import ARRTPosition2DPlanner
from swarm_nfomp.arrt.rrt_position2d_planner import Position2DPlannerTask, RRTPosition2DPlanner
from swarm_nfomp.plotting.plotting import show_tree, show_robot_collision_detector, show_position, \
    show_position_array2d, show_transformed_robot_shapes
from swarm_nfomp.utils.universal_factory import UniversalFactory


def show(fig, planner, result):
    fig.clear()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    bounds = planner.planner_task.bounds
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_y, bounds.max_y)
    show_robot_collision_detector(planner.planner_task.collision_detector, fig)
    show_tree(planner.tree, fig)
    show_position(planner.planner_task.start, fig, color="green")
    show_position(planner.planner_task.goal, fig, color="red")
    show_position_array2d(result, fig, color="red")
    show_transformed_robot_shapes(result, planner.planner_task.collision_detector.robot_shape, fig)
    plt.pause(0.01)


def plot_process_function(queue: Queue, fig):
    while True:
        queue_size = queue.qsize()
        previous_queue_result = None
        for i in range(queue_size - 1):
            previous_queue_result = queue.get()
        queue_result = queue.get()

        if queue_result is not None:
            planner, result = queue_result
            show(fig, planner, result)
        elif previous_queue_result is not None:
            planner, result = previous_queue_result
            show(fig, planner, result)
            break
        else:
            break
    plt.show()


def main():
    # task_config_path = "configs/path_planner_tasks/simple_rectangle_robot_task.yaml"
    # task_config_path = "configs/path_planner_tasks/narrow_rectangle_robot_task.yaml"
    task_config_path = "configs/path_planner_tasks/long_road.yaml"
    # planner_config_path = "configs/planner_parameters/rrt_position2d_planner.yaml"
    # planner_config_path = "configs/planner_parameters/arrt_position2d_planner.yaml"
    planner_config_path = "configs/planner_parameters/eggs_arrt_position2d_planner.yaml"

    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_path, task_config_path), "r") as f:
        task_parameters = yaml.safe_load(f)

    with open(os.path.join(parent_path, planner_config_path), "r") as f:
        planner_parameters = yaml.safe_load(f)

    planner: ARRTPosition2DPlanner = UniversalFactory().make(ARRTPosition2DPlanner, planner_parameters)
    # planner: RRTPosition2DPlanner = UniversalFactory().make(RRTPosition2DPlanner, planner_parameters)
    planner_task = UniversalFactory().make(Position2DPlannerTask, task_parameters)
    planner.set_planner_task(planner_task)

    iterations = 1000
    fig = plt.figure(dpi=200)
    queue = Queue()
    p = Process(target=plot_process_function, args=(queue, fig))
    p.start()
    while not planner.is_goal_reached:
        result = planner.plan()
        queue.put((copy.deepcopy(planner), copy.deepcopy(result)))
    queue.put(None)
    print(f"Number collision checks = {planner_task.collision_detector.number_collision_between_checks}")
    p.join()


if __name__ == '__main__':
    main()
