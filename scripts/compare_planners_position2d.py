import copy
import os
from multiprocessing import Process, Queue

import yaml
from matplotlib import pyplot as plt

from swarm_nfomp.arrt.arrt_position2d_planner import ARRTPosition2DPlanner
from swarm_nfomp.arrt.rrt_position2d_planner import RRTPosition2DPlanner, Position2DPlannerTask
from swarm_nfomp.plotting.plotting import show_tree, show_robot_collision_detector, show_position, \
    show_position_array2d, show_transformed_robot_shapes
from swarm_nfomp.utils.universal_factory import UniversalFactory
import copy
import os
from multiprocessing import Process, Queue

import yaml
from matplotlib import pyplot as plt

from swarm_nfomp.arrt.rrt_position2d_planner import RRTPosition2DPlanner, Position2DPlannerTask
from swarm_nfomp.plotting.plotting import show_tree, show_robot_collision_detector, show_position, \
    show_position_array2d, show_transformed_robot_shapes
from swarm_nfomp.utils.universal_factory import UniversalFactory


def main():
    # task_config_path = "configs/path_planner_tasks/simple_rectangle_robot_task.yaml"
    task_config_path = "configs/path_planner_tasks/narrow_rectangle_robot_task.yaml"
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

    while not planner.is_goal_reached:
        planner.plan()
    print(f"Number collision checks = {planner_task.collision_detector.number_collision_between_checks}")


if __name__ == '__main__':
    main()
