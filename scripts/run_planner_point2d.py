import copy
from multiprocessing import Process, Queue

from matplotlib import pyplot as plt

from swarm_nfomp.arrt.arrt_point2d_planner import ARRTPoint2DPlanner
from swarm_nfomp.arrt.rrt_point2d_planner import Point2DPlannerTask, RRTPoint2DPlanner
from swarm_nfomp.plotting.plotting import show_tree, show_array_collision_detector, show_point, show_path
from swarm_nfomp.utils.universal_factory import UniversalFactory


def plot_process_function(queue: Queue, fig):
    while True:
        queue_result = queue.get()
        if queue_result is None:
            break
        planner, result = queue_result
        fig.clear()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        show_array_collision_detector(planner.planner_task.collision_detector, fig)
        bounds = planner.planner_task.bounds
        ax.set_xlim(bounds.min_x, bounds.max_x)
        ax.set_ylim(bounds.min_y, bounds.max_y)
        show_tree(planner.tree, fig)
        show_point(planner.planner_task.start, fig, color="green")
        show_point(planner.planner_task.goal, fig, color="red")
        show_path(result, fig)
        plt.pause(0.01)


planner_parameters = {
    "type_": "RRTStarParameters",
    "collision_step": 0.1,
    "iterations": 1,
    "steer_distance": 0.2,
    "goal_point_probability": 0.4,
    "a_star_side_count": 10,
    "a_star_iterations": 60,
    "path_factory": {}
}

task_parameters = {
    "start": {
        "x": 0,
        "y": 2,
    },
    "goal": {
        "x": 15,
        "y": 8,
    },
    "collision_detector": {
        "type_": "PointArrayCollisionDetector",
        "outside_rectangle_region_array": [[-10, 20, 0, 10]],
        "inside_rectangle_region_array": [[4, 6, 0, 4], [4, 6, 6, 10], [9, 10, 2, 8], [9, 10, 8.3, 15]]
    },
    "bounds": {
        "min_x": -10,
        "max_x": 20,
        "min_y": 0,
        "max_y": 10
    },

}

planner: RRTPosition2DPlanner = UniversalFactory().make(RRTPoint2DPlanner, planner_parameters)
planner_task = UniversalFactory().make(Point2DPlannerTask, task_parameters)
planner.set_planner_task(planner_task)

iterations = 1000
fig = plt.figure(dpi=200)
result = planner.plan()
queue = Queue(1)
p = Process(target=plot_process_function, args=(queue, fig))
p.start()
for i in range(iterations):
    result = planner.plan()
    queue.put((copy.deepcopy(planner), copy.deepcopy(result)))
queue.put(None)
p.join()
