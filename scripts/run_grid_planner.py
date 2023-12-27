import copy
import os
from multiprocessing import Process, Queue

import yaml
from matplotlib import pyplot as plt

from swarm_nfomp.grid_planner.eggs_grid_planner import EGGSGridPlanner
from swarm_nfomp.grid_planner.grid_planner import GridPlannerTask
from swarm_nfomp.grid_planner.grid_result_visualizer import GridPlannerResultVisualizer
from swarm_nfomp.utils.universal_factory import UniversalFactory


def plot_process_function(queue: Queue):
    result_visualizer = GridPlannerResultVisualizer()

    while True:
        queue_size = queue.qsize()
        previous_queue_result = None
        for i in range(queue_size - 1):
            previous_queue_result = queue.get()
        queue_result = queue.get()

        if queue_result is not None:
            planner, result = queue_result
            result_visualizer.visualize(planner, result)
            plt.pause(0.01)
        elif previous_queue_result is not None:
            print(previous_queue_result)
            planner, result = previous_queue_result
            result_visualizer.visualize(planner, result)
            plt.pause(0.01)
            break
        else:
            break

    plt.show()


def main():
    task_config_path = "configs/path_planner_tasks/simple_grid_task.yaml"

    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(parent_path, task_config_path), "r") as f:
        task_parameters = yaml.safe_load(f)

    planner: EGGSGridPlanner = UniversalFactory().make(EGGSGridPlanner, {})
    planner_task = UniversalFactory().make(GridPlannerTask, task_parameters)
    planner.set_planner_task(planner_task)
    iterations = 1
    queue = Queue()
    p = Process(target=plot_process_function, args=(queue,))
    p.start()
    for i in range(iterations):
        result = planner.plan()
        queue.put((copy.deepcopy(planner), copy.deepcopy(result)))
    queue.put(None)
    p.join()


if __name__ == '__main__':
    main()
