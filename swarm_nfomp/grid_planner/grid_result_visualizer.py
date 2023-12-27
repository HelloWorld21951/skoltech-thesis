import numpy as np
from matplotlib import pyplot as plt

from swarm_nfomp.grid_planner.grid_planner import GridPlanner


class GridPlannerResultVisualizer():
    def __init__(self):
        self._fig = plt.figure(dpi=200)

    def visualize(self, planner: GridPlanner, result):
        image = np.zeros((planner.planner_task.grid.shape[0], planner.planner_task.grid.shape[1], 3),
                         dtype=np.uint8)
        grid = planner.planner_task.grid.astype(np.bool)
        image = np.where(~grid[:, :, None], [255, 255, 255], image)
        for node in planner.closed_set:
            image[node[0], node[1]] = [0, 125, 125]
        for node in result:
            image[node[0], node[1]] = [0, 0, 255]
        image[planner.planner_task.start_point[0], planner.planner_task.start_point[1]] = [0, 255, 0]
        image[planner.planner_task.goal_point[0], planner.planner_task.goal_point[1]] = [255, 0, 0]

        plt.imshow(image)
