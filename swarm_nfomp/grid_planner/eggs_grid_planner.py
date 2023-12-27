from collections import deque

import numpy as np

from swarm_nfomp.grid_planner.grid_planner import GridPlanner, GridPlannerTask


class BFS:
    def __init__(self):
        self._grid = None

    def is_valid(self, x, y):
        return 0 <= x < self._grid.shape[0] and 0 <= y < self._grid.shape[1] and not self._grid[x][y]

    def update(self, grid, goal):
        self._grid = grid
        goal = goal

        cost_grid = np.ones_like(grid) * np.inf
        visited = np.zeros_like(grid, dtype=bool)

        queue = deque()
        queue.append(goal)
        cost_grid[goal[0], goal[1]] = 0
        visited[goal[0], goal[1]] = True

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = x + dx, y + dy

                if self.is_valid(new_x, new_y) and not visited[new_x][new_y]:
                    visited[new_x][new_y] = True
                    queue.append((new_x, new_y))
                    cost_grid[new_x][new_y] = cost_grid[x][y] + 1

        return cost_grid


class EGGSGridPlanner(GridPlanner):
    def __init__(self):
        super().__init__()
        self._bfs = BFS()
        self._open_node_list = []
        self._closed_grid = None
        self._goal = None
        self._bfs_grid = None
        self._parents = {}

    def set_planner_task(self, planner_task: GridPlannerTask):
        self._open_node_list = [planner_task.start_point]
        self._goal = planner_task.goal_point
        self._closed_grid = np.zeros_like(planner_task.grid, dtype=bool)
        self._parents[planner_task.start_point] = None
        super().set_planner_task(planner_task)

    def plan(self):
        current_node = None
        while len(self._open_node_list) > 0:
            current_node = self._step()
        if current_node is None:
            return None
        return self._path(current_node)

    def _step(self):
        self._bfs_grid = self._bfs.update(self._closed_grid, self._goal)

        current_node = None
        while current_node is None:
            current_node = self._find_next_open_node_and_update_node_list()
            if current_node is None:
                return
            parent_node = self._parents[current_node]
            if parent_node is None:
                break
            if not self._is_valid(current_node[0], current_node[1], parent_node[0], parent_node[1]):
                current_node = None
        if current_node == self._goal:
            return current_node

        x, y = current_node
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            is_inside = 0 <= new_x < self.planner_task.grid.shape[0] and 0 <= new_y < self.planner_task.grid.shape[1]
            if is_inside and not (new_x, new_y) in self.closed_set:
                self._open_node_list.append((new_x, new_y))
                self._parents[(new_x, new_y)] = current_node
        self.closed_set.add(current_node)
        self._closed_grid[current_node[0], current_node[1]] = True
        return current_node

    def _find_next_open_node_and_update_node_list(self):
        minimal_cost = np.inf
        minimal_cost_node = None
        new_open_node_list = []
        for node in self._open_node_list:
            cost = self._bfs_grid[node[0], node[1]]
            if cost < minimal_cost:
                minimal_cost = cost
                if minimal_cost_node is not None:
                    new_open_node_list.append(minimal_cost_node)
                minimal_cost_node = node
            elif cost != np.inf:
                new_open_node_list.append(node)
        self._open_node_list = new_open_node_list
        return minimal_cost_node

    def _is_valid(self, x, y, parent_x, parent_y):
        return not self.planner_task.grid[x][y]

    def _path(self, goal_node):
        path = []
        node = goal_node
        while node:
            path.append((node[0], node[1]))
            node = self._parents[node]
        return path[::-1]
