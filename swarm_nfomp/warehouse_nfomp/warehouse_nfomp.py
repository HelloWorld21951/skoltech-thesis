import dataclasses
import math
from multiprocessing import Queue, Process
from typing import Optional, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from swarm_nfomp.collision_detector.multi_robot_collision_detector import MultiRobotCollisionDetector
from swarm_nfomp.utils.math import interpolate_1d_pytorch, wrap_angles
from swarm_nfomp.utils.metric_manager import MetricManager
from swarm_nfomp.utils.position_array2d import PositionArray2D
from swarm_nfomp.utils.rectangle_bounds import RectangleBounds2D
from swarm_nfomp.utils.timer import Timer


@dataclasses.dataclass
class MultiRobotPathPlannerTask:
    start: PositionArray2D
    goal: PositionArray2D
    collision_detector: MultiRobotCollisionDetector
    bounds: RectangleBounds2D


@dataclasses.dataclass
class MultiRobotResultPath:
    positions_: np.array  # [ n_optimized_states, n_robots, 3]

    @property
    def robot_paths(self) -> List[PositionArray2D]:
        return [PositionArray2D.from_vec(self.positions_[:, i]) for i in range(self.positions_.shape[1])]

    @property
    def numpy_positions(self):
        return self.positions_


@dataclasses.dataclass
class MultiRobotPathOptimizedState:
    positions_: torch.Tensor  # [ n_optimized_states, n_robots, 3]
    direction_constraint_multipliers: torch.Tensor
    start_position: torch.Tensor  # [n_robots, 3]
    goal_position: torch.Tensor  # [n_robots, 3]
    device: str

    @property
    def result_path(self) -> MultiRobotResultPath:
        return MultiRobotResultPath(
            self.positions.cpu().detach().numpy()
        )

    @property
    def positions(self) -> torch.Tensor:
        return torch.cat(
            [self.start_position[None], self.positions_, self.goal_position[None]], dim=0)

    def reparametrize(self):
        distances = self.calculate_distances()
        old_times = torch.cumsum(distances, dim=0)
        old_times = torch.cat([torch.zeros(1, device=self.device), old_times], dim=0)
        new_times = torch.linspace(0, old_times[-1], old_times.shape[0], device=self.device)
        positions: torch.Tensor = self.positions
        reshaped_positions = positions.reshape(self.positions.shape[0], -1)
        interpolated_positions = interpolate_1d_pytorch(reshaped_positions, old_times, new_times)[1:-1]
        self.positions_.data = interpolated_positions.reshape(*self.positions_.shape)
        multipliers_old_times = (old_times[:-1] + old_times[1:]) / 2
        multipliers_new_times = (new_times[:-1] + new_times[1:]) / 2
        self.direction_constraint_multipliers.data = interpolate_1d_pytorch(
            self.direction_constraint_multipliers, multipliers_old_times, multipliers_new_times)

    def calculate_distances(self) -> torch.Tensor:
        points = self.positions[:, :, :2]
        drone_distances = torch.linalg.norm(points[1:] - points[:-1], dim=2)
        return torch.max(drone_distances, dim=1).values

    @property
    def optimized_parameters(self):
        return [self.positions_]


@dataclasses.dataclass
class OptimizerImplConfig:
    lr: float
    beta1: float
    beta2: float
    step_size: int
    gamma: float


class OptimizerImpl:
    def __init__(self, parameters: OptimizerImplConfig):
        self._optimizer = None
        self._scheduler = None
        self._parameters = parameters

    def setup(self, model_parameters):
        self._optimizer = torch.optim.Adam(model_parameters, lr=self._parameters.lr,
                                           betas=(self._parameters.beta1, self._parameters.beta2))
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=self._parameters.step_size,
                                                          gamma=self._parameters.gamma)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()
        self._scheduler.step()


@dataclasses.dataclass
class OptimizerWithLagrangeMultipliersConfig:
    lr: float
    beta1: float
    beta2: float
    lagrange_multiplier_lr: float
    base_lr: float
    max_lr: float
    step_size_up: int
    step_size_down: int


class OptimizerWithLagrangeMultipliers:
    def __init__(self, parameters: OptimizerWithLagrangeMultipliersConfig):
        self._parameters = parameters
        self._lagrange_multiplier_parameters = None
        self._scheduler = None
        self._optimizer = None

    # noinspection PyMethodOverriding
    def setup(self, model_parameters, lagrange_multiplier_parameters):
        self._optimizer = torch.optim.RMSprop(model_parameters, lr=self._parameters.lr)
        self._scheduler = torch.optim.lr_scheduler.CyclicLR(self._optimizer, base_lr=self._parameters.base_lr,
                                                            max_lr=self._parameters.max_lr,
                                                            step_size_up=self._parameters.step_size_up,
                                                            step_size_down=self._parameters.step_size_down,
                                                            cycle_momentum=False)
        self._lagrange_multiplier_parameters = lagrange_multiplier_parameters

    def zero_grad(self):
        self._optimizer.zero_grad()
        for p in self._lagrange_multiplier_parameters:
            p.grad = None

    def step(self):
        self._optimizer.step()
        self._scheduler.step()
        with torch.no_grad():
            for p in self._lagrange_multiplier_parameters:
                p.data += self._parameters.lagrange_multiplier_lr * p.grad


class PathOptimizedStateInitializer:
    def __init__(self, planner_task: MultiRobotPathPlannerTask, path_state_count: int, device: str):
        self._planner_task = planner_task
        self._path_state_count = path_state_count
        self._device = device

    def init(self) -> MultiRobotPathOptimizedState:
        with torch.no_grad():
            positions = self._initialize_positions()
            start_position = torch.tensor(self._planner_task.start.as_vec(), requires_grad=False, device=self._device,
                                          dtype=torch.float32)
            goal_position = torch.tensor(self._planner_task.goal.as_vec(), requires_grad=False, device=self._device,
                                         dtype=torch.float32)
            goal_position[:, 2] = start_position[:, 2] + wrap_angles(goal_position[:, 2] - start_position[:, 2])
            return MultiRobotPathOptimizedState(
                positions_=positions[1:-1].clone().detach().requires_grad_(True),
                start_position=start_position,
                goal_position=goal_position,
                direction_constraint_multipliers=torch.zeros(self._path_state_count + 1, len(self._planner_task.start),
                                                             requires_grad=True,
                                                             device=self._device, dtype=torch.float32),
                device=self._device
            )

    def _initialize_positions(self) -> torch.Tensor:
        start_point: np.ndarray = self._planner_task.start.as_vec()
        goal_point: np.ndarray = self._planner_task.goal.as_vec()
        trajectory_length = self._path_state_count + 2
        trajectory = torch.zeros(self._path_state_count + 2, len(start_point), 3, requires_grad=True,
                                 device=self._device, dtype=torch.float32)
        for i in range(len(start_point)):
            trajectory[:, i, 0] = torch.linspace(start_point[i, 0], goal_point[i, 0], trajectory_length,
                                                 device=self._device)
            trajectory[:, i, 1] = torch.linspace(start_point[i, 1], goal_point[i, 1], trajectory_length)
            trajectory[:, i, 2] = start_point[i, 2] + torch.linspace(0,
                                                                     wrap_angles(goal_point[i, 2] - start_point[i, 2]),
                                                                     trajectory_length, device=self._device)
        return trajectory


@dataclasses.dataclass
class PathLossBuilderConfig:
    regularization_weight: float
    collision_weight: float
    direction_constraint_weight: float
    second_differences_weight: float


class MultiRobotPathLossBuilder:
    def __init__(self, planner_task: MultiRobotPathPlannerTask, parameters: PathLossBuilderConfig,
                 metric_manager: MetricManager, device):
        self._device = device
        self._parameters = parameters
        self._planner_task = planner_task
        self._metric_manager = metric_manager

    def get_loss(self, collision_model: nn.Module, optimized_state: MultiRobotPathOptimizedState):
        distance_loss = self._distance_loss(optimized_state) * self._parameters.regularization_weight
        collision_loss = self._collision_loss(collision_model, optimized_state) * self._parameters.collision_weight
        direction_constraint_loss = self._direction_constraint_loss(optimized_state)
        second_differences_loss = self._second_differences_loss(
            optimized_state) * self._parameters.second_differences_weight
        self._metric_manager.add_metric("Path Optimization Losses", distance_loss.item(), 'distance_loss')
        self._metric_manager.add_metric("Path Optimization Losses", collision_loss.item(), 'collision_loss')
        self._metric_manager.add_metric("Path Optimization Losses", direction_constraint_loss.item(), 'direction_loss')
        self._metric_manager.add_metric("Path Optimization Losses", second_differences_loss.item(),
                                        'second_differences_loss')
        loss = distance_loss
        loss = loss + collision_loss
        loss = loss + direction_constraint_loss
        loss = loss + second_differences_loss
        self._metric_manager.add_metric("Path Optimization Losses", loss.item(), 'total_loss')
        return loss

    @staticmethod
    def _distance_loss(optimized_state: MultiRobotPathOptimizedState):
        points = optimized_state.positions
        delta = points[1:] - points[:-1]
        delta[:, :, 2] = wrap_angles(delta[:, :, 2])
        return torch.mean(torch.abs(delta) ** 1.5)

    def _collision_loss(self, collision_model: nn.Module, optimized_state: MultiRobotPathOptimizedState):
        positions: torch.Tensor = optimized_state.positions
        positions = self.get_intermediate_points(positions)
        positions = positions.reshape(positions.shape[0], -1)
        return torch.mean(torch.nn.functional.softplus(collision_model(positions)))

    def get_intermediate_points(self, positions):
        t = torch.rand(positions.shape[0] - 1, device=self._device)
        delta = positions[1:] - positions[:-1]
        delta[:, :, 2] = wrap_angles(delta[:, :, 2])
        return positions[1:] + t[:, None, None] * delta

    def _direction_constraint_loss(self, optimized_state: MultiRobotPathOptimizedState):
        deltas = self.non_holonomic_constraint_deltas(optimized_state.positions)
        constraint_deltas = torch.mean(deltas ** 2)
        self._metric_manager.add_metric('constraint_deltas', math.sqrt(constraint_deltas.item()))
        return self._parameters.direction_constraint_weight * constraint_deltas + torch.mean(
            optimized_state.direction_constraint_multipliers * deltas)

    @staticmethod
    def non_holonomic_constraint_deltas(positions):
        dx = positions[1:, :, 0] - positions[:-1, :, 0]
        dy = positions[1:, :, 1] - positions[:-1, :, 1]
        angles = positions[:, :, 2]
        delta_angles = wrap_angles(angles[1:] - angles[:-1])
        mean_angles = angles[:-1] + delta_angles / 2
        return dx * torch.sin(mean_angles) - dy * torch.cos(mean_angles)

    @staticmethod
    def _second_differences_loss(optimized_state: MultiRobotPathOptimizedState):
        return torch.mean(
            (optimized_state.positions[2:] - 2 * optimized_state.positions[1:-1] + optimized_state.positions[:-2]) ** 2)


class GradPreconditioner:
    def __init__(self, device, velocity_hessian_weight: float):
        self._device = device
        self._velocity_hessian_weight = velocity_hessian_weight
        self._inverse_hessian = None

    def precondition(self, optimized_state: MultiRobotPathOptimizedState):
        point_count = optimized_state.positions_.shape[0]
        if self._inverse_hessian is None:
            self._inverse_hessian = self._calculate_inv_hessian(point_count)
        reshaped_grad = optimized_state.positions_.grad.reshape(point_count, -1)
        reshaped_grad = self._inverse_hessian @ reshaped_grad
        optimized_state.positions_.grad = reshaped_grad.reshape(optimized_state.positions_.grad.shape)

    def _calculate_inv_hessian(self, point_count):
        hessian = self._velocity_hessian_weight * self._calculate_velocity_hessian(point_count) + np.eye(point_count)
        inv_hessian = np.linalg.inv(hessian)
        return torch.tensor(inv_hessian.astype(np.float32), device=self._device)

    @staticmethod
    def _calculate_velocity_hessian(point_count):
        hessian = np.zeros((point_count, point_count), dtype=np.float32)
        for i in range(point_count):
            if i == 0:
                hessian[i, i] = 2
            elif i == point_count - 1:
                hessian[i, i] = 2
            else:
                hessian[i, i] = 4
            if i > 0:
                hessian[i, i - 1] = -2
                hessian[i - 1, i] = -2
        return hessian

    @staticmethod
    def _calculate_acceleration_hessian(point_count):
        hessian = np.zeros((point_count, point_count))

        for i in range(0, point_count):
            if i == 0:
                hessian[i, i] = 2
            elif i == 1:
                hessian[i, i] = 10
            elif i == point_count - 1:
                hessian[i, i] = 2
            elif i == point_count - 2:
                hessian[i, i] = 10
            else:
                hessian[i, i] = 12
            if i == 1:
                hessian[i, i - 1] = -4
                hessian[i - 1, i] = -4
            elif i == point_count - 1:
                hessian[i, i - 1] = -4
                hessian[i - 1, i] = -4
            else:
                hessian[i, i - 1] = -8
                hessian[i - 1, i] = -8
            if i > 2:
                hessian[i, i - 2] = 2
                hessian[i - 2, i] = 2

        return hessian


class PathOptimizer:
    def __init__(self, timer: Timer, optimizer: OptimizerWithLagrangeMultipliers,
                 loss_builder: MultiRobotPathLossBuilder,
                 state_initializer: PathOptimizedStateInitializer, grad_preconditioner: GradPreconditioner):
        self._loss_builder = loss_builder
        self._optimizer = optimizer
        self._timer = timer
        self._grad_preconditioner = grad_preconditioner
        self._state_initializer = state_initializer
        self._optimized_state: Optional[MultiRobotPathOptimizedState] = None

    def setup(self):
        self._optimized_state = self._state_initializer.init()
        self._optimizer.setup(self._optimized_state.optimized_parameters,
                              [self._optimized_state.direction_constraint_multipliers])

    def step(self, collision_model):
        self._optimizer.zero_grad()
        loss = self._loss_builder.get_loss(collision_model, self._optimized_state)
        self._timer.tick("trajectory_backward")
        loss.backward()
        self._timer.tock("trajectory_backward")
        self._timer.tick("inv_hes_grad_multiplication")
        self._grad_preconditioner.precondition(self._optimized_state)
        self._timer.tock("inv_hes_grad_multiplication")
        self._timer.tick("trajectory_optimizer_step")
        self._optimizer.step()
        self._timer.tock("trajectory_optimizer_step")

    def reparametrize(self):
        with torch.no_grad():
            self._optimized_state.reparametrize()

    @property
    def result_path(self) -> MultiRobotResultPath:
        return self._optimized_state.result_path


class CollisionModelPointSampler:
    def __init__(self, fine_random_offset: float, course_random_offset: float, angle_random_offset, point_count: int):
        self._fine_random_offset = fine_random_offset
        self._course_random_offset = course_random_offset
        self._angle_random_offset = angle_random_offset
        self._point_count = point_count
        self._positions = None

    def sample(self, result_path: MultiRobotResultPath) -> np.ndarray:
        positions: np.ndarray = result_path.numpy_positions
        if self._positions is None:
            self._positions = np.zeros((0, positions.shape[1], positions.shape[2]))
        points = positions[:, :, :2]
        angles = positions[:, :, 2]

        course_points = points + np.random.randn(*points.shape) * self._course_random_offset
        fine_points = points + np.random.randn(*points.shape) * self._fine_random_offset
        points = np.concatenate([course_points, fine_points], axis=0)
        angles = np.concatenate([angles, angles], axis=0) + np.random.randn(2 * angles.shape[0],
                                                                            angles.shape[1]) * self._angle_random_offset

        positions = np.concatenate([points, angles[:, :, None]], axis=2)
        self._positions = np.concatenate([self._positions, positions], axis=0)
        if self._positions.shape[0] > self._point_count:
            self._positions = self._positions[-self._point_count:]
        return positions


@dataclasses.dataclass
class ONFModelConfig:
    mean: float
    sigma: float
    input_dimension: int
    encoding_dimension: int
    output_dimension: int
    hidden_dimensions: List[int]


class ONF(nn.Module):
    def __init__(self, parameters: ONFModelConfig):
        super().__init__()
        self.encoding_layer = nn.Linear(parameters.input_dimension, parameters.encoding_dimension)
        self.mlp1 = self.make_mlp(parameters.encoding_dimension, parameters.hidden_dimensions[:-1],
                                  parameters.hidden_dimensions[-1])
        self.mlp2 = self.make_mlp(parameters.encoding_dimension + parameters.hidden_dimensions[-1], [],
                                  parameters.output_dimension)
        self._mean = parameters.mean
        self._sigma = parameters.sigma

    @classmethod
    def make_mlp(cls, input_dimension, hidden_dimensions, output_dimension):
        modules = []
        for dimension in hidden_dimensions:
            modules.append(nn.Linear(input_dimension, dimension))
            modules.append(nn.ReLU())
            input_dimension = dimension
        modules.append(nn.Linear(input_dimension, output_dimension))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = (x - self._mean) / self._sigma
        x = self.encoding_layer(x)
        x = torch.sin(x)
        input_x = x
        x = self.mlp1(input_x)
        x = torch.cat([x, input_x], dim=1)
        x = self.mlp2(x)
        return x


class CollisionModelFactory:
    def __init__(self, parameters: ONFModelConfig, device):
        self._device = device
        self._parameters = parameters

    def make_collision_model(self):
        return ONF(self._parameters).to(self._device)


class CollisionNeuralFieldModelTrainer:
    def __init__(self, timer: Timer, planner_task: MultiRobotPathPlannerTask,
                 optimizer: OptimizerImpl, collision_model_factory: CollisionModelFactory,
                 device: str):
        self._timer = timer
        self._optimizer = optimizer
        self._collision_model_factory = collision_model_factory
        self._planner_task = planner_task
        self._collision_model: Optional[nn.Module] = None
        self._device = device
        self._collision_loss_function = nn.BCEWithLogitsLoss()
        self._timer = timer

    def setup(self):
        self._collision_model = self._collision_model_factory.make_collision_model()
        # self._collision_model = torch.compile(self._collision_model)
        self._optimizer.setup(self._collision_model.parameters())

    def learning_step(self, points):
        self._timer.tick("optimize_collision_model")
        self._collision_model.requires_grad_(True)
        self._optimizer.zero_grad()
        predicted_collision = self._calculate_predicted_collision(points)
        truth_collision = self._calculate_truth_collision(points)
        truth_collision = torch.tensor(truth_collision.astype(np.float32), device=self._device)
        loss = self._collision_loss_function(predicted_collision, truth_collision)
        loss.backward()
        self._optimizer.step()
        self._collision_model.requires_grad_(False)
        self._timer.tock("optimize_collision_model")

    def _calculate_predicted_collision(self, positions: np.array):
        positions = positions.reshape(positions.shape[0], -1)
        result = self._collision_model(torch.tensor(positions.astype(np.float32), device=self._device))
        return result

    def _calculate_truth_collision(self, positions: np.array):
        result = self._planner_task.collision_detector.is_collision_for_each_robot_for_list(
            [PositionArray2D.from_vec(x) for x in positions])
        return result

    @property
    def collision_model(self) -> nn.Module:
        return self._collision_model


class WarehouseNFOMP:
    def __init__(self, planner_task: MultiRobotPathPlannerTask,
                 collision_neural_field_model_trainer: CollisionNeuralFieldModelTrainer, path_optimizer: PathOptimizer,
                 collision_model_point_sampler: CollisionModelPointSampler,
                 iterations: int, reparametrize_rate: int, collision_model_optimization_rate):
        self._reparametrize_rate = reparametrize_rate
        self.planner_task = planner_task
        self._path_optimizer = path_optimizer
        self._collision_neural_field_model_trainer = collision_neural_field_model_trainer
        self._collision_model_point_sampler = collision_model_point_sampler
        self._iterations = iterations
        self._collision_model_optimization_rate = collision_model_optimization_rate
        self._current_iteration = 0

    def plan(self) -> MultiRobotResultPath:
        self.setup()
        for i in tqdm(range(self._iterations)):
            self.step()
        self._path_optimizer.reparametrize()
        return self.get_result()

    def setup(self):
        self._path_optimizer.setup()
        self._collision_neural_field_model_trainer.setup()
        self._current_iteration = 0

    def get_result(self) -> MultiRobotResultPath:
        return self._path_optimizer.result_path

    def step(self):
        path: MultiRobotResultPath = self._path_optimizer.result_path
        points = self._collision_model_point_sampler.sample(path)
        if self._current_iteration % self._collision_model_optimization_rate == 0:
            self._collision_neural_field_model_trainer.learning_step(points)
        collision_model = self._collision_neural_field_model_trainer.collision_model
        self._path_optimizer.step(collision_model)
        if self._reparametrize_rate != -1 and self._current_iteration % self._reparametrize_rate == 0:
            self._path_optimizer.reparametrize()
        self._current_iteration += 1

    def stop(self):
        pass


class CustomQueue:
    def __init__(self, maximal_previous_result_retry_count: int = None):
        self._queue = Queue()
        self._previous_result = None
        self._maximal_previous_result_retry_count = maximal_previous_result_retry_count
        self._previous_result_retry_count = 0

    def should_use_previous_result(self):
        if self._previous_result is None:
            return False
        if self._maximal_previous_result_retry_count is None:
            return True
        return self._previous_result_retry_count < self._maximal_previous_result_retry_count

    def get(self, block=True, timeout=None):
        queue_size = self._queue.qsize()
        if queue_size == 0 and self.should_use_previous_result():
            self._previous_result_retry_count += 1
            return self._previous_result
        for i in range(queue_size - 1):
            self._queue.get()
        self._previous_result = self._queue.get(block, timeout)
        self._previous_result_retry_count = 1
        return self._previous_result

    def put(self, item, block=True, timeout=None):
        self._queue.put(item, block, timeout)


class MultiProcessWarehouseNFOMP(WarehouseNFOMP):
    def __init__(self, planner_task: MultiRobotPathPlannerTask,
                 collision_neural_field_model_trainer: CollisionNeuralFieldModelTrainer, path_optimizer: PathOptimizer,
                 collision_model_point_sampler: CollisionModelPointSampler,
                 iterations: int, reparametrize_rate: int):
        super().__init__(planner_task, collision_neural_field_model_trainer, path_optimizer,
                         collision_model_point_sampler, iterations,
                         reparametrize_rate)
        self._path_queue = CustomQueue()
        self._collision_model_queue = CustomQueue(10)
        self._points_sampler_queue = CustomQueue()
        self._current_iteration = 0
        self._points_sampler_process: Optional[Process] = None
        self._collision_model_process: Optional[Process] = None

    def setup(self):
        super().setup()
        self._path_queue.put(self._path_optimizer.result_path)
        self._collision_model_process = Process(target=self.collision_model_function)
        self._collision_model_process.start()
        self._points_sampler_process = Process(target=self.points_sampler_function)
        self._points_sampler_process.start()

    def collision_model_function(self):
        while True:
            result = self.collision_model_step()
            if not result:
                break

    def collision_model_step(self):
        points = self._points_sampler_queue.get()
        if points is None:
            return False
        self._collision_neural_field_model_trainer.learning_step(points)
        collision_model = self._collision_neural_field_model_trainer.collision_model
        self._collision_model_queue.put(collision_model)
        return True

    def points_sampler_function(self):
        while True:
            result = self.points_sampler_step()
            if not result:
                break

    def points_sampler_step(self):
        path = self._path_queue.get()
        if path is None:
            self._points_sampler_queue.put(None)
            return False
        points = self._collision_model_point_sampler.sample(path)
        self._points_sampler_queue.put(points)
        return True

    def step(self):
        self.path_step()

    def path_step(self):
        collision_model = self._collision_model_queue.get()
        if collision_model is None:
            return False
        self._path_optimizer.step(collision_model)
        if self._reparametrize_rate != -1 and self._current_iteration % self._reparametrize_rate == 0:
            self._path_optimizer.reparametrize()
        path: MultiRobotResultPath = self._path_optimizer.result_path
        self._path_queue.put(path)
        self._current_iteration += 1
        return True

    def stop(self):
        self._path_queue.put(None)
        self._points_sampler_process.join()
        self._collision_model_process.join()
