import numpy as np
import torch


def wrap_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def interpolate_1d_pytorch(x: torch.Tensor, old_times: torch.Tensor, new_times: torch.Tensor) -> torch.Tensor:
    upper_indices: torch.Tensor = torch.searchsorted(old_times, new_times)
    lower_indices: torch.Tensor = upper_indices - 1
    lower_indices = torch.where(lower_indices > 0, lower_indices, 0)
    upper_indices = torch.where(upper_indices < x.shape[0], upper_indices, x.shape[0] - 1)
    lower_old_times = torch.gather(old_times, 0, lower_indices)
    upper_old_times = torch.gather(old_times, 0, upper_indices)
    lower_x: torch.Tensor = x[lower_indices]
    upper_x: torch.Tensor = x[upper_indices]
    denominator = upper_old_times - lower_old_times
    denominator = torch.where(torch.abs(denominator) > 1e-6, denominator, torch.full_like(denominator, 1e-6))
    weights = (new_times - lower_old_times) / denominator
    weights = weights[:, None]
    interpolated_values = (1 - weights) * lower_x + weights * upper_x
    return interpolated_values


def unfold_angles(angles: torch.Tensor):
    angles: torch.Tensor = wrap_angles(angles)
    delta: torch.Tensor = angles[1:] - angles[:-1]
    delta = torch.where(delta > np.pi, delta - 2 * np.pi, delta)
    delta = torch.where(delta < -np.pi, delta + 2 * np.pi, delta)
    if len(angles.shape) == 1:
        return angles[0] + torch.cat([torch.zeros(1), torch.cumsum(delta, dim=0)], dim=0)
    return angles[0, None] + torch.cat([torch.zeros(1, delta.shape[1]), torch.cumsum(delta, dim=0)],
                                       dim=0)
