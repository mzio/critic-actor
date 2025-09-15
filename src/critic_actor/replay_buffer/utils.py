"""
Replay buffer utilities
"""
from typing import Any

import torch


def _can_itemize(k: str, v: Any) -> bool:
    """
    Check if a value should be itemized (e.g., tensor with shape (1,))
    """
    return isinstance(v, torch.Tensor) and "embed" not in k


def compute_returns_with_last_value(
    rewards: torch.Tensor,
    discount_factor: float = 0.9,
    dones: list[bool] | torch.Tensor | None = None,
    last_value: float = 0.0,) -> torch.Tensor:
    """
    Compute returns from rewards with last value
    - Applied over a single rollout
    """
    returns = torch.zeros_like(rewards).float()  # shape is T
    _return = last_value
    for t in reversed(range(len(rewards))):
        mask = not dones[t] if dones is not None else 1
        _return = rewards[t] + discount_factor * _return * mask
        returns[t] = _return
    return returns
