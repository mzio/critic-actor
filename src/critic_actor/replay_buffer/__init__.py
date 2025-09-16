"""
Replay buffer
"""
from typing import Any

from .base import ReplayBuffer
from .critic_actor import CriticActorReplayBuffer


def get_replay_buffer(name: str, **kwargs: Any,) -> ReplayBuffer:
    """
    Get replay buffer
    """
    if name == "critic_actor":
        return CriticActorReplayBuffer(**kwargs)

    raise ValueError(f"Sorry invalid replay buffer: '{name}'.")


def load_replay_buffer(name: str, **kwargs: Any,) -> ReplayBuffer:
    """
    Alias for get_replay_buffer
    """
    return get_replay_buffer(name, **kwargs)


__all__ = [
    "get_replay_buffer",
    "load_replay_buffer",
    "ReplayBuffer",
    "CriticActorReplayBuffer",
]
