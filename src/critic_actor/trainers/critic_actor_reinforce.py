"""
Critic-Actor Trainer with REINFORCE-like training
-> Advantage is just the (discounted) returnß
"""

from typing import Any

import torch

from ..replay_buffer.critic_actor import CriticActorReplayBuffer

from .critic_actor import CriticActorTrainer as BaseCriticActorTrainer


def advantage_fn(
    returns: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Dummy function which just returns the returnsß
    """
    return returns


class CriticActorTrainer(BaseCriticActorTrainer):
    """
    Critic-Actor Trainer
    """
    def __init__(
        self,
        discount_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.discount_factor = discount_factor

    def get_replay_buffer(self, **replay_buffer_config: Any) -> CriticActorReplayBuffer:
        """
        Return a Critic-Actor replay buffer object
        -> By default, we use discount_factor = 1 and GRPO-like "advantage"
        """
        replay_buffer = CriticActorReplayBuffer(**replay_buffer_config)
        replay_buffer.discount_factor = self.discount_factor
        replay_buffer.normalize_returns = False
        replay_buffer.normalize_advantages = False
        replay_buffer.negative_returns = True
        replay_buffer.advantage_fn = advantage_fn
        return replay_buffer
