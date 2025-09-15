"""
Critic-Actor Replay Buffer

Store states and actions as tensor embeddings, and
old_logprobs as a distribution over number of actions.
"""
from typing import Any

import torch
from datasets import concatenate_datasets, Dataset as HFDataset

from .base import EpisodeStep, ReplayBuffer


class CriticActorEpisodeStep(EpisodeStep):
    """
    Single episode step
    """
    state: torch.Tensor   # q_embeds
    action: torch.Tensor  # k_embeds
    all_actions: list[torch.Tensor]
    action_label: int

    state_input_ids: torch.Tensor
    action_input_ids: list[torch.Tensor]

    next_obs: Any | None = None  # may not include

    old_logprobs: torch.Tensor   # one for each action
    temperature: float

    reward: float
    done: bool
    timestep: int
    try_step: int

    sample_id: int
    batch_id: int
    data_sample_id: int
    generation_id: int
    is_train: bool         # only train on these samples
    
    return_: float = -2    # these are updated later
    advantage: float = -2  # -2 as a placeholder
    value: float = 0
    return_is_computed: bool = False
    advantage_is_computed: bool = False


def advantage_fn(
    returns: torch.Tensor,
    # values: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute advantages from returns and values
    - use GRPO-like advantage function
    """
    mu, std = returns.mean(), returns.std()
    if std == 0:
        return torch.zeros_like(returns)  # no variance
    return (returns - mu) / std


class CriticActorReplayBuffer(ReplayBuffer):
    """
    Critic-Actor Replay Buffer
    """
    step_cls = CriticActorEpisodeStep
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.advantage_fn = advantage_fn

    def register_tensors(self) -> None:
        """
        Register default tensors
        -> These are scalars we can easily mask and retrieve on
        """
        super().register_tensors()
        self._register_as_tensor("action_label", torch.long)

   # ---------- gathering ----------
    def get_data_dict(self, **filters: Any) -> tuple[dict[str, Any], torch.Tensor]:
        """
        Returns a dict of filtered columns.
        Numeric/meta -> tensor indexed.
        Ragged       -> list indexed by keep_indices.
        Also returns the keep_indices for further slicing.
        """
        # Retrieve parent class data
        filtered_dict, mask = super().get_data_dict(**filters)
        keep_indices = filtered_dict["keep_indices"]
        ix = keep_indices.tolist()

        # Add critic-actor specific data
        filtered_dict.update(dict(
            action_label = [self.episode_steps[i].action_label for i in ix],
            all_actions  = [self.episode_steps[i].all_actions for i in ix],
            state_input_ids = [self.episode_steps[i].state_input_ids for i in ix],
            action_input_ids = [self.episode_steps[i].action_input_ids for i in ix],
        ))
        return filtered_dict, mask

    def get_data_steps(self, **filters) -> tuple[list[CriticActorEpisodeStep], torch.Tensor]:
        """
        Consolidate filtered data into a list of CriticActorEpisodeStep objects.
        Returns (list[CriticActorEpisodeStep], filtering_mask)
        """
        data_dict, mask = self.get_data_dict(**filters)
        print(f"self.step_cls: {self.step_cls}")
        out = [
            CriticActorEpisodeStep(**{k: v[i] for k, v in data_dict.items()})
            for i in range(len(data_dict["state"]))
        ]
        return out, mask

    def get_data(self, **filters: Any) -> tuple[list[CriticActorEpisodeStep], torch.Tensor]:
        """
        Alias for get_data_steps
        """
        return self.get_data_steps(**filters)

    def load_from_hf_datasets(self, save_paths: list[str]) -> None:
        """
        Load the replay buffer from a Hugging Face dataset
        """
        self.reset_buffer()
        dataset = concatenate_datasets([
            HFDataset.load_from_disk(path) for path in save_paths
        ])
        _tensor_keys = list(self._tensor_casts.keys())
        for data_dict in dataset:
            data_dict = {  # cast to correct dtype
                k: torch.tensor(v, dtype=self._tensor_casts[k])
                if k in _tensor_keys
                else v
                for k, v in data_dict.items()
            }
            # Cast "ragged" attributes to tensors
            data_dict["state"] = torch.tensor(data_dict["state"])
            data_dict["action"] = torch.tensor(data_dict["action"])
            data_dict["old_logprobs"] = torch.tensor(data_dict["old_logprobs"])
            data_dict["all_actions"] = [
                torch.tensor(a) for a in data_dict["all_actions"]
            ]
            data_dict["state_input_ids"] = torch.tensor(data_dict["state_input_ids"])
            data_dict["action_input_ids"] = [
                torch.tensor(a) for a in data_dict["action_input_ids"]
            ]
            self.add(**data_dict)
