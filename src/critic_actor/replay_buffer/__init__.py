"""
Replay buffer
"""
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict
from rich import print as rich_print
from rich.panel import Panel
import torch
from datasets import Dataset as HFDataset, concatenate_datasets, load_from_disk

from .utils import compute_returns_with_last_value


class TorchModel(BaseModel):
    """
    Base model for torch tensors
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)


class EpisodeStep(TorchModel):
    """
    Single episode step
    """
    state: Any
    action: Any
    next_obs: Any | None

    old_logprobs: torch.Tensor
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
    subtask_id: int = 0

    # For ICL copy samples, we repeat tokens twice.
    # This can make logprobs always high, so account for logprobs without context.
    old_logprobs_icl_no_ctx: torch.Tensor | None = None


def advantage_fn(
    returns: torch.Tensor,
    values: torch.Tensor | float = 0.,
) -> torch.Tensor:
    """
    Default / dummy advantage function
    """
    return returns - values


class ReplayBuffer:
    """
    Replay buffer
    """
    step_cls = EpisodeStep  # override in subclasses

    def __init__(
        self,
        discount_factor: float = 0.9,
        normalize_returns: bool = False,
        normalize_advantages: bool = False,
        negative_returns: bool = True,
        tensor_device: str = "cpu",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self.advantage_fn = advantage_fn

        self.discount_factor = discount_factor
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        self.negative_returns = negative_returns

        # Initialize default state
        # -> We append to EpisodeStep objects this during sampling
        # -> Then we create some mappings for faster retrieval
        self.episode_steps: list[self.step_cls] = []
        # self._tensor_types: dict[str, torch.dtype] = {}
        self._tensor_casts: dict[str, torch.dtype] = {}
        self._tensor_cache: dict[str, torch.Tensor] = {}  # cached tensors
        self._dirty = True  # tensors not cached yet

        self.tensor_device = tensor_device
        self.register_tensors()  # populate self._tensor_dtype

        self.verbose = verbose  # print additional info

    def __len__(self):
        """Get size of replay buffer"""
        return len(self.episode_steps)

    def add(self, **kwargs: Any) -> None:
        """
        Add a step to the replay buffer
        """
        step = self.step_cls(**kwargs)
        self.episode_steps.append(step)
        self._dirty = True

    def add_step(self, step: EpisodeStep) -> None:
        """
        Add a step to the replay buffer
        """
        assert isinstance(step, self.step_cls)
        self.episode_steps.append(step)
        self._dirty = True

    def reset_buffer(self) -> None:
        """
        Initialize or reset replay buffer
        """
        self.episode_steps: list[self.step_cls] = []
        # self._tensor_types.clear()  # We keep this mapping
        self._tensor_cache.clear()
        self._dirty = True

    def _register_as_tensor(self, name: str, dtype: torch.dtype) -> None:
        """
        Register a tensor attribute
        """
        self._tensor_casts[name] = dtype
        self._dirty = True

    def register_tensors(self) -> None:
        """
        Register default tensors
        """
        self._register_as_tensor("reward", torch.float32)
        self._register_as_tensor("done", torch.bool)
        # self._register_as_tensor("old_logprobs", torch.float32)
        self._register_as_tensor("temperature", torch.float32)

        self._register_as_tensor("timestep", torch.long)
        self._register_as_tensor("try_step", torch.long)

        self._register_as_tensor("sample_id", torch.long)
        self._register_as_tensor("batch_id", torch.long)
        self._register_as_tensor("data_sample_id", torch.long)
        self._register_as_tensor("generation_id", torch.long)
        self._register_as_tensor("is_train", torch.bool)

        self._register_as_tensor("return_", torch.float32)
        self._register_as_tensor("advantage", torch.float32)
        self._register_as_tensor("value", torch.float32)
        self._register_as_tensor("return_is_computed", torch.bool)
        self._register_as_tensor("advantage_is_computed", torch.bool)

        self._register_as_tensor("subtask_id", torch.long)

    def _ensure_tensors(self) -> None:
        """
        Ensure tensor attributes are cached and retrievable
        """
        if not self._dirty:
            return  # everything ensured

        self._tensor_cache.clear()
        # For each tensor attribute, combines the values for all steps
        for name, dtype in self._tensor_casts.items():
            # if name not in ["old_logprobs"]:  # hack but only consider tensors of 
            vals = [getattr(step, name) for step in self.episode_steps]
            self._tensor_cache[name] = torch.tensor(
                vals, dtype=dtype, device=self.tensor_device,
            )
        self._dirty = False

    def compute_returns(
        self,
        last_value: float,
        sample_id: int,
        is_train: bool,
        try_step: int | None = None,
        discount_factor: float | None = None,
        print_returns: bool = False,
        subtask_id: int = 0,
        is_icl_copy: bool = False,
    ) -> torch.Tensor:
        """
        Compute returns from saved rewards.
        """
        self._ensure_tensors()
        mask = (
            (self._tensor_cache["sample_id"] == sample_id)
            & (self._tensor_cache["is_train"] == is_train)
            & (self._tensor_cache["subtask_id"] == subtask_id)
            & (self._tensor_cache["is_icl_copy"] == is_icl_copy)
        )
        if try_step is not None:
            mask &= (self._tensor_cache["try_step"] == try_step)

        rewards = self._tensor_cache["reward"][mask]
        dones   = self._tensor_cache["done"][mask]
        discount_factor = discount_factor or self.discount_factor

        if self.negative_returns and rewards.min() == 0:
            rewards[-1] = rewards[-1] * 2 - 1  # hack, assumes only get reward at end

        returns = compute_returns_with_last_value(
            rewards, discount_factor, dones, last_value,
        )
        keep_indices = torch.nonzero(mask, as_tuple=True)[0]
        for _idx, keep_idx in enumerate(keep_indices):
            self.episode_steps[keep_idx].return_ = returns[_idx].item()
            self.episode_steps[keep_idx].return_is_computed = True
        self._tensor_cache["return_"][keep_indices] = returns
        self._tensor_cache["return_is_computed"][keep_indices] = True

        if print_returns or self.verbose:
            title = (
                "replay_buffer.compute_returns"
                f"(sample_id: {sample_id}, is_train: {is_train}, try_step: {try_step})"
            )
            rich_print(Panel(
                f"Returns: [cyan]{returns}[/cyan]\n"
                f"Rewards: [cyan]{rewards}[/cyan]\n"
                f"Dones: [cyan]{dones}[/cyan]\n",
                title=title,
                border_style="cyan",
            ))
        return returns

    def compute_advantages(
        self,
        data_sample_id: int,
        is_train: bool,
        try_step: int | None = None,
        print_advantages: bool = False,
        subtask_id: int = 0,
        is_icl_copy: bool = False,
    ) -> torch.Tensor:
        """
        Compute advantages from saved returns.
        """
        self._ensure_tensors()
        mask = (
            (self._tensor_cache["data_sample_id"] == data_sample_id)
            & (self._tensor_cache["is_train"] == is_train)
            & (self._tensor_cache["subtask_id"] == subtask_id)
            & (self._tensor_cache["is_icl_copy"] == is_icl_copy)
        )
        if try_step is not None:
            mask &= (self._tensor_cache["try_step"] == try_step)
        returns = self._tensor_cache["return_"][mask]
        values = self._tensor_cache["value"][mask]

        # By default, advantages is just returns - values
        advantages = self.advantage_fn(returns, values=values)
        keep_indices = torch.nonzero(mask, as_tuple=True)[0]
        for _idx, keep_idx in enumerate(keep_indices):
            self.episode_steps[keep_idx].advantage = advantages[_idx].item()
            self.episode_steps[keep_idx].advantage_is_computed = True
        self._tensor_cache["advantage"][keep_indices] = advantages
        self._tensor_cache["advantage_is_computed"][keep_indices] = True

        if print_advantages or self.verbose:
            title = (
                "replay_buffer.compute_advantages ("
                f"data_sample_id: {data_sample_id}, is_train: {is_train}, "
                f"try_step: {try_step}, subtask_id: {subtask_id})"
            )
            rich_print(Panel(
                f"Advantages: [green]{advantages}[/green]\n"
                f"Returns: [green]{returns}[/green]\n"
                f"Values: [green]{values}[/green]\n",
                title=title,
                border_style="green",
            ))
        return advantages

    # ---------- masking ----------
    def get_keep_indices(
        self,
        **filters: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build a boolean mask from filters. Supports single values or iterables.
        Returns tuple of (keep_indices, mask)
        """
        self._ensure_tensors()
        mask = torch.ones(len(self), dtype=torch.bool)  # for all buffer samples

        # build mask based on filters
        for k, v in filters.items():
            # for now, assume v is not an Iterable
            if v is None:
                continue
            if k.startswith("not_"):
                k = k[len("not_"):]
                mask &= (self._tensor_cache[k] != v)
            elif k.startswith("leq_"):
                k = k[len("leq_"):]
                mask &= (self._tensor_cache[k] <= v)
            elif k.startswith("geq_"):
                k = k[len("geq_"):]
                mask &= (self._tensor_cache[k] >= v)
            else:
                mask &= (self._tensor_cache[k] == v)
                # _tensors = self._tensor_cache[k]
                # _mask = self._maybe_isin(_tensors, v)
                # if _mask is not None:
                #     mask &= _mask
        return mask.nonzero(as_tuple=True)[0], mask  # LongTensor[K], BoolTensor[N]

    # ---------- gathering ----------
    def get_data_dict(self, **filters: Any) -> tuple[dict[str, Any], torch.Tensor]:
        """
        Returns a dict of filtered columns.
        Numeric/meta -> tensor indexed.
        Ragged       -> list indexed by keep_indices.
        Also returns the keep_indices for further slicing.
        """
        keep_indices, mask = self.get_keep_indices(**filters)
        self._ensure_tensors()

        # Get all tensor attributes
        filtered_dict = {
            name: self._tensor_cache[name][keep_indices]
            for name in self._tensor_cache
        }

        if self.verbose:
            title = "replay_buffer.get_data_dict"
            _filters_str = str({k: v for k, v in filters.items()})
            _buffer_str  = str({k: v.shape for k, v in self._tensor_cache.items()})
            _filtered_buffer_str = str({k: v.shape for k, v in filtered_dict.items()})
            rich_print(Panel(
                f"Filters: [magenta]{_filters_str}[/magenta]\n"
                f"Buffer: [green]{_buffer_str}[/green]\n"
                f"Filtered Buffer: [cyan]{_filtered_buffer_str}[/cyan]\n",
                title=title,
                border_style="cyan",
            ))

        # Optional normalization on the *filtered subset*
        ret = filtered_dict["return_"]
        adv = filtered_dict["advantage"]
        if self.normalize_returns and ret.numel():
            mu, sd = ret.mean(), ret.std(unbiased=False)
            ret = (ret - mu) / (sd + 1e-8)
        if self.normalize_advantages and adv.numel():
            mu, sd = adv.mean(), adv.std(unbiased=False)
            adv = (adv - mu) / (sd + 1e-8)

        # Gather ragged/object columns (indexing lists)
        ix = keep_indices.tolist()
        try:
            filtered_dict.update(dict(
                state        = [self.episode_steps[i].state for i in ix],
                action       = [self.episode_steps[i].action for i in ix],
                next_obs     = [self.episode_steps[i].next_obs for i in ix],
                old_logprobs = [self.episode_steps[i].old_logprobs.detach().cpu() for i in ix],
                keep_indices = keep_indices,
            ))
            filtered_dict["old_logprobs_icl_no_ctx"] = [
                self.episode_steps[i].old_logprobs_icl_no_ctx.detach().cpu() for i in ix
            ]
        except AttributeError as e:
            print("old_logprobs", self.episode_steps[0].old_logprobs)
            print(e)
            breakpoint()
        return filtered_dict, mask

    def get_data_steps(self, **filters) -> tuple[list[EpisodeStep], torch.Tensor]:
        """
        Consolidate filtered data into a list of EpisodeStep objects.
        Returns (list[EpisodeStep], filtering_mask)
        """
        data_dict, mask = self.get_data_dict(**filters)
        out = [
            self.step_cls(**{k: v[i] for k, v in data_dict.items()})
            for i in range(len(data_dict["state"]))
        ]
        return out, mask

    def get_data(self, **filters: Any) -> tuple[list[EpisodeStep], torch.Tensor]:
        """
        Alias for get_data_steps
        """
        return self.get_data_steps(**filters)

    def save_to_hf_dataset(self, save_path: str, **kwargs: Any) -> None:
        """
        Save the replay buffer to a Hugging Face dataset
        """
        data_dict, _ = self.get_data_dict(**kwargs)
        HFDataset.from_dict(data_dict).save_to_disk(save_path)
        self._ensure_tensors()

    def load_from_hf_datasets(self, save_paths: list[str]) -> None:
        """
        Load the replay buffer from a Hugging Face dataset
        """
        self.reset_buffer()
        datasets = [load_from_disk(path) for path in save_paths]
        dataset = concatenate_datasets([
            # HFDataset.load_from_disk(path) for path in save_paths
            dataset["train"] if dataset.get("train", None) is not None else dataset
            for dataset in datasets
        ])
        _tensor_keys = list(self._tensor_casts.keys())
        for data_dict in dataset:
            data_dict = {  # cast to correct dtype
                k: torch.tensor(v, dtype=self._tensor_casts[k])
                if k in _tensor_keys
                else v
                for k, v in data_dict.items()
            }
            data_dict["old_logprobs"] = torch.tensor(data_dict["old_logprobs"])
            # episode_step = self.step_cls(**data_dict)
            # self.add(episode_step)
            self.add(**data_dict)
