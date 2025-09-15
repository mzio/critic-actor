"""
Critic-Actor dataloader
"""
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from critic_actor.replay_buffer.critic_actor import CriticActorEpisodeStep


class CriticActorLoader():
    """
    Critic-Actor policy update dataloaders
    """
    def __init__(
        self,
        data: list[CriticActorEpisodeStep],
        dataloader_config: dict[str, Any],
        val_size: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.data = data
        self.dataloader_config = dataloader_config
        self.val_size = val_size
        self.seed = seed

        self.collate_fn = collate_fn

        # Construct splits
        indices = np.arange(len(self.data))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.val_size)
        self.train_indices = indices[split_idx:]
        self.val_indices = indices[:split_idx]
        if len(self.val_indices) == 0:
            self.val_indices = [0, 1, 2]  # hack, but why so heinous? (MZ 9/15/25)

        # Get datasets and dataloaders
        self.train_dataset = CriticActorDataset([self.data[i] for i in self.train_indices])
        self.val_dataset   = CriticActorDataset([self.data[i] for i in self.val_indices])

        self.dataloader_config["shuffle"] = True
        self.dataloader_config["collate_fn"] = self.collate_fn
        self.train_loader = DataLoader(self.train_dataset, **self.dataloader_config)
        self.val_loader   = DataLoader(self.val_dataset, **self.dataloader_config)
        self.eval_loader  = self.val_loader


class CriticActorDataset(Dataset):
    """
    Critic-Actor dataset
    """
    def __init__(self, data: list[CriticActorEpisodeStep]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]
        return {
            "embed_q":      sample.state,
            "embed_k":      sample.all_actions,
            "label":        sample.action_label,
            "old_logprobs": sample.old_logprobs,
            "advantage":    sample.advantage,
            "return_":      sample.return_,
            "temperature":  sample.temperature,
        }

def _pad_2d_list(
    tensors: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of (seq_len, embed_dim) tensors to a
    tensor of shape (batch_size, max_seq_len, embed_dim).
    -> Also return sequence lengths
    """
    assert len(tensors) > 0
    embed_dim = tensors[0].size(-1)
    _kwargs = dict(device=tensors[0].device, dtype=torch.long)
    seq_lens = torch.as_tensor([t.size(0) for t in tensors], **_kwargs)
    max_seq_len = int(seq_lens.max().item())
    out = tensors[0].new_full((len(tensors), max_seq_len, embed_dim), pad_value)
    for i, t in enumerate(tensors):
        out[i, :t.size(0)] = t
    return out, seq_lens


def lengths_to_mask(seq_lens: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    Helper to convert seq lengths to mask tensors
    [batch_size] -> [batch_size, max_seq_len] boolean.
    """
    return torch.arange(
        max_seq_len, device=seq_lens.device,
    )[None, :] < seq_lens[:, None]
    

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Collate function for critic-actor dataset.
    batch: list[dict[str, Any]] -> dict[str, torch.Tensor]
    """
    batch_size = len(batch)
    num_keys = len(batch[0]["embed_k"])

    # queries
    q_list = [b["embed_q"] for b in batch]          # b x (q_len, d)
    q_pad, q_len = _pad_2d_list(q_list)             # (b, max_q_len, d), q_lens
    mask_q = lengths_to_mask(q_len, q_pad.size(1))  # (b, max_q_len)

    # keys: flatten B*N, pad to max_seq_len, then reshape back to
    #       [batch_size, num_keys, max_seq_len, embed_dim]
    flat_k = [k for b in batch for k in b["embed_k"]]  # (b * n) x (k_len, d)
    k_pad_flat, k_len_flat = _pad_2d_list(flat_k)       # (b * n, max_k_len, d), k_lens
    max_seq_len, embed_dim = k_pad_flat.shape[1], k_pad_flat.shape[2]

    k_pad = k_pad_flat.view(batch_size, num_keys, max_seq_len, embed_dim)
    k_len = k_len_flat.view(batch_size, num_keys)
    mask_k = lengths_to_mask(k_len.view(-1), max_seq_len).view(
        batch_size, num_keys, max_seq_len,
    )

    # labels, returns, advantages, temperatures
    _tensor_kwargs = dict(dtype=torch.float, device=q_pad.device)
    label = torch.as_tensor([b["label"] for b in batch], device=q_pad.device).long()
    old_logprobs = torch.stack([b["old_logprobs"].to(**_tensor_kwargs) for b in batch], dim=0)
    advantage = torch.as_tensor([b["advantage"] for b in batch], **_tensor_kwargs)
    return_ = torch.as_tensor([b["return_"] for b in batch], **_tensor_kwargs)
    temperature = torch.as_tensor([b["temperature"] for b in batch], **_tensor_kwargs)

    return {
        "embed_q": q_pad,
        "embed_k": k_pad,
        "mask_q": mask_q,
        "mask_k": mask_k,
        "q_len": q_len,
        "k_len": k_len,
        "label": label,
        "old_logprobs": old_logprobs,
        "advantage": advantage,
        "return_": return_,
        "temperature": temperature,
    }
