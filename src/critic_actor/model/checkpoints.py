"""
Model checkpointing and weight saving
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from ..utils.logging import print_header


def save_trainable_weights(
    model: nn.Module | PreTrainedModel,
) -> OrderedDict:
    """
    Save checkpoint with only weights actively being trained (e.g., for adapters).
    Make sure to later load with model.load_state_dict(state_dict, strict=False)
    """
    with torch.no_grad():
        state_dict = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p.cpu()  # assurance
        return state_dict


def load_model_checkpoint(
    model: nn.Module | PreTrainedModel,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[
    nn.Module | PreTrainedModel,
    torch.optim.Optimizer | None, 
    torch.optim.lr_scheduler.LRScheduler | None,
]:
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    _keys = model.load_state_dict(state_dict, strict=False)

    try:  # Check that all expected keys matched successfully
        assert len(_keys.unexpected_keys) == 0
        print_header("*** All expected keys matched successfully ***")
        print(f"-> Loaded checkpoint from {checkpoint_path}")
        print(f"-> Last grad step: {checkpoint['grad_step']}")
    except Exception as e:
        print(e)
        print_header("*** Error: unexpected keys in checkpoint ***")
        print("Unexpected keys:")
        for k in _keys.unexpected_keys:
            print(k)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler
