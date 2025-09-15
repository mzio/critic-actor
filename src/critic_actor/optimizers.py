"""
Optimizers and schedulers
"""
from typing import Any

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    optim: str = "sgd",
    **kwargs: Any,
) -> Any:  # torch.optim.Optimizer | HuggingFaceOptimizer
    """
    Return PyTorch or Hugging Face optimizer
    """
    _parameters = [p for p in model.parameters() if p.requires_grad]
    if optim == "sgd":
        return torch.optim.SGD(_parameters, **kwargs)
    elif optim == "adam":
        return torch.optim.Adam(_parameters, **kwargs)
    elif optim in ["adamw", "adamw_torch"]:
        return torch.optim.AdamW(_parameters, **kwargs)
    elif optim == "adamw_torch_fused":
        return torch.optim.AdamW(_parameters, **kwargs, fused=True)
    elif optim == "adafactor":
        from transformers.optimization import Adafactor
        kwargs["relative_step"] = False  # for now
        return Adafactor(_parameters, **kwargs)
    else:
        raise NotImplementedError(f"Sorry, {optim} optimizer not implemented.")


def get_scheduler(
    optimizer: Any,
    lr_scheduler_type: str = "none",
    **kwargs: Any,
) -> Any:  # torch.optim.lr_scheduler.LRScheduler | HuggingFaceScheduler
    """
    Return PyTorch or Hugging Face scheduler
    """
    if lr_scheduler_type in ["plateau", "reduce_lr_on_plateau"]:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(optimizer=optimizer, **kwargs)

    elif lr_scheduler_type == "cosine_warmup":
        from transformers.optimization import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(optimizer=optimizer, **kwargs)

    elif lr_scheduler_type in ["linear_warmup", "linear"]:
        from transformers.optimization import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(optimizer=optimizer, **kwargs)

    else:
        return None
