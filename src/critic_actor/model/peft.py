"""
Parameter-efficient fine-tuning
- For now, only supports LoRA
"""
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from omegaconf import DictConfig

from peft import LoraConfig, TaskType, get_peft_model
# Enable and disable adapters
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.peft_utils import check_peft_version

# Minimum PEFT version supported for the integration
MIN_PEFT_VERSION = "0.5.0"


def create_peft_config(
    model: AutoModelForCausalLM | PreTrainedModel,
    peft_config: dict[str, Any] | DictConfig,
    adapter_name: str = "default",
    **peft_kwargs: Any,
) -> tuple[
    AutoModelForCausalLM | PreTrainedModel, 
    LoraConfig | dict[Any, Any],
]:
    """
    Create a model ready for LoRA finetuning
    - See ./configs/lora/r8_a16_qkvo.yaml for example
    """
    if peft_config["kwargs"] is not None:
        _peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_config["kwargs"],
        )
        with torch.no_grad():
            model = get_peft_model(
                model, _peft_config, adapter_name=adapter_name,
            )
            model.print_trainable_parameters()
    else:
        _peft_config = {}
    if "finetune" in peft_config:
        with torch.no_grad():
            for n, p in model.named_parameters():
                for _n in peft_config["finetune"]:
                    if _n in n:
                        p.requires_grad = True

    # optional convenience alias
    # model.forward = model.base_model.model.forward
    return model, _peft_config


def count_parameters(
    model: nn.Module,
    requires_grad: bool = True,
) -> int:
    """
    Return total number of trainable parameters
    """
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    return int(sum([np.prod(p.size()) for p in model_parameters]))


def add_peft_lora(
    model: PreTrainedModel, lora_config: LoraConfig, adapter_name: str = "default"
) -> PreTrainedModel:
    """
    Add LoRA to model
    - alias for model.add_adapter(peft_config=_peft_config, adapter_name="act")
    """
    with torch.no_grad():
        model.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
        return model


# Copied and modified from https://github.com/huggingface/transformers/blob/878562b68d06536b475a61496e3c2a26fdb95af1/src/transformers/integrations/peft.py#L365
def disable_adapters(
    model: nn.Module,
    adapter_names: list[str] | None = None,
) -> None:
    """
    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
    official documentation: https://huggingface.co/docs/peft

    Disable all adapters that are attached to the model. This leads to inferring with the base model only.
    """
    check_peft_version(min_version=MIN_PEFT_VERSION)

    for name, module in model.named_modules():
        if adapter_names is not None and name not in adapter_names:
            continue
        if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
            # The recent version of PEFT need to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=False)
            else:
                setattr(module, "disable_adapters", True)


# Copied and modified from https://github.com/huggingface/transformers/blob/878562b68d06536b475a61496e3c2a26fdb95af1/src/transformers/integrations/peft.py#L388
def enable_adapters(
    model: nn.Module, 
    adapter_names: list[str] | None = None,
) -> None:
    """
    If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
    official documentation: https://huggingface.co/docs/peft

    Enable adapters that are attached to the model.
    """
    check_peft_version(min_version=MIN_PEFT_VERSION)

    for name, module in model.named_modules():
        if adapter_names is not None and name not in adapter_names:
            continue
        if isinstance(module, BaseTunerLayer):
            # The recent version of PEFT need to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=True)
            else:
                setattr(module, "disable_adapters", False)
