"""
Load model objects
"""
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from ..llm import load_llm
from .huggingface import load_model_and_tokenizer


def load_model(
    name: str,
    model_type: str,
    model_config: dict[str, Any],
    is_async: bool = False,
    generation_config: dict[str, Any] | None = None,
) -> tuple[
    PreTrainedModel | AutoModelForCausalLM,
    None | AutoTokenizer
]:
    """
    Load model based on config (see `configs/model/*.yaml`)
    """
    if model_type == "huggingface":
        return load_model_and_tokenizer(name, model_config)

    if model_type == "llm":
        return load_llm(name, model_config, is_async)

    else:
        raise ValueError(f"Invalid model type: {model_type}")
