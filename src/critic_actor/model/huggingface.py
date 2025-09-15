"""
Pretrained model loading from Hugging Face
"""
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    name: str,
    model_config: dict[str, Any],
    update_tokenizer: bool = True,
    generation_config: dict[str, Any] | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer
    - Automatically sets tokenizer up for language modeling
    """
    if name == "hf_transformer":
        model = AutoModelForCausalLM.from_pretrained(**model_config)
        tokenizer = AutoTokenizer.from_pretrained(**model_config)
    else:
        raise ValueError('Sorry name "{name}" not implemented!')

    if update_tokenizer:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    return model, tokenizer
