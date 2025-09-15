"""
Sampling classes for LLMs
"""
from typing import Any

from .base import ActionFromLLM, LLM
from .openai import (
    OpenAIResponsesLLM, AsyncOpenAIResponsesLLM, Response,
)


def load_llm(
    name: str,
    model_config: dict[str, Any],
    is_async: bool = False,
    model_type: str = "llm",  # ignored
) -> LLM:
    """
    Load LLM
    """
    if name == "openai":
        if is_async:
            return AsyncOpenAIResponsesLLM(**model_config)
        else:
            return OpenAIResponsesLLM(**model_config)
    else:
        raise ValueError(f"Invalid model name: {name}")


__all__ = [
    "load_llm",
    "ActionFromLLM",
    "LLM",
    "Response",
    "OpenAIResponsesLLM",
    "AsyncOpenAIResponsesLLM",
]
