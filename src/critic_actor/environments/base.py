"""
Parent environment class
References: https://github.com/bentrevett/pytorch-rl/blob/master/5%20-%20Proximal%20Policy%20Optimization%20(PPO)%20%5BCartPole%5D.ipynb
"""

from typing import Any

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from pydantic import BaseModel
from transformers import AutoTokenizer


class EnvironmentState(BaseModel):
    """
    State of the environment after a step
    -> TODO: support Chat Completions objects
    """

    system_prompt: str
    messages: list[dict[str, Any]]
    model_response: (
        str
        | list[dict[str, Any]]
        | Response
        | ResponseReasoningItem
    ) | None
    prior_messages: list[
        dict[str, Any]
        | ResponseOutputMessage
        | ResponseFunctionToolCall
        | ResponseReasoningItem
    ]
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class EnvironmentStateWithAnswer(EnvironmentState):
    """
    State of the environment after a step with the answer
    """

    answer: str


class EnvironmentStepResult(BaseModel):
    """
    Result of a step in the environment
    """

    state: EnvironmentState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]


class Environment:
    """
    Parent class for environment
    """

    def __init__(
        self,
        max_turns: int = 10000,
        max_tries: int = 1,
        pretrained_model_config: dict[str, Any] | None = None,
        tool_role: str = "tool",
        seed: int = 0,
        verbose: bool = False,
        truncation_message: str = "Sorry, too many turns.",
    ) -> None:
        super().__init__()
        self.max_turns = max_turns
        self.max_tries = max_tries
        self.tool_role = tool_role
        self.pretrained_model_config = pretrained_model_config
        self.seed = seed
        self.verbose = verbose

    def __len__(self) -> int:
        raise NotImplementedError

    def _init_tokenizer(self) -> AutoTokenizer | None:
        if self.pretrained_model_config is not None:
            tokenizer = AutoTokenizer.from_pretrained(**self.pretrained_model_config)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = None
        return tokenizer

    def reset(self, sample_idx: int) -> tuple[Any, dict[str, Any]]:
        """
        Reset environment (starting new episode, or working on a new sample)
        """
        raise NotImplementedError

    def step(self, **kwargs: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Perform one step of the environment
        By Gym convention, return (state, reward, done, truncated, info)
        """
        raise NotImplementedError

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle environment's samples (e.g., after going through all during training)
        """
        raise NotImplementedError
