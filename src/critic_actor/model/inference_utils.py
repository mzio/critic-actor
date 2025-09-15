"""
Helper functions for model inference
-> Computing token log-probs, embeddings, etc.
"""
from copy import deepcopy
from typing import Any, Callable

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer



def _embed_reduce(reduction: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Reduce embeddings along the sequence dimension
    -> Mean or last token
    """
    return (
        lambda x: x.mean(dim=-2, keepdim=True) if reduction == "mean" else \
        lambda x: x[..., -1:, :]
    )


def _next_token_shift(
    logits: torch.Tensor,
    hiddens: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shift logits and hiddens to the next token
    """
    return logits[..., :-1, :], hiddens[..., :-1, :], labels[..., 1:]


def _gather_label_logprobs(
    shifted_logits: torch.Tensor,
    shifted_labels: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """
    Gather log-probs for provided labels
    """
    _dtype = shifted_logits.dtype
    logp_vocab = F.log_softmax(shifted_logits.float() / temperature, dim=-1).to(_dtype)
    label_logp = torch.gather(
        logp_vocab, dim=-1, index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)  # (batch_size, seq_len)
    return label_logp


def process_state_action_inputs(
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    batch_state_action_dicts: list[list[dict[str, Any]]],
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Process batched chat messages (list[dict[str, Any]]) into
    batch of input_ids and attention_masks for model inference
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {
            "add_special_tokens": False,
            "padding": True,
            "padding_side": "right",
            "return_tensors": "pt"
        }
    # Build texts with chat template
    state_texts = [
        tokenizer.apply_chat_template(
            sa[:-1], add_generation_prompt=True, tokenize=False,
        ) for sa in batch_state_action_dicts
    ]
    state_action_texts = [
        tokenizer.apply_chat_template(
            sa, add_generation_prompt=False, tokenize=False,
        ) for sa in batch_state_action_dicts
    ]
    # Tokenize (batched)
    state_inputs = tokenizer(state_texts, **tokenizer_kwargs)
    state_action_inputs = tokenizer(state_action_texts, **tokenizer_kwargs)

    # Get masks
    state_mask = state_inputs["attention_mask"]
    state_action_mask = state_action_inputs["attention_mask"]
    # We do state_action_mask - state_mask to get the action masks
    # But to compute log-probs on the first action token, we turn off
    # the last real state token (set to 0 in state_mask)
    state_lens = state_mask.sum(dim=1)
    last_idx = (state_lens - 1).clamp_min(0)
    rows = torch.arange(state_mask.shape[0], device=state_mask.device)
    state_mask[rows, last_idx] = 0
    action_mask = deepcopy(state_action_mask)
    action_mask[:, :state_mask.shape[1]] -= state_mask
    action_lens = action_mask.sum(dim=1)

    return {
        "model_input": state_action_inputs,
        "action_mask": action_mask,
        "action_lens": action_lens,
        "state_mask":  state_mask,
    }


def get_state_action_logprobs(
    model: AutoModelForCausalLM | PreTrainedModel,
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    batch_state_action_dicts: list[list[dict[str, Any]]],
    tokenizer_kwargs: dict[str, Any] | None = None,
    temperature: float = 1.0,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Get log-probs for actions in a batch of state-action dicts.

    Returns:
        batch_logps: list[torch.Tensor], log-probs for each action
        batch_action_lens: torch.Tensor, token lengths for each action
    """
    # Template and tokenize messages
    batch_sa_inputs = process_state_action_inputs(
        tokenizer=tokenizer,
        batch_state_action_dicts=batch_state_action_dicts,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    
    # Get logits for state-action tokens
    batch_logits = model(
        **batch_sa_inputs["model_input"].to(model.device),
        return_dict=True,
        use_cache=False,
    ).logits.cpu()
    batch_labels = batch_sa_inputs["model_input"]["input_ids"].cpu()
    batch_a_mask = batch_sa_inputs["action_mask"].bool()

    # Filter for action tokens
    _mask_logits, _mask_labels = zip(*[
        (batch_logits[i, act_mask], batch_labels[i, act_mask])
        for i, act_mask in enumerate(batch_a_mask)
    ])

    # Compute log-probs; shape is batch_size x (seq_len - 1,)
    batch_logps = [
        _gather_label_logprobs(
            shifted_logits=_mask_logits[_idx][:-1],
            shifted_labels=_mask_labels[_idx][1:],
            temperature=temperature,
        )
        for _idx in range(len(_mask_logits))
    ]
    return batch_logps, batch_sa_inputs["action_lens"]


def get_state_and_action_logprobs(
    model: AutoModelForCausalLM | PreTrainedModel,
    tokenizer: AutoTokenizer | PreTrainedTokenizer,
    batch_state_action_dicts: list[list[dict[str, Any]]],
    tokenizer_kwargs: dict[str, Any] | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get log-probs for state and action tokens from a batch of state-action dicts.

    Returns:
        batch_logps: torch.Tensor, log-probs for state and action tokens
        batch_action_lens: torch.Tensor, token lengths for each action
        batch_a_mask: torch.Tensor, mask for action tokens
    """
    # Template and tokenize messages
    batch_sa_inputs = process_state_action_inputs(
        tokenizer=tokenizer,
        batch_state_action_dicts=batch_state_action_dicts,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    
    # Get logits for state-action tokens
    batch_logits = model(
        **batch_sa_inputs["model_input"].to(model.device),
        return_dict=True,
        use_cache=False,
    ).logits.cpu()
    batch_labels = batch_sa_inputs["model_input"]["input_ids"].cpu()
    batch_a_mask = batch_sa_inputs["action_mask"].bool()

    # Compute log-probs; shape is batch_size x (seq_len - 1,)
    batch_logps = _gather_label_logprobs(
        shifted_logits=batch_logits[:, :-1, :],
        shifted_labels=batch_labels[:, 1:],
        temperature=temperature,
    )

    return batch_logps, batch_sa_inputs["action_lens"], batch_a_mask
