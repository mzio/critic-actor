"""
Critic-Actor model
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_critic_actor(
    llm_encoder: Any | None = None,
    **kwargs: Any,
 ) -> nn.Module:
    """
    Load a Critic-Actor model
    """
    if llm_encoder is None:
        assert (
            kwargs.get("hidden_size", None) is not None
        ), "hidden_size must be provided if llm_encoder is not provided"
        hidden_size = kwargs.pop("hidden_size")
    else:
        hidden_size = llm_encoder.config.hidden_size

    return CriticActor(hidden_size=hidden_size, **kwargs)


class CriticActor(nn.Module):
    """
    Critic-Actor model
    """
    def __init__(
        self,
        q_head_type: str,
        k_head_type: str,
        cos_sim: bool = False,
        temperature: float = 0.1,
        **head_kwargs: Any,
    ) -> None:
        super().__init__()
        self.q_head_type = q_head_type
        self.k_head_type = k_head_type
        self.cos_sim = cos_sim
        self.temperature = temperature
        self.inv_temp = 1 / temperature  # multiply faster than divide?

        self.q_head = self.init_head(q_head_type, **head_kwargs)
        self.k_head = self.init_head(k_head_type, **head_kwargs)

    def init_head(self, head_type: str, **head_kwargs: Any) -> nn.Module:
        """
        Initialize the head modules.
        """
        if head_type == "attn_pool":
            return AttnPoolCriticHead(**head_kwargs)
        elif head_type == "last_token":
            return LastTokenCriticHead(**head_kwargs)
        else:
            raise NotImplementedError(f"Sorry head type {head_type} not implemented yet.")

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().to(device, dtype, **kwargs)
        self.device = device

    def forward(
        self,
        embed_q: torch.Tensor,
        embed_k: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
        - embed_q: (batch_size, seq_len, hidden_size)
        - embed_k: (batch_size, num_actions, seq_len, hidden_size)
        - mask_q: (batch_size, seq_len)
        - mask_k: (batch_size, num_actions, seq_len)

        Returns:
        - scores: (1, num_actions)
        """
        b, n, l, d = embed_k.shape
        embed_q = self.q_head(embed_q, mask_q).unsqueeze(1)  # (batch_size, 1, hidden_size)
        embed_k = self.k_head(embed_k.view(-1, l, d), mask_k.view(-1, l))  # (batch_size * num_actions, hidden_size)
        embed_k = embed_k.view(b, n, -1)
        if self.cos_sim:
            return torch.cosine_similarity(embed_q, embed_k, dim=-1) * self.inv_temp  # (batch_size, num_actions)
        else:
            return torch.einsum("b1d,bnd->bn", embed_q, embed_k) * self.inv_temp


class AttnPoolCriticHead(nn.Module):
    """
    Critic head that operates on attention pooled last hidden states
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
        scaled_init: bool = True,
        q_bias: bool = False,
    ) -> None:
        super().__init__()
        mlp_hidden = mlp_hidden or (4 * hidden_size)
        self.q_proj = nn.Linear(output_size, 1, bias=q_bias)
        self.k_proj = nn.Linear(hidden_size, output_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_size),
        )
        # Initialization
        if scaled_init:
            nn.init.normal_(self.q_proj.weight, std=output_size**-0.5)

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        embed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute critic projections
        - last_hidden_states: (batch_size, seq_len, hidden_size)
        - embed_mask: (batch_size, seq_len) with True on tokens to embed
        """
        # k = self.k_proj(last_hidden_states)
        # w = torch.einsum("d,bld->bl", self.q_proj, k)
        w = self.q_proj(self.k_proj(last_hidden_states)).squeeze(-1)  # same as above
        # print('self.q_proj(self.k_proj(last_hidden_states)).squeeze(-1).shape', w.shape)
        w = w.masked_fill(~embed_mask, -float("inf"))
        # print('w.masked_fill(~embed_mask, -float("inf")).shape', w.shape)
        w = F.softmax(w, dim=-1)
        # print('F.softmax(w, dim=-1).shape', w.shape)

        return self.mlp(
            torch.einsum("bl,bld->bd", w, last_hidden_states)
        )


class LastTokenCriticHead(nn.Module):
    """
    Critic head that operates on last token's last hidden state
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
        offset: int = 1,
    ) -> None:
        super().__init__()
        mlp_hidden = mlp_hidden or (4 * hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_size),
        )
        self.offset = offset

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        embed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute critic projections
        - last_hidden_states: (batch_size, seq_len, hidden_size)
        - embed_mask: (batch_size, seq_len) with True on tokens to embed

        Pass only with last_hidden_states[:, :-1, :] and embed_mask[:, :-1] ?
        """
        # index the last generated token per sequence

        batch_size, seq_len = last_hidden_states.shape[:2]
        # position of the rightmost True per row
        # convert to long indices; assume at least one True per row
        _device = last_hidden_states.device
        idx = torch.argmax(embed_mask.long() * torch.arange(seq_len, device=_device), dim=1)  # [b]
        tok = last_hidden_states[torch.arange(batch_size, device=_device), idx - self.offset]  # [b, d]
        return self.mlp(tok)
