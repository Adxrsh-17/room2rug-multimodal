"""Lightweight residual projection head for contrastive retrieval fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualProjectionHead(nn.Module):
    """Residual projection adapter for vision and multimodal representations.

    Transforms frozen backbone embeddings into an aligned room-to-rug metric space:
        y = L2Norm( x + W_proj( GELU( LN(x) W_1 + b_1 ) W_2 + b_2 ) )
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Residual shortcut projection if dimensions differ
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual skip and L2 normalization."""
        residual = self.shortcut(x)
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        out = residual + h
        return F.normalize(out, p=2, dim=-1)


class ContrastiveRetrievalModel(nn.Module):
    """Dual-tower / asymmetric contrastive model for room-to-rug retrieval."""

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        self.query_adapter = ResidualProjectionHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            dropout=dropout,
        )
        if share_weights:
            self.item_adapter = self.query_adapter
        else:
            self.item_adapter = ResidualProjectionHead(
                in_dim=embed_dim,
                hidden_dim=hidden_dim,
                out_dim=embed_dim,
                dropout=dropout,
            )

    def encode_query(self, query_tensor: torch.Tensor) -> torch.Tensor:
        return self.query_adapter(query_tensor)

    def encode_item(self, item_tensor: torch.Tensor) -> torch.Tensor:
        return self.item_adapter(item_tensor)

    def forward(
        self,
        query_tensor: torch.Tensor,
        item_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between queries and items."""
        q_proj = self.encode_query(query_tensor)
        i_proj = self.encode_item(item_tensor)
        return torch.sum(q_proj * i_proj, dim=-1)
