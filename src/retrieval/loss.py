"""InfoNCE contrastive loss with in-batch and explicit hard negative mining."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE / NT-Xent contrastive loss.

    Supports:
    - Symmetric in-batch negatives
    - Explicitly mined hard negative tensors
    - Fixed or learnable temperature parameter tau
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ) -> None:
        super().__init__()
        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_tau)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        hard_negative_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute contrastive InfoNCE loss.

        Args:
            query_embeddings: (B, D) L2-normalized query representations.
            positive_embeddings: (B, D) L2-normalized positive item representations.
            hard_negative_embeddings: Optional (B, M, D) L2-normalized mined hard negatives.

        Returns:
            Scalar cross-entropy contrastive loss.
        """
        batch_size = query_embeddings.size(0)
        tau = self.temperature.clamp(min=1e-3, max=1.0)

        # In-batch cosine similarities: (B, B)
        in_batch_sims = torch.matmul(query_embeddings, positive_embeddings.T) / tau

        if hard_negative_embeddings is not None and hard_negative_embeddings.numel() > 0:
            # hard_negative_embeddings: (B, M, D)
            # query_embeddings.unsqueeze(1): (B, 1, D)
            # hard_sims: (B, M)
            hard_sims = (
                torch.sum(
                    query_embeddings.unsqueeze(1) * hard_negative_embeddings,
                    dim=-1,
                )
                / tau
            )
            # Concatenate along candidate dimension: (B, B + M)
            all_logits = torch.cat([in_batch_sims, hard_sims], dim=-1)
        else:
            all_logits = in_batch_sims

        # Ground truth positive is the diagonal index i for each query i
        targets = torch.arange(batch_size, device=query_embeddings.device, dtype=torch.long)
        return F.cross_entropy(all_logits, targets)
