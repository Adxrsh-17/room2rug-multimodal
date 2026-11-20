"""Unit tests for InfoNCE contrastive loss."""

import torch

from src.retrieval.loss import InfoNCELoss


def test_infonce_loss_in_batch():
    loss_fn = InfoNCELoss(temperature=0.07)
    q = torch.nn.functional.normalize(torch.randn(4, 128), p=2, dim=-1)
    pos = torch.nn.functional.normalize(torch.randn(4, 128), p=2, dim=-1)

    loss = loss_fn(q, pos)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0


def test_infonce_loss_with_hard_negatives():
    loss_fn = InfoNCELoss(temperature=0.07)
    B, M, D = 4, 3, 128
    q = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)
    pos = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=-1)
    hard_negs = torch.nn.functional.normalize(torch.randn(B, M, D), p=2, dim=-1)

    loss_with_hard = loss_fn(q, pos, hard_negs)
    loss_without_hard = loss_fn(q, pos)

    assert loss_with_hard.item() > 0.0
    # Hard negatives increase denominator, so loss must be strictly greater or equal
    assert loss_with_hard.item() >= loss_without_hard.item() - 1e-4
