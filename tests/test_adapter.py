"""Unit tests for residual projection adapter and contrastive model."""

import torch

from src.retrieval.adapter import ContrastiveRetrievalModel, ResidualProjectionHead


def test_residual_projection_head_shapes_and_normalization():
    head = ResidualProjectionHead(in_dim=512, hidden_dim=256, out_dim=512, dropout=0.0)
    x = torch.randn(4, 512)
    out = head(x)

    assert out.shape == (4, 512)
    # Assert output is unit L2-normalized
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_contrastive_retrieval_model_forward():
    model = ContrastiveRetrievalModel(embed_dim=512, hidden_dim=256, share_weights=False)
    q = torch.randn(3, 512)
    items = torch.randn(3, 512)

    sims = model(q, items)
    assert sims.shape == (3,)
    # Cosine similarities must be bounded between [-1.0, 1.0]
    assert (sims >= -1.0 - 1e-5).all() and (sims <= 1.0 + 1e-5).all()
