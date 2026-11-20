"""Contrastive retrieval training script with InfoNCE loss."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.retrieval.adapter import ContrastiveRetrievalModel
from src.retrieval.dataset import build_split_datasets
from src.retrieval.loss import InfoNCELoss
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_contrastive_retriever(
    epochs: int = 25,
    lr: float = 1e-4,
    temperature: float = 0.07,
    batch_size: int = 8,
    use_hard_negatives: bool = True,
    train_fraction: float = 1.0,
    seed: int = 42,
    output_checkpoint: str = "models/retrieval_adapter.pt",
) -> dict[str, float]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training contrastive retriever on device: %s", device)

    train_ds, val_ds, _ = build_split_datasets(
        use_hard_negatives=use_hard_negatives,
        random_state=seed,
    )

    if train_fraction < 1.0:
        n_sub = max(2, int(len(train_ds) * train_fraction))
        train_ds.samples = train_ds.samples[:n_sub]
        logger.info("Subsampled training set to %d pairs (%.0f%%)", n_sub, train_fraction * 100)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ContrastiveRetrievalModel(embed_dim=train_ds.embed_dim).to(device)
    loss_fn = InfoNCELoss(temperature=temperature).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            q = batch["query"].to(device)
            pos = batch["positive"].to(device)
            negs = (
                batch["hard_negatives"].to(device)
                if use_hard_negatives and batch["hard_negatives"].numel() > 0
                else None
            )

            q_proj = model.encode_query(q)
            pos_proj = model.encode_item(pos)

            neg_proj = None
            if negs is not None and negs.size(1) > 0:
                # Shape: (B, M, D)
                B, M, D = negs.shape
                negs_flat = negs.view(B * M, D)
                neg_proj_flat = model.encode_item(negs_flat)
                neg_proj = neg_proj_flat.view(B, M, D)

            loss = loss_fn(q_proj, pos_proj, neg_proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, n_train_batches)

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                q = batch["query"].to(device)
                pos = batch["positive"].to(device)
                negs = (
                    batch["hard_negatives"].to(device)
                    if use_hard_negatives and batch["hard_negatives"].numel() > 0
                    else None
                )

                q_proj = model.encode_query(q)
                pos_proj = model.encode_item(pos)

                neg_proj = None
                if negs is not None and negs.size(1) > 0:
                    B, M, D = negs.shape
                    negs_flat = negs.view(B * M, D)
                    neg_proj = model.encode_item(negs_flat).view(B, M, D)

                v_loss = loss_fn(q_proj, pos_proj, neg_proj)
                total_val_loss += v_loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(1, n_val_batches)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info("Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f", epoch, epochs, avg_train_loss, avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_checkpoint)

    logger.info("Training complete. Best Val Loss: %.4f saved to %s", best_val_loss, output_checkpoint)
    return {
        "final_train_loss": float(avg_train_loss),
        "best_val_loss": float(best_val_loss),
    }


def main():
    parser = argparse.ArgumentParser(description="Train contrastive retrieval adapter.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="models/retrieval_adapter.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_contrastive_retriever(
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        use_hard_negatives=not args.no_hard_negatives,
        train_fraction=args.fraction,
        output_checkpoint=args.checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
