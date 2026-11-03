"""Train the visual baseline on prepared .npy arrays.

Run from repository root: python -m training.train --help
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build_model import make_model
from .dataset import PreparedFERDataset, read_manifest
from .engine import run_epoch


def split_training(rows: list[dict], seed: int = 42, fraction: float = 0.1):
    """Reserve a stratified validation set from the training split."""
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {}
    for row in rows:
        by_class.setdefault(int(row["paper_label"]), []).append(row)
    train, valid = [], []
    for _, items in sorted(by_class.items()):
        rng.shuffle(items)
        take = max(1, round(len(items) * fraction)) if len(items) > 1 else 0
        valid.extend(items[:take])
        train.extend(items[take:])
    if not train or not valid:
        raise ValueError("Need enough training samples for an internal validation split")
    return train, valid


def main():
    parser = argparse.ArgumentParser(description="Train the visual-only M1 baseline")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("M1",), default="M1")
    parser.add_argument("--dataset", choices=("rafdb", "fer2013"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint outside public repository")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1:
        parser.error("epochs >= 1, batch-size >= 2, and patience >= 1 are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    rows = read_manifest(args.manifest)
    if args.dataset == "rafdb":
        available = [row for row in rows if row["split"].lower() == "train"]
        train_rows, val_rows = split_training(available, args.seed)
        print("RAF-DB: reserving 10% of training images for validation; test images are untouched")
    else:
        train_rows = [row for row in rows if row["split"] == "Training"]
        val_rows = [row for row in rows if row["split"] == "PublicTest"]
    if not train_rows or not val_rows:
        parser.error("Training/validation partition is empty; check source dataset split names")

    train_data = PreparedFERDataset(args.manifest, train_rows)
    val_data = PreparedFERDataset(args.manifest, val_rows)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = make_model(args.variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None
    args.out = args.out.expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    if args.out.is_relative_to(repo_root):
        parser.error("Save checkpoints outside the public Git repository")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs_without_gain = 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler, amp=amp)
        val = run_epoch(model, val_loader, device, amp=amp)
        print(f"{epoch:03d} train_loss={train['loss']:.4f} train_acc={train['accuracy']:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}")
        if val["loss"] < best:
            best = val["loss"]
            epochs_without_gain = 0
            torch.save({
                "variant": args.variant,
                "model_state": model.state_dict(),
                "dataset": args.dataset,
                "checkpoint_selection": "validation cross-entropy",
                "seed": args.seed,
                "source": "visual baseline implementation",
            }, args.out)
        else:
            epochs_without_gain += 1
        if epochs_without_gain >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best validation loss={best:.4f}; checkpoint={args.out}")


if __name__ == "__main__":
    main()
