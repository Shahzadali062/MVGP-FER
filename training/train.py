From the repository root: python -m training.train --help
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .build_model import load_prototypes, make_model
from .dataset import PreparedFERDataset, compute_train_statistics, read_manifest
from .engine import run_epoch


def split_training(rows: list[dict], seed: int = 42, fraction: float = 0.1):
    """Hold out 10% of RAF-DB training images, stratified by class."""
    rng = random.Random(seed)
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["paper_label"]), []).append(row)
    train, valid = [], []
    for group in grouped.values():
        rng.shuffle(group)
        count = max(1, round(len(group) * fraction)) if len(group) > 1 else 0
        valid.extend(group[:count])
        train.extend(group[count:])
    if not train or not valid:
        raise ValueError("Not enough training samples for the RAF-DB validation split")
    return train, valid


def main():
    parser = argparse.ArgumentParser(description="Train MVGP-FER reference models")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("M1", "M2", "M3"), default="M1")
    parser.add_argument("--dataset", choices=("rafdb", "fer2013"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint path outside Git repository")
    parser.add_argument("--prototypes", type=Path, help="M3 only: original 7 x 16 prototype matrix as JSON")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.lr <= 0:
        parser.error("epochs >= 1, batch-size >= 2, patience >= 1, and lr > 0 are required")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.variant == "M3" and args.prototypes is None:
        parser.error("M3 requires --prototypes with the verified 7 x 16 values")
    if args.variant != "M3" and args.prototypes is not None:
        parser.error("--prototypes is only used with M3")

    repo_root = Path(__file__).resolve().parent.parent
    output = args.out.expanduser().resolve()
    if output.is_relative_to(repo_root):
        parser.error("Store checkpoints outside the public Git repository")
    if args.prototypes is not None and args.prototypes.expanduser().resolve().is_relative_to(repo_root):
        parser.error("Keep unpublished prototype values outside this Git repository")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)

    rows = read_manifest(args.manifest)
    if args.dataset == "rafdb":
        train_rows, val_rows = split_training(
            [r for r in rows if r["split"] == "train"], args.seed
        )
        print("RAF-DB: stratified training holdout used for validation (10%)")
    else:
        train_rows = [r for r in rows if r["split"] == "Training"]
        val_rows = [r for r in rows if r["split"] == "PublicTest"]
        print("FER2013: PublicTest used for validation")
    if not train_rows or not val_rows:
        parser.error("Empty training or validation split; check manifest labels")

    mean = std = prototypes = None
    if args.variant == "M3":
        prototypes = load_prototypes(args.prototypes)
        mean, std = compute_train_statistics(args.manifest, train_rows)
        count = sum(
            bool(r.get("geometry_104_npy", "").strip()) or
            (bool(r.get("landmarks_88_npy", "").strip()) and
             bool(r.get("geometric_cues_16_npy", "").strip()))
            for r in train_rows if r.get("landmark_status", "valid").lower() == "valid"
        )
        print(f"M3: {count} training rows provide complete geometric features")

    if len(train_rows) < 2:
        parser.error("At least two training samples are required for batch normalization")
    train = PreparedFERDataset(args.manifest, train_rows, mean, std)
    valid = PreparedFERDataset(args.manifest, val_rows, mean, std)
    # Landmark batch normalization cannot train on a final batch of one.
    drop_single = (args.variant == "M3" and len(train) > args.batch_size
                   and len(train) % args.batch_size == 1)
    training_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=drop_single,
    )
    validation_loader = DataLoader(
        valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(args.variant, prototypes=prototypes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, training_loader, device, optimizer=optimizer, scaler=scaler, amp=use_amp
        )
        val_metrics = run_epoch(model, validation_loader, device, amp=use_amp)
        print(
            f"{epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            stale_epochs = 0
            torch.save({
                "variant": args.variant,
                "dataset": args.dataset,
                "model_state": model.state_dict(),
                "geometry_mean": torch.from_numpy(mean.copy()) if mean is not None else None,
                "geometry_std": torch.from_numpy(std.copy()) if std is not None else None,
                "prototypes": prototypes.cpu().clone() if prototypes is not None else None,
                "checkpoint_selection": "minimum validation cross-entropy",
                "seed": args.seed,
                "implementation": "reference, not verified against original experiment",
            }, output)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print(f"Best validation loss: {best:.4f}; checkpoint: {output}")


if __name__ == "__main__":
    main()
