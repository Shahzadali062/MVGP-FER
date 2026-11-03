"""Evaluate a locally trained M1, M2, or M3 reference checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .build_model import make_model
from .dataset import PreparedFERDataset, read_manifest
from .engine import run_epoch


def main():
    parser = argparse.ArgumentParser(description="Evaluate a locally trained FER checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True, help="RAF-DB: test; FER2013: PrivateTest")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    variant = checkpoint["variant"]
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        parser.error("No manifest rows for the selected split")
    mean = checkpoint.get("geometry_mean")
    std = checkpoint.get("geometry_std")
    if variant == "M3" and (mean is None or std is None or checkpoint.get("prototypes") is None):
        parser.error("M3 checkpoint is missing prototypes or training statistics")
    dataset = PreparedFERDataset(
        args.manifest, rows,
        mean.numpy() if mean is not None else None,
        std.numpy() if std is not None else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = make_model(variant, checkpoint.get("prototypes")).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_epoch(model, loader, device)
    print(
        f"{variant} | {args.split} | accuracy={metrics['accuracy']:.4f} "
        f"loss={metrics['loss']:.4f} samples={metrics['samples']}"
    )
    print("Results apply to this local reference checkpoint, not the manuscript experiments.")


if __name__ == "__main__":
    main()
