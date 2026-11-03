"""Evaluate a locally trained checkpoint on a named dataset partition."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .build_model import make_model
from .dataset import PreparedFERDataset, read_manifest
from .engine import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True, help="RAF-DB: test; FER2013: PrivateTest")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        parser.error("No rows found for the requested split")
    dataset = PreparedFERDataset(args.manifest, rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = make_model(saved["variant"]).to(device)
    model.load_state_dict(saved["model_state"])
    scores = run_epoch(model, loader, device)
    print(f"Reference model {saved['variant']} | {args.split}: "
          f"accuracy={scores['accuracy']:.4f}, loss={scores['loss']:.4f}, "
          f"samples={scores['samples']}")
    print("These scores describe this checkpoint, not the manuscript experiments.")


if __name__ == "__main__":
    main()
