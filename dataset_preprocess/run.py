"""Command-line entry point for image, FER2013, and RAF-DB preprocessing."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .export import check_output_path, prepare_dataset
from .images import preprocess_image


def main():
    parser = argparse.ArgumentParser(description="Prepare MVGP-FER input data")
    commands = parser.add_subparsers(dest="command", required=True)

    image = commands.add_parser("image", help="Prepare one image")
    image.add_argument("--input", type=Path, required=True)
    image.add_argument("--out", type=Path, required=True)

    fer = commands.add_parser("fer2013", help="Read original FER2013 CSV")
    fer.add_argument("--csv", type=Path, required=True)
    raf = commands.add_parser("rafdb", help="Read aligned RAF-DB images")
    raf.add_argument("--root", type=Path, required=True)

    for command in (fer, raf):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--landmarker-model", type=Path)
        command.add_argument("--landmark-indices", type=Path)

    args = parser.parse_args()
    if args.command == "image":
        target = check_output_path(args.out)
        if target.suffix.lower() != ".npy":
            parser.error("Image output must end in .npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(args.input) as source:
            pixels, _ = preprocess_image(source)
        np.save(target, pixels, allow_pickle=False)
        print(f"Saved {target} ({pixels.shape})")
        return

    source = args.csv if args.command == "fer2013" else args.root
    count = prepare_dataset(args.command, source, args.out,
                            args.landmarker_model, args.landmark_indices)
    print(f"Processed {count} records -> {args.out.expanduser().resolve()}")
    if not args.landmarker_model:
        print("Landmark extraction was skipped (model and landmark indices not provided).")


if __name__ == "__main__":
    main()
