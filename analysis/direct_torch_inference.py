#!/usr/bin/env python3
"""Run the scripted model directly on one captured CMI input tensor."""

import argparse
from pathlib import Path

import numpy as np
import torch


INPUT_SHAPE = (3456, 5, 512)
OUTPUT_SHAPE = (3456, 2, 512)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.batch_size <= 0 or INPUT_SHAPE[0] % args.batch_size:
        raise SystemExit("batch size must be a positive divisor of 3456")

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    values = np.fromfile(args.input, dtype=np.float32)
    if values.size != int(np.prod(INPUT_SHAPE)):
        raise SystemExit(f"input has {values.size} values, expected {np.prod(INPUT_SHAPE)}")

    model = torch.jit.load(str(args.model), map_location="cpu")
    model.eval()
    tensor = torch.from_numpy(values.reshape(INPUT_SHAPE))

    outputs = []
    with torch.no_grad():
        for start in range(0, INPUT_SHAPE[0], args.batch_size):
            outputs.append(model(tensor[start : start + args.batch_size]))
    result = torch.cat(outputs, dim=0).contiguous().cpu().numpy()

    if result.shape != OUTPUT_SHAPE:
        raise SystemExit(f"model returned {result.shape}, expected {OUTPUT_SHAPE}")
    result.astype(np.float32, copy=False).tofile(args.output)
    print(
        f"torch={torch.__version__} threads={torch.get_num_threads()} "
        f"batch={args.batch_size} output={args.output} "
        f"min={result.min():.9e} max={result.max():.9e}"
    )


if __name__ == "__main__":
    main()
