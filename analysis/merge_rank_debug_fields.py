#!/usr/bin/env python3
"""Merge rank-qualified CMI reconstructed-field debug buffers."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def manifest_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--implementation", choices=("current", "legacy"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifests = sorted(args.input_dir.glob(f"{args.implementation}_rank_*_inference_1_manifest.txt"))
    if not manifests:
        raise SystemExit("No rank manifests found")
    blocks = []
    global_shape = np.zeros(3, dtype=int)
    for manifest in manifests:
        data = manifest_values(manifest)
        rank = int(data["rank"])
        shape = np.fromstring(data["n_cells"], sep=",", dtype=int)
        offset = np.fromstring(data["offsets"], sep=",", dtype=int)
        global_shape = np.maximum(global_shape, offset + shape)
        blocks.append((rank, manifest, shape, offset))

    sums = {field: np.zeros(global_shape, dtype=np.float64) for field in range(3)}
    owners = np.zeros(global_shape, dtype=np.uint16)
    dtype = np.float32 if args.implementation == "current" else np.float64
    for rank, manifest, shape, offset in blocks:
        slices = tuple(slice(offset[axis], offset[axis] + shape[axis]) for axis in range(3))
        for field in range(3):
            if args.implementation == "current":
                name = f"current_rank_{rank}_inference_1_reconstructed_fields_field{field}.bin"
            else:
                name = f"legacy_rank_{rank}_inference_1_reconstructed_field{field}.bin"
            values = np.fromfile(args.input_dir / name, dtype=dtype).reshape(shape)
            sums[field][slices] += values
        owners[slices] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        output.attrs["implementation"] = args.implementation
        output.attrs["rank_blocks"] = len(blocks)
        output.create_dataset("ownership", data=owners, compression="gzip")
        for field, values in sums.items():
            output.create_dataset("UVW"[field], data=np.divide(values, owners, out=np.full(global_shape, np.nan), where=owners > 0), compression="gzip")
    print(f"Merged {len(blocks)} rank buffers: shape={tuple(global_shape)}, uncovered={np.count_nonzero(owners == 0)}, max_owners={owners.max()}")


if __name__ == "__main__":
    main()
