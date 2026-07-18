#!/usr/bin/env python3
"""Merge legacy OUTPUT_FIELDS rank files into the common snapshot format."""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


PATTERN = re.compile(r"(sent|received)_rank_(\d+)_t(\d+)\.h5")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    events = {}
    for path in args.input_dir.glob("*.h5"):
        match = PATTERN.fullmatch(path.name)
        if match:
            kind, _, step = match.groups()
            events.setdefault((int(step), kind), []).append(path)
    if not events:
        raise SystemExit("No legacy rank field files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        for index, ((step, kind), paths) in enumerate(sorted(events.items())):
            blocks = []
            shape = np.zeros(3, dtype=int)
            for path in paths:
                with h5py.File(path, "r") as handle:
                    flow = handle["flow"]
                    local_shape = np.asarray(flow["U"].shape, dtype=int)
                    offset = np.asarray(flow["offsets"], dtype=int)
                    shape = np.maximum(shape, offset + local_shape)
                    blocks.append((path, offset, local_shape))
            sums = {field: np.zeros(shape, dtype=np.float64) for field in "UVW"}
            owners = np.zeros(shape, dtype=np.uint16)
            for path, offset, local_shape in blocks:
                slices = tuple(slice(offset[axis], offset[axis] + local_shape[axis]) for axis in range(3))
                with h5py.File(path, "r") as handle:
                    for field in sums:
                        sums[field][slices] += handle["flow"][field][...]
                owners[slices] += 1
            group = output.create_group(f"step_{index:04d}")
            group.attrs["globalTimeStep"] = step
            group.attrs["type"] = kind
            group.attrs["rank_blocks"] = len(blocks)
            group.attrs["uncovered_cells"] = int(np.count_nonzero(owners == 0))
            group.attrs["max_owners"] = int(owners.max())
            group.create_dataset("ownership", data=owners, compression="gzip")
            for field, values in sums.items():
                group.create_dataset(field, data=np.divide(values, owners, out=np.full(shape, np.nan), where=owners > 0), compression="gzip")
            print(f"{step} {kind}: {len(blocks)} ranks, shape={tuple(shape)}, uncovered={np.count_nonzero(owners == 0)}")


if __name__ == "__main__":
    main()
