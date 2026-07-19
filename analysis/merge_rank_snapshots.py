#!/usr/bin/env python3
"""Merge opt-in per-rank MAIA snapshots into decomposition-independent fields."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def snapshot_index(paths):
    index = {}
    for path in paths:
        with h5py.File(path, "r") as handle:
            for name, group in handle.items():
                step = group.attrs.get("globalTimeStep")
                kind = group.attrs.get("type")
                if isinstance(kind, bytes):
                    kind = kind.decode()
                if step is not None and kind is not None:
                    index.setdefault((int(step), str(kind)), []).append((path, name))
    return index


def merge_event(entries):
    blocks = []
    global_shape = np.zeros(3, dtype=int)
    for path, group_name in entries:
        with h5py.File(path, "r") as handle:
            group = handle[group_name]
            shape = np.asarray(group["U"].shape, dtype=int)
            offset = np.asarray(group["nOffsetCells"], dtype=int)
            global_shape = np.maximum(global_shape, offset + shape)
            blocks.append((path, group_name, offset, shape))

    sums = {field: np.zeros(global_shape, dtype=np.float64) for field in ("U", "V", "W")}
    counts = np.zeros(global_shape, dtype=np.uint16)
    for path, group_name, offset, shape in blocks:
        slices = tuple(slice(offset[axis], offset[axis] + shape[axis]) for axis in range(3))
        with h5py.File(path, "r") as handle:
            group = handle[group_name]
            for field in sums:
                sums[field][slices] += group[field][...]
        counts[slices] += 1

    merged = {field: np.divide(values, counts, out=np.full(global_shape, np.nan), where=counts > 0)
              for field, values in sums.items()}
    return merged, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-id", help="Restrict input to snapshots from one Slurm job")
    args = parser.parse_args()

    pattern = f"snapshots_{args.job_id}_rank_*.h5" if args.job_id else "snapshots_*_rank_*.h5"
    paths = sorted(args.input_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No per-rank snapshots found in {args.input_dir}")
    events = snapshot_index(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        output.attrs["source_rank_files"] = len(paths)
        for index, ((step, kind), entries) in enumerate(sorted(events.items())):
            fields, counts = merge_event(entries)
            group = output.create_group(f"step_{index:04d}")
            group.attrs["globalTimeStep"] = step
            group.attrs["type"] = kind
            group.attrs["rank_blocks"] = len(entries)
            group.attrs["uncovered_cells"] = int(np.count_nonzero(counts == 0))
            group.attrs["max_owners"] = int(counts.max())
            group.create_dataset("ownership", data=counts, compression="gzip")
            for field, values in fields.items():
                group.create_dataset(field, data=values, compression="gzip")
            print(f"{step} {kind}: {len(entries)} ranks, shape={counts.shape}, "
                  f"uncovered={np.count_nonzero(counts == 0)}, max_owners={counts.max()}")


if __name__ == "__main__":
    main()
