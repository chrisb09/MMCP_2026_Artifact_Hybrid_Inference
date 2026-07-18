#!/usr/bin/env python3
"""Plot global per-step maximum field differences from a legacy snapshot set."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def index(handle):
    result = {}
    for name, group in handle.items():
        step = group.attrs.get("globalTimeStep")
        kind = group.attrs.get("type")
        if isinstance(kind, bytes):
            kind = kind.decode()
        if step is not None and kind is not None:
            result[(int(step), str(kind))] = name
    return result


def parse_provider(value):
    name, path = value.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--provider", action="append", type=parse_provider, required=True,
                        help="NAME=merged_snapshot.h5; repeat for each provider")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.legacy, "r") as legacy:
        legacy_index = index(legacy)
        records = []
        for provider, path in args.provider:
            with h5py.File(path, "r") as candidate:
                candidate_index = index(candidate)
                for key in sorted(set(legacy_index) & set(candidate_index)):
                    legacy_group = legacy[legacy_index[key]]
                    candidate_group = candidate[candidate_index[key]]
                    for field in "UVW":
                        diff = np.abs(legacy_group[field][...] - candidate_group[field][...])
                        records.append((provider, key[0], key[1], field, float(np.nanmax(diff))))

    with args.csv.open("w") as output:
        output.write("provider,step,type,field,max_abs_diff\n")
        for row in records:
            output.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.17e}\n")

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, field in zip(axes, "UVW"):
        for provider in sorted({record[0] for record in records}):
            points = sorted((step + (0.1 if kind == "received" else 0.0), value)
                            for name, step, kind, item_field, value in records
                            if name == provider and item_field == field)
            if points:
                axis.plot(*zip(*points), marker="o", label=provider)
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.set_ylabel(f"{field} max |diff|")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("global step (received points offset by +0.1)")
    figure.suptitle("Global maximum absolute difference from legacy AIX")
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    print(f"Wrote {args.csv} and {args.output}")


if __name__ == "__main__":
    main()
