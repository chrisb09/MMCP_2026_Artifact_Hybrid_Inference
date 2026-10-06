#!/usr/bin/env python3
"""Require CUDA execution evidence for both AIX smoke inference calls."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collective", "pipelined"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    pattern = "aix_rank_*.csv" if args.mode == "collective" else "aix_p2p_timeline_rank_*.csv"
    calls = set()
    for path in args.directory.glob(pattern):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["is_controller"] != "1":
                    continue
                if args.mode == "collective":
                    if int(row["device_batches"]) > 0 and float(row["forward_gpu_ms"]) > 0:
                        calls.add(int(row["call"]))
                elif row["event"] == "torch_forward_end" and int(row["sample_count"]) > 0:
                    calls.add(int(row["step"]))
    if len(calls) < 2:
        raise SystemExit(f"AIX_GPU_EVIDENCE_FAIL: {len(calls)}/2 CUDA inference calls in {args.directory}")
    print(f"AIX_GPU_EVIDENCE_PASS: {len(calls)} CUDA inference calls ({args.mode})")


if __name__ == "__main__":
    main()
