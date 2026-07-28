#!/usr/bin/env python3
"""
Compare intermediate CMI binary dumps between reference (AIx) and test providers.
Usage:
    python compare_cmi_dumps.py --aix-dir <path> --test-dir <path> [--inferences 1,2,3]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

def compare_dumps(aix_dir: Path, test_dir: Path, inferences: list, rank: int = 0):
    print(f"=" * 80)
    print(f"CMI Intermediate Tensor Bitwise Verification (Rank {rank})")
    print(f"AIx directory : {aix_dir}")
    print(f"Test directory: {test_dir}")
    print(f"=" * 80)

    stages = [
        'assembled_input',
        'raw_provider_output',
        'reconstructed_fields_field0',
        'reconstructed_fields_field1',
        'reconstructed_fields_field2'
    ]

    all_match = True

    for inf_idx in inferences:
        print(f"\n--- Inference Cycle {inf_idx} ---")
        for stage in stages:
            aix_file = aix_dir / f"current_rank_{rank}_inference_{inf_idx}_{stage}.bin"
            test_file = test_dir / f"current_rank_{rank}_inference_{inf_idx}_{stage}.bin"

            if not aix_file.exists():
                print(f"  {stage:<30s}: AIx reference file missing ({aix_file.name})")
                all_match = False
                continue
            if not test_file.exists():
                print(f"  {stage:<30s}: Test provider file missing ({test_file.name})")
                all_match = False
                continue

            dtype = np.float32 if ('assembled' in stage or 'raw_provider' in stage) else np.float64
            a = np.fromfile(aix_file, dtype=dtype)
            t = np.fromfile(test_file, dtype=dtype)

            if a.shape != t.shape:
                print(f"  {stage:<30s}: Shape mismatch {a.shape} vs {t.shape}")
                all_match = False
                continue

            diff = np.abs(a.astype(np.float64) - t.astype(np.float64))
            same = np.array_equal(a, t)
            if not same:
                all_match = False

            status = "BITWISE IDENTICAL" if same else f"DIFF (max={diff.max():.4e}, nonzeros={np.count_nonzero(diff)})"
            print(f"  {stage:<30s}: {status}")

    print("\n" + "=" * 80)
    if all_match:
        print("RESULT: ALL CMI INTERMEDIATE TENSORS ARE BITWISE IDENTICAL (0.000e+00 DIFF)")
    else:
        print("RESULT: DIFFERENCES DETECTED")
    print("=" * 80)
    return all_match

def main():
    parser = argparse.ArgumentParser(description="Compare CMI intermediate binary dumps")
    parser.add_argument("--aix-dir", required=True, type=Path, help="Directory containing AIx CMI dumps")
    parser.add_argument("--test-dir", required=True, type=Path, help="Directory containing test provider CMI dumps")
    parser.add_argument("--inferences", default="1,2,3", help="Comma-separated list of inference indices (default: 1,2,3)")
    parser.add_argument("--rank", type=int, default=0, help="Rank index to compare (default: 0)")
    args = parser.parse_args()

    inf_list = [int(i.strip()) for i in args.inferences.split(',') if i.strip()]
    success = compare_dumps(args.aix_dir, args.test_dir, inf_list, args.rank)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
