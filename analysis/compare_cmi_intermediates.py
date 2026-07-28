#!/usr/bin/env python3
"""Compare legacy vs current CMI intermediate dumps, aligned by inference index.

Usage
-----
python compare_cmi_intermediates.py \
    --legacy-dir  <path/to/debug/cmi>  \
    --current-dir <path/to/debug/cmi>  \
    [--inferences 1]                   \
    [--rank 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

def read_manifest(path: Path) -> dict:
    m = {}
    if not path.exists():
        return m
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            m[k.strip()] = v.strip()
    return m

def load_bin(path: Path, dtype: np.dtype, count: int) -> np.ndarray:
    expected_bytes = count * np.dtype(dtype).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path.name}: expected {expected_bytes} bytes ({count} × {np.dtype(dtype).itemsize}), "
            f"got {actual_bytes} bytes"
        )
    return np.fromfile(path, dtype=dtype)

def stats(label: str, legacy: np.ndarray, current: np.ndarray) -> str:
    a = legacy.astype(np.float64)
    b = current.astype(np.float64)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(b), 1e-30)
    rel   = diff / scale
    
    # Calculate float32 ULP error for reconstructed fields
    eps = np.finfo(np.float32).eps
    ulp_diff = diff / (scale * eps + 1e-30)
    
    return (
        f"  {label:<25s}  "
        f"maxabs={diff.max():.3e}  meanabs={diff.mean():.3e}  "
        f"rms={np.sqrt((diff**2).mean()):.3e}  "
        f"maxrel={rel.max():.3e}  "
        f"max_ulp={ulp_diff.max():.2f}"
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-dir",  required=True, type=Path)
    ap.add_argument("--current-dir", required=True, type=Path)
    ap.add_argument("--inferences",  type=int, default=1)
    ap.add_argument("--rank",        type=int, default=0)
    args = ap.parse_args()

    any_error = False

    for idx in range(1, args.inferences + 1):
        rank = args.rank
        print(f"\n{'='*70}")
        print(f"Inference index {idx}  (rank {rank})")
        print(f"{'='*70}")

        lm_path = args.legacy_dir  / f"legacy_rank_{rank}_inference_{idx}_manifest.txt"
        cm_path = args.current_dir / f"current_rank_{rank}_inference_{idx}_manifest.txt"

        lm = read_manifest(lm_path)
        cm = read_manifest(cm_path)

        # Basic verification
        num_cubes = int(lm.get("num_cubes", 1152))
        cube_size = int(lm.get("cube_size", 512))
        n_fields = int(lm.get("n_fields", 3))
        input_seq_len = int(lm.get("input_sequence_length", 5))
        forecast_window = int(lm.get("forecast_window", 2))

        ai_count = n_fields * num_cubes * input_seq_len * cube_size
        rpo_count = n_fields * num_cubes * forecast_window * cube_size
        rec_count = int(lm.get("reconstructed_field_count", 582498))

        # ── assembled_input ───────────────────────────────────────────
        try:
            la = load_bin(args.legacy_dir  / f"legacy_rank_{rank}_inference_{idx}_assembled_input.bin", np.float32, ai_count)
            ca = load_bin(args.current_dir / f"current_rank_{rank}_inference_{idx}_assembled_input.bin", np.float32, ai_count)
            print(stats("assembled_input", la, ca))
        except Exception as e:
            print(f"  assembled_input ERROR: {e}")
            any_error = True

        # ── raw_provider_output ───────────────────────────────────────
        try:
            lo = load_bin(args.legacy_dir  / f"legacy_rank_{rank}_inference_{idx}_raw_provider_output.bin", np.float32, rpo_count)
            co = load_bin(args.current_dir / f"current_rank_{rank}_inference_{idx}_raw_provider_output.bin", np.float32, rpo_count)
            print(stats("raw_provider_output", lo, co))
        except Exception as e:
            print(f"  raw_provider_output ERROR: {e}")
            any_error = True

        # ── reconstructed fields ──────────────────────────────────────
        for field in range(n_fields):
            try:
                lf_path = args.legacy_dir  / f"legacy_rank_{rank}_inference_{idx}_reconstructed_field{field}.bin"
                cf_path = args.current_dir / f"current_rank_{rank}_inference_{idx}_reconstructed_fields_field{field}.bin"
                lf = load_bin(lf_path, np.float64, rec_count)
                cf = load_bin(cf_path, np.float32, rec_count)
                print(stats(f"reconstructed_field{field}", lf, cf))
            except Exception as e:
                print(f"  reconstructed_field{field} ERROR: {e}")
                any_error = True

    print()
    if any_error:
        sys.exit(1)
    else:
        print("All comparisons completed successfully.")

if __name__ == "__main__":
    main()
