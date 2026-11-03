#!/usr/bin/env python3
"""
Correctness checker: replay PhyDLL cap-50000 debug-export pilot against the same
TorchScript model at two caps (default: 2048 and 50000) and compare outputs.

Reads exported .bin files only -- no Slurm, no GPU, no SSDB required.

Usage:
    python3 scripts/check_phydll_cap_correctness.py \\
        --pilot-run-dir scratch/phydll_cap50000_pilots/pilot_XXX/debug_dumps \\
        --model input/transformer_inference_scripted_fw2.pt
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_ranks(total, limit):
    """Return all ranks, or at most *limit* evenly spaced ranks including the ends."""
    if limit is None or limit >= total:
        return list(range(total))
    if limit < 1:
        raise ValueError("max-ranks must be positive")
    if limit == 1:
        return [0]
    return sorted({int(round(i * (total - 1) / (limit - 1))) for i in range(limit)})


def parse_manifest(path: Path) -> dict:
    """Parse key=value pairs from a manifest text file."""
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def check_finite(arr: np.ndarray, label: str) -> bool:
    """Return True if all finite, print diagnostics and return False otherwise."""
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        print(f"  [FAIL] {label}: {nan_count} NaN(s), {inf_count} Inf(s)")
        return False
    return True


def run_model_chunked(model: torch.jit.ScriptModule, inp: np.ndarray, cap: int) -> np.ndarray:
    """
    Run model forward in sub-batches of size `cap`, concatenate results.

    inp shape: [N, 5, 512]  (N = 3 * num_cubes)
    Returns:   [N, 2, 512]
    """
    N = inp.shape[0]
    outputs = []
    for start in range(0, N, cap):
        chunk = inp[start : start + cap]           # [chunk_size, 5, 512]
        tensor = torch.from_numpy(chunk)           # float32, CPU
        with torch.no_grad():
            out = model(tensor)                    # [chunk_size, 2, 512]
        outputs.append(out.numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 2, 512), dtype=np.float32)


def compare_outputs(
    out_a: np.ndarray,
    out_b: np.ndarray,
    atol: float,
    rtol: float,
    label: str,
) -> tuple[bool, dict]:
    """Compare two output arrays; return (pass, stats)."""
    if out_a.shape != out_b.shape:
        return False, {"error": f"shape mismatch {out_a.shape} vs {out_b.shape}"}
    diff = np.abs(out_a.astype(np.float64) - out_b.astype(np.float64))
    abs_ref = np.abs(out_b.astype(np.float64))
    max_abs_err = float(diff.max()) if diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    denom = np.where(abs_ref > 0, abs_ref, 1.0)
    rel_err = float((diff / denom).max()) if diff.size else 0.0
    passed = bool(np.allclose(out_a, out_b, atol=atol, rtol=rtol, equal_nan=False))
    return passed, {
        "label": label,
        "max_abs_err": max_abs_err,
        "rmse": rmse,
        "rel_err": rel_err,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pilot-run-dir", required=True, type=Path,
                        help="Directory containing rank/inference debug export files")
    parser.add_argument("--model", default="./input/transformer_inference_scripted_fw2.pt",
                        type=Path, help="Path to TorchScript model (.pt)")
    parser.add_argument("--ranks", type=int, default=24,
                        help="Number of solver ranks to check (default: 24)")
    parser.add_argument("--max-ranks", type=int, default=None,
                        help="Check at most this many evenly spaced ranks (default: all)")
    parser.add_argument("--rank-list", default=None,
                        help="Comma-separated rank indices; overrides --ranks/--max-ranks")
    parser.add_argument("--inferences", type=int, default=2,
                        help="Number of inferences per rank to check (default: 2)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for np.allclose (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for np.allclose (default: 1e-4)")
    parser.add_argument("--pilot-cap", type=int, default=50000,
                        help="Cap used during the pilot run (default: 50000)")
    args = parser.parse_args()

    cap_a = 2048
    cap_b = args.pilot_cap

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    model_sha256 = sha256_file(model_path)
    if args.rank_list is not None:
        try:
            rank_indices = sorted({int(item) for item in args.rank_list.split(",") if item.strip()})
        except ValueError:
            print(f"ERROR: invalid --rank-list: {args.rank_list}", file=sys.stderr)
            return 1
        if not rank_indices or rank_indices[0] < 0 or rank_indices[-1] >= args.ranks:
            print(f"ERROR: --rank-list out of range 0..{args.ranks - 1}", file=sys.stderr)
            return 1
    else:
        rank_indices = select_ranks(args.ranks, args.max_ranks)
    print("=" * 72)
    print("PhyDLL cap correctness checker")
    print("=" * 72)
    print(f"model_sha256    : {model_sha256}")
    print(f"model_path      : {model_path}")
    print(f"pilot_run_dir   : {args.pilot_run_dir.resolve()}")
    print(f"num_ranks       : {args.ranks}")
    print(f"selected_ranks  : {rank_indices}")
    print(f"num_inferences  : {args.inferences}")
    print(f"cap_a           : {cap_a}")
    print(f"cap_b (pilot)   : {cap_b}")
    print(f"declared_atol   : {args.atol}")
    print(f"declared_rtol   : {args.rtol}")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading model ...", end=" ", flush=True)
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
    except Exception as exc:
        print(f"FAILED\nERROR loading model: {exc}", file=sys.stderr)
        return 1
    print("OK")

    # -----------------------------------------------------------------------
    # Per-rank / per-inference loop
    # -----------------------------------------------------------------------
    run_dir = args.pilot_run_dir.resolve()
    # Exports live in a debug_dumps/ subdirectory; accept either the run root
    # or the debug_dumps directory itself so both invocation styles work.
    if (run_dir / "debug_dumps").is_dir():
        run_dir = run_dir / "debug_dumps"
    if not run_dir.is_dir():
        print(f"ERROR: pilot-run-dir not a directory: {run_dir}", file=sys.stderr)
        return 1

    total = 0
    failures = 0
    stats_table = []

    for rank in rank_indices:
        for inf_offset in range(args.inferences):
            inf_n = inf_offset + 1  # debug exports are numbered from 1
            total += 1
            label = f"rank={rank:02d} inf={inf_n}"

            # ----------------------------------------------------------------
            # Locate and parse manifest
            # ----------------------------------------------------------------
            manifest_path = run_dir / f"current_rank_{rank}_inference_{inf_n}_manifest.txt"
            if not manifest_path.is_file():
                print(f"  [{label}] SKIP manifest missing: {manifest_path}")
                failures += 1
                continue

            manifest = parse_manifest(manifest_path)
            try:
                num_cubes = int(manifest["num_cubes"])
            except (KeyError, ValueError) as exc:
                print(f"  [{label}] FAIL cannot read num_cubes from manifest: {exc}")
                failures += 1
                continue

            n_samples = 3 * num_cubes  # 3 input frames per cube

            # ----------------------------------------------------------------
            # Read normalized_input.bin
            # ----------------------------------------------------------------
            inp_path = run_dir / f"current_rank_{rank}_inference_{inf_n}_normalized_input.bin"
            if not inp_path.is_file():
                # fall back: some exports use a flat name in the same dir
                inp_path = run_dir / "normalized_input.bin"
            if not inp_path.is_file():
                print(f"  [{label}] FAIL normalized_input.bin not found")
                failures += 1
                continue
            try:
                inp_raw = np.frombuffer(inp_path.read_bytes(), dtype=np.float32)
                inp = inp_raw.reshape(n_samples, 5, 512)
            except Exception as exc:
                print(f"  [{label}] FAIL reading normalized_input.bin: {exc}")
                failures += 1
                continue

            # ----------------------------------------------------------------
            # Read raw_provider_output.bin (reference from pilot)
            # ----------------------------------------------------------------
            ref_path = run_dir / f"current_rank_{rank}_inference_{inf_n}_raw_provider_output.bin"
            if not ref_path.is_file():
                ref_path = run_dir / "raw_provider_output.bin"
            ref_output: np.ndarray | None = None
            if ref_path.is_file():
                try:
                    ref_raw = np.frombuffer(ref_path.read_bytes(), dtype=np.float32)
                    ref_output = ref_raw.reshape(n_samples, 2, 512)
                except Exception as exc:
                    print(f"  [{label}] WARNING could not read reference output: {exc}")

            # ----------------------------------------------------------------
            # Finite checks on inputs
            # ----------------------------------------------------------------
            ok = True
            if not check_finite(inp, f"{label} normalized_input"):
                ok = False

            # ----------------------------------------------------------------
            # Forward pass at cap_a and cap_b
            # ----------------------------------------------------------------
            try:
                out_a = run_model_chunked(model, inp, cap_a)
                out_b = run_model_chunked(model, inp, cap_b)
            except Exception as exc:
                print(f"  [{label}] FAIL forward error: {exc}")
                failures += 1
                continue

            if not check_finite(out_a, f"{label} output@cap_a={cap_a}"):
                ok = False
            if not check_finite(out_b, f"{label} output@cap_b={cap_b}"):
                ok = False

            # ----------------------------------------------------------------
            # Compare cap_a vs cap_b
            # ----------------------------------------------------------------
            passed_ab, s_ab = compare_outputs(out_a, out_b, args.atol, args.rtol,
                                              f"{label} cap_a vs cap_b")
            stats_table.append(s_ab)

            status_ab = "PASS" if passed_ab else "FAIL"
            print(
                f"  [{label}] cap{cap_a} vs cap{cap_b}: {status_ab}"
                f"  max_abs={s_ab['max_abs_err']:.3e}"
                f"  rmse={s_ab['rmse']:.3e}"
                f"  rel={s_ab['rel_err']:.3e}"
            )

            # ----------------------------------------------------------------
            # Compare cap_b vs pilot reference (informational)
            # ----------------------------------------------------------------
            if ref_output is not None:
                if not check_finite(ref_output, f"{label} raw_provider_output"):
                    ok = False
                passed_ref, s_ref = compare_outputs(out_b, ref_output, args.atol, args.rtol,
                                                    f"{label} cap_b vs pilot_ref")
                status_ref = "PASS" if passed_ref else "WARN"
                print(
                    f"  [{label}] cap{cap_b} vs pilot_ref: {status_ref}"
                    f"  max_abs={s_ref['max_abs_err']:.3e}"
                    f"  rmse={s_ref['rmse']:.3e}"
                    f"  rel={s_ref['rel_err']:.3e}"
                )

            if not ok or not passed_ab:
                failures += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 72)
    print(f"Checked: {total}  Failures: {failures}")
    if stats_table:
        all_max = max(s["max_abs_err"] for s in stats_table if "max_abs_err" in s)
        all_rmse = float(np.sqrt(np.mean([s["rmse"] ** 2 for s in stats_table if "rmse" in s])))
        print(f"Overall  max_abs_err={all_max:.3e}  rms_rmse={all_rmse:.3e}")
    print(f"atol={args.atol}  rtol={args.rtol}")
    if failures == 0:
        print("OVERALL: PASS")
        return 0
    else:
        print(f"OVERALL: FAIL  ({failures}/{total} items failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
