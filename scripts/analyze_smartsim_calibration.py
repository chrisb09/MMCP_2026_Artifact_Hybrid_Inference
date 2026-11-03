#!/usr/bin/env python3
"""Analyze SmartSim RedisAI aggregation calibration runs.

For each model x placement, compare the control (aggregation disabled) with the
1 ms and 5 ms aggregation candidates: measured inference time and numerical
agreement of rank-0 exported outputs. Emits a CSV and a per-group recommendation.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import sys
import tomllib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEPS = (15, 44, 73, 102, 131, 160, 189)
MEASURED = (102, 131, 160, 189)
VARIANTS = ("ctrl", "t1ms", "t5ms")
MODELS = {
    "transformer": "transformer_inference_scripted_fw2.pt",
    "mlp": "mmcp_test_mlp_m5_cpu.pt",
}
RANKS = {"24c1g": 24, "24p24c1g": 48, "96c4g": 96, "96p96c4g": 192}


def info(path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def debug_manifest(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def find_run(scratch, cell, jobid):
    matches = []
    for candidate in scratch.iterdir():
        if not candidate.is_dir() or jobid not in candidate.name:
            continue
        run_info = candidate / "run.info"
        if run_info.is_file() and info(run_info).get("CELL") == cell:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def load_output(run, inference):
    root = run / "debug_dumps"
    manifest = debug_manifest(root / f"current_rank_0_inference_{inference}_manifest.txt")
    # The exporter dumps working_output_ in CouplingOutput precision, with the
    # same shape as raw_provider_output (which can have a different dtype).
    dtype = np.dtype({"float32": np.float32, "float64": np.float64}[
        manifest["reconstructed_field_dtype"]])
    count = int(manifest["raw_provider_output_count"])
    shape = (int(manifest["n_fields"]) * int(manifest["num_cubes"]),
             int(manifest["forecast_window"]), int(manifest["cube_size"]))
    if any(dim <= 0 for dim in shape) or math.prod(shape) != count:
        raise ValueError("invalid denormalized output dimensions or count")
    path = root / f"current_rank_0_inference_{inference}_denormalized_output.bin"
    if count <= 0 or path.stat().st_size != count * dtype.itemsize:
        raise ValueError(f"{path.name}: invalid size for {count} {dtype} values")
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        raise ValueError(f"{path.name}: {values.size} values, expected {count}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: non-finite values")
    return values.reshape(shape)


def compare(candidate, control):
    if candidate.shape != control.shape:
        return None, None
    absolute = float(np.max(np.abs(candidate - control)))
    scale = np.max(np.abs(control))
    relative = absolute / float(scale) if scale else (0.0 if absolute == 0 else math.inf)
    return absolute, relative


def analyze(row, scratch, atol, rtol):
    model, placement = row["model"], row["placement"]
    run = find_run(scratch, row["cell"], row["jobid"])
    if run is None:
        raise ValueError("run directory missing")
    run_info = info(run / "run.info")
    for key, value in {
        "EXIT_STATUS": "0",
        "SOLVER_EXIT_STATUS": "0",
        "ML_CUBE_OVERLAP": "0" if model == "transformer" else "4",
    }.items():
        if run_info.get(key) != value:
            raise ValueError(f"{key}: {run_info.get(key)!r} != {value!r}")
    if Path(run_info.get("MODEL", "")).name != MODELS[model]:
        raise ValueError(f"MODEL: {run_info.get('MODEL')!r} != {MODELS[model]!r}")
    enabled = row["variant"] != "ctrl"
    expected_enabled = "1" if enabled else "0"
    if run_info.get("CELL") != row["cell"]:
        raise ValueError(f"CELL: {run_info.get('CELL')!r} != {row['cell']!r}")
    if run_info.get("SMARTSIM_AGGREGATION_ENABLED") != expected_enabled:
        raise ValueError("aggregation enabled flag mismatch")
    if run_info.get("SMARTSIM_BATCH_SIZE_EFFECTIVE") != row["batch_size"]:
        raise ValueError("batch size effective mismatch")
    if run_info.get("SMARTSIM_MIN_BATCH_SIZE_EFFECTIVE") != row["min_batch_size"]:
        raise ValueError("min batch size effective mismatch")
    if run_info.get("SMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE") != row["timeout_ms"]:
        raise ValueError("timeout effective mismatch")
    config = tomllib.loads((run / "config_smartsim.toml").read_text())["library"]
    if enabled:
        for key, expected in (("batch_size", row["batch_size"]),
                              ("min_batch_size", row["min_batch_size"]),
                              ("min_batch_timeout", row["timeout_ms"])):
            if str(config.get(key)) != expected:
                raise ValueError(f"config {key}: {config.get(key)!r} != {expected!r}")

    solver = (run / "solver.log").read_text(errors="replace")
    steps = re.findall(r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)", solver)
    if tuple(int(step) for step, _ in steps) != STEPS:
        raise ValueError("unexpected inference steps")
    timings = {int(step): float(value) for step, value in steps}
    if not all(math.isfinite(value) and value > 0 for value in timings.values()):
        raise ValueError("invalid inference timings")
    work = re.findall(r"ML_WORKLOAD_LOG\s+([^\n]+)", solver)
    if len(work) != 1:
        raise ValueError("expected one workload record")
    ranks = int(dict(re.findall(r"(\w+)=(\d+)", work[0]))["solver_ranks"])
    if ranks != RANKS[placement]:
        raise ValueError(f"solver_ranks: {ranks} != {RANKS[placement]}")

    outputs = [load_output(run, inference) for inference in (1, 2)]
    measured = [timings[step] for step in MEASURED]
    return {
        "run": run,
        "mean_ms": statistics.mean(measured),
        "stdev_ms": statistics.stdev(measured),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch", type=Path,
                        default=ROOT / "scratch/smartsim_calibration")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("ERROR: empty manifest", file=sys.stderr)
        return 1
    groups = {}
    failures = 0
    for row in rows:
        variant = row["cell"].rsplit("-", 1)[-1]
        if variant not in VARIANTS:
            print(f"[FAIL] {row['cell']}: unknown variant", file=sys.stderr)
            failures += 1
            continue
        row = dict(row, variant=variant)
        try:
            result = analyze(row, args.scratch, args.atol, args.rtol)
        except (ValueError, KeyError, OSError) as error:
            print(f"[FAIL] {row['cell']}: {error}", file=sys.stderr)
            failures += 1
            groups.setdefault((row["placement"], row["model"]), {})[variant] = None
            continue
        groups.setdefault((row["placement"], row["model"]), {})[variant] = (row, result)

    report_rows = []
    for (placement, model), variants in sorted(groups.items()):
        control = variants.get("ctrl")
        if not control:
            print(f"[FAIL] {placement}/{model}: control run missing or invalid", file=sys.stderr)
            failures += 1
            continue
        control_row, control_result = control
        for variant in VARIANTS:
            entry = variants.get(variant)
            if not entry:
                failures += 1
                report_rows.append({
                    "placement": placement, "model": model, "variant": variant,
                    "status": "fail", "reason": "missing or invalid run",
                })
                continue
            row, result = entry
            absolute = relative = None
            if variant != "ctrl":
                diffs = [compare(candidate, reference)
                         for candidate, reference in zip(result["outputs"], control_result["outputs"])]
                if any(item[0] is None for item in diffs):
                    failures += 1
                    report_rows.append({
                        "placement": placement, "model": model, "variant": variant,
                        "status": "fail", "reason": "output shape mismatch",
                    })
                    continue
                absolute = max(item[0] for item in diffs)
                relative = max(item[1] for item in diffs)
            status = "ok"
            reason = ""
            if absolute is not None and not (absolute <= args.atol or relative <= args.rtol):
                status, reason = "fail", "output differs from control"
                failures += 1
                print(f"[FAIL] {placement}/{model}/{variant}: output differs "
                      f"(abs={absolute:.6g}, rel={relative:.6g})", file=sys.stderr)
            report_rows.append({
                "placement": placement, "model": model, "variant": variant,
                "batch_size": row["batch_size"], "min_batch_size": row["min_batch_size"],
                "timeout_ms": row["timeout_ms"],
                "enabled": "0" if variant == "ctrl" else "1",
                "mean_ms": f"{result['mean_ms']:.6f}",
                "stdev_ms": f"{result['stdev_ms']:.6f}",
                "speedup_vs_ctrl": "" if variant == "ctrl" else
                    f"{control_result['mean_ms'] / result['mean_ms']:.4f}",
                "max_abs_diff": "" if absolute is None else f"{absolute:.6g}",
                "max_rel_diff": "" if relative is None else f"{relative:.6g}",
                "status": status, "reason": reason,
            })
        candidates = [entry for entry in report_rows
                      if entry["placement"] == placement and entry["model"] == model
                      and entry["variant"] != "ctrl" and entry["status"] == "ok"]
        if candidates:
            best = min(candidates, key=lambda entry: float(entry["mean_ms"]))
            print(f"[pick] {placement}/{model}: {best['variant']} "
                  f"({float(best['mean_ms']):.3f} ms, "
                  f"{float(best['speedup_vs_ctrl']):.3f}x)")

    fields = ["placement", "model", "variant", "batch_size", "min_batch_size",
              "timeout_ms", "enabled", "mean_ms", "stdev_ms", "speedup_vs_ctrl",
              "max_abs_diff", "max_rel_diff", "status", "reason"]
    report = args.report or args.manifest.parent / "analysis.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(f"\nreport: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
