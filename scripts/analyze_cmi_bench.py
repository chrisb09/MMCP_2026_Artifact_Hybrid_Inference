#!/usr/bin/env python3
"""Summarize MAIA [TBL] ML-step benchmark runs into a CSV.

Scans run directories (from slurm/run_cmi_smoke_gpu.sh and
slurm/cmi_bench_het_engine.sh) and extracts, per run:

  - cell identity: model, overlap, solver ranks, GPUs, library/mode, batch
  - per-inference ML-step makespan (ML_STEP_TIMING ml_ms)
  - warmup vs measured classification (first two calls are warmup)
  - host memory peak (cgroup sample) and solver-reported total RAM
  - GPU execution evidence and solver wall time

Usage:
  scripts/analyze_cmi_bench.py [ROOT ...] [--out results/cmi_bench_summary.csv]

ROOT defaults to scratch/cmi_bench and scratch/cmi_smokes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics

STEP_RE = re.compile(r"ML_STEP_TIMING step=(\d+) ml_ms=([0-9.]+)")
WORKLOAD_RE = re.compile(
    r"ML_WORKLOAD_LOG cubes_per_rank=(\d+) samples_per_rank=(\d+) "
    r"cubes_aggregate=(\d+) samples_aggregate=(\d+)")
AIXMEM_RE = re.compile(
    r"AIX_GPU_MEM device=(\d+) batch_dim=(-?\d+) batchsize=(\d+) "
    r"num_batches=(\d+) used_mib=(\d+) total_mib=(\d+) call=(\d+)")
WARMUP_CALLS = 2


def read_kv(path: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if "=" in line:
                    k, _, v = line.partition("=")
                    kv[k.strip()] = v.strip()
    except OSError:
        pass
    return kv


def _mib_to_gb(mib: str) -> str:
    try:
        return f"{float(mib) / 1024.0:.2f}"
    except (TypeError, ValueError):
        return ""


def node_peaks(kv: dict[str, str]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split NODE_RAM_PEAK_MIB_<host> / GPU_VRAM_PEAK_MIB_<host>_gpu<i> keys."""
    node: dict[str, str] = {}
    gpu: dict[str, dict[str, str]] = {}
    for key, val in kv.items():
        if key.startswith("NODE_RAM_PEAK_MIB_"):
            node[key[len("NODE_RAM_PEAK_MIB_"):]] = val
        elif key.startswith("GPU_VRAM_PEAK_MIB_"):
            host, _, gidx = key[len("GPU_VRAM_PEAK_MIB_"):].rpartition("_gpu")
            gpu.setdefault(host, {})[gidx] = val
    return node, gpu


def model_label(path: str) -> str:
    if "mlp" in path:
        return "MLP"
    if "transformer" in path:
        return "Transformer"
    return "?"


def solver_memory_total(log_path: str) -> str:
    total = ""
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                if "Total physical memory usage (RAM)" in line:
                    m = re.search(r":\s*([0-9.]+)\s*GB", line)
                    if m:
                        total = m.group(1)
    except OSError:
        pass
    return total


def parse_run(run_dir: str) -> dict[str, object] | None:
    info = read_kv(os.path.join(run_dir, "run.info"))
    if not info:
        return None
    log = os.path.join(run_dir, "solver.log")
    steps: list[tuple[int, float]] = []
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                m = STEP_RE.search(line)
                if m:
                    steps.append((int(m.group(1)), float(m.group(2))))
    except OSError:
        pass
    measured = [v for _, v in steps[WARMUP_CALLS:]]
    warmup = [v for _, v in steps[:WARMUP_CALLS]]
    workload = {"cubes_per_rank": "", "samples_per_rank": "",
                "cubes_aggregate": "", "samples_aggregate": ""}
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                m = WORKLOAD_RE.search(line)
                if m:
                    workload = dict(zip(workload.keys(), m.groups()))
    except OSError:
        pass
    aixmem = []
    try:
        with open(log, errors="replace") as fh:
            for line in fh:
                m = AIXMEM_RE.search(line)
                if m:
                    aixmem.append(m.groups())
    except OSError:
        pass
    evidence = glob.glob(os.path.join(run_dir, "aix_gpu_diagnostics", "aix_rank_*.csv"))
    node_ram, gpu_vram = node_peaks(info)
    proc_peaks = [int(v) for k, v in info.items()
                  if k.startswith("GPU_PROC_PEAK_") and v.isdigit()]
    batch_host = info.get("BATCH_HOST", "")
    ml_host = info.get("HET_GPU_NODELIST", batch_host)
    ml_host = ml_host.split(",")[0].split()[0] if ml_host else batch_host
    solver_mib = node_ram.get(batch_host, "")
    ml_mib = node_ram.get(ml_host, solver_mib)
    ml_gpus = gpu_vram.get(ml_host, {})
    gpu_peak = max((int(v) for v in ml_gpus.values() if v.isdigit()), default=0)
    return {
        **workload,
        "batch_host": batch_host,
        "ml_host": ml_host,
        "solver_node_ram_peak_gb": _mib_to_gb(solver_mib),
        "ml_node_ram_peak_gb": _mib_to_gb(ml_mib),
        "gpu_vram_peak_gb": f"{gpu_peak / 1024.0:.2f}" if gpu_peak else "",
        "gpu_proc_peak": str(max(proc_peaks)) if proc_peaks else "",
        "aix_gpu_used_mib_first": aixmem[0][4] if aixmem else "",
        "aix_gpu_used_mib_max": str(max(int(r[4]) for r in aixmem)) if aixmem else "",
        "aix_gpu_batch_dim": aixmem[0][1] if aixmem else "",
        "aix_gpu_batchsize": aixmem[0][2] if aixmem else "",
        "aix_gpu_num_batches": aixmem[0][3] if aixmem else "",
        "gpu_vram_per_gpu_mib": ";".join(f"{k}={v}" for k, v in sorted(ml_gpus.items())),
        "node_ram_peaks_mib": ";".join(f"{k}={v}" for k, v in sorted(node_ram.items())),
        "run_dir": run_dir,
        "cell": info.get("CELL", info.get("CASE", "?")),
        "model": model_label(info.get("MODEL", "")),
        "overlap": info.get("ML_CUBE_OVERLAP", ""),
        "solver_ranks": info.get("SOLVER_RANKS", info.get("HET_CPU_TASKS", "")),
        "het_cpu_tasks": info.get("HET_CPU_TASKS", ""),
        "het_gpu_tasks": info.get("HET_GPU_TASKS", ""),
        "gpus": info.get("HET_GPUS", info.get("SLURM_JOB_GPUS", "")),
        "aix_comm": info.get("AIX_COMM", ""),
        "aix_credits": info.get("AIX_CREDITS", ""),
        "aix_batchsize": info.get("MAIA_AIX_BATCHSIZE", ""),
        "solution_interval": info.get("SOLUTION_INTERVAL", ""),
        "n_inferences": len(steps),
        "n_measured": len(measured),
        "warmup_ms": ";".join(f"{v:.1f}" for v in warmup),
        "measured_ms": ";".join(f"{v:.1f}" for v in measured),
        "measured_median_ms": f"{statistics.median(measured):.1f}" if measured else "",
        "measured_mean_ms": f"{statistics.mean(measured):.1f}" if measured else "",
        "mem_peak_bytes": info.get("MEMORY_PEAK_BYTES", ""),
        "solver_ram_gb": solver_memory_total(log),
        "wall_seconds": info.get("BENCHMARK_SOLVER_WALL_SECONDS", ""),
        "solver_exit": info.get("SOLVER_EXIT_STATUS", ""),
        "gpu_evidence": len(evidence),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=None)
    ap.add_argument("--out", default="results/cmi_bench_summary.csv")
    args = ap.parse_args()
    roots = args.roots or ["scratch/cmi_bench", "scratch/cmi_smokes"]
    rows = []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(run_dir):
                row = parse_run(run_dir)
                if row:
                    rows.append(row)
    if not rows:
        print("No runs found.")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.out}")
    for r in rows:
        print(f"  {r['cell']:32s} {str(r['model']):12s} ov={str(r['overlap']):2s} "
              f"meas={str(r['n_measured']):2s} median={str(r['measured_median_ms']):>9s}ms "
              f"ram={str(r['solver_ram_gb']):>7s}GB")


if __name__ == "__main__":
    main()
