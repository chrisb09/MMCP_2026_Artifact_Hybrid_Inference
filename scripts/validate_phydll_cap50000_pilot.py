#!/usr/bin/env python3
"""Check preserved pilot artifacts against a canonical run, not numerical parity."""
import argparse
from pathlib import Path
import re
import sys
import tomllib


def info(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def workload(text):
    matches = re.findall(r"ML_WORKLOAD_LOG\s+([^\n]+)", text)
    if len(matches) != 1:
        raise ValueError(f"expected one workload record, found {len(matches)}")
    return dict(re.findall(r"(\w+)=(\d+)", matches[0]))


def validate(run, canonical):
    current, baseline = info(run / "run.info"), info(canonical / "run.info")
    expected = {"CASE": "phydll-blocking", "RUN_STEPS": "200", "SOLVER_RANKS": "24",
                "DL_RANKS": "1", "MAIA_PHYDLL_BATCHSIZE": "50000", "DEBUG_EXPORT": "1",
                "ML_CUBE_OVERLAP": "0", "SOLUTION_INTERVAL": "10000", "EXIT_STATUS": "0",
                "SOLVER_EXIT_STATUS": "0"}
    for key, value in expected.items():
        if current.get(key) != value:
            raise ValueError(f"{key}: expected {value}, got {current.get(key)}")
    for key in ("CASE", "RUN_STEPS", "SOLVER_RANKS", "DL_RANKS", "ML_CUBE_OVERLAP", "SOLUTION_INTERVAL", "MODEL"):
        if not baseline.get(key) or current.get(key) != baseline[key]:
            raise ValueError(f"canonical mismatch: {key}")
    with (run / "properties.toml").open("rb") as f:
        props = tomllib.load(f)
    with (canonical / "properties.toml").open("rb") as f:
        refprops = tomllib.load(f)
    if props != refprops:
        raise ValueError("properties differ from canonical")
    log = (run / "solver.log").read_text(errors="replace")
    ref = (canonical / "solver.log").read_text(errors="replace")
    steps = re.findall(r"ML_STEP_TIMING\s+step=(\d+)", log)
    if steps != ["15", "44", "73", "102", "131", "160", "189"]:
        raise ValueError(f"unexpected inference step records: {steps}")
    work = workload(log)
    if set(work) != {"cubes_per_rank", "samples_per_rank", "cubes_aggregate", "samples_aggregate", "cubeD", "overlap", "solver_ranks"} or work != workload(ref):
        raise ValueError("workload missing fields or differs from canonical")
    if not any(k.startswith("GPU_VRAM_PEAK_MIB_") for k in current) or not any(k.startswith("NODE_RAM_PEAK_MIB_") for k in current):
        raise ValueError("missing GPU or node RAM sampler telemetry")
    if not (run / "debug_dumps").is_dir() or not any((run / "debug_dumps").iterdir()):
        raise ValueError("debug exports missing")
    print("Artifact checks passed; cap is configured, not independently measured.")
    print("Telemetry presence does not prove GPU execution; inspect solver.log and telemetry.")
    print("Numerical correctness is NOT established; compare debug exports directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("canonical", type=Path)
    args = parser.parse_args()
    try:
        validate(args.run, args.canonical)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
