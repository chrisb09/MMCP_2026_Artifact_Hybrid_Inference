#!/usr/bin/env python3
"""Validate the PhyDLL cap-50000 rerun campaign against canonical artifacts.

Checks structure, configuration, workload, telemetry and completion. Numerical
correctness of the 50000 cap is established separately by
check_phydll_cap_correctness.py; this script does not replay tensors.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
STEPS = (15, 44, 73, 102, 131, 160, 189)
MEASURED = (102, 131, 160, 189)
RANKS = {"24c1g": 24, "24p24c1g": 48, "96c4g": 96, "96p96c4g": 192}
OVERLAP = {"transformer": "0", "mlp": "4"}
CAP = {"transformer": "50000", "mlp": "1000000"}
MODELS = {
    "transformer": "transformer_inference_scripted_fw2.pt",
    "mlp": "mmcp_test_mlp_m5_cpu.pt",
}
CASE = {"default": "phydll-blocking", "readiness": "phydll-readiness"}
WORKLOAD_KEYS = {
    "cubes_per_rank", "samples_per_rank", "cubes_aggregate",
    "samples_aggregate", "cubeD", "overlap", "solver_ranks",
}
CANONICAL_FIELDS = {
    "cubes_per_rank": "cubes_per_rank",
    "samples_per_rank": "samples_per_rank",
    "cubes_aggregate": "cubes_aggregate",
    "samples_aggregate": "samples_aggregate",
}


def info(path):
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def workload(text):
    matches = re.findall(r"ML_WORKLOAD_LOG\s+([^\n]+)", text)
    if len(matches) != 1:
        raise ValueError(f"expected one workload record, found {len(matches)}")
    return {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", matches[0])}


def resolve(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_canonical(path):
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows[row["cell"]] = row
    return rows


def resolve_run(row):
    status_dir = row["status_dir"]
    jobid = row["jobid"]
    if not status_dir or not re.fullmatch(r"[0-9]+", jobid):
        raise ValueError(f"invalid status_dir or jobid: {status_dir!r}, {jobid!r}")
    parent = resolve(status_dir).parent
    matches = []
    for candidate in parent.iterdir():
        if not candidate.is_dir() or jobid not in candidate.name.split("_"):
            continue
        run_info = candidate / "run.info"
        if run_info.is_file() and info(run_info).get("CELL") == row["cell"]:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"expected unique run directory for jobid={jobid} CELL={row['cell']!r} "
            f"in {parent}; found {len(matches)}: {', '.join(map(str, sorted(matches)))}"
        )
    return matches[0]


def validate(row, canonical):
    run = resolve_run(row)
    run_info = info(run / "run.info")
    model = row["model"]
    placement = row["placement"]
    wait = row["wait"]
    cap = row["cap"]
    if run_info.get("CELL") != row["cell"]:
        raise ValueError(f"CELL: {run_info.get('CELL')!r} != {row['cell']!r}")
    if run_info.get("REP") != row["rep"]:
        raise ValueError(f"REP: {run_info.get('REP')!r} != {row['rep']!r}")
    for key, value in {
        "RUN_STEPS": "200",
        "SOLUTION_INTERVAL": "10000",
        "ML_CUBE_OVERLAP": OVERLAP[model],
        "MAIA_PHYDLL_BATCHSIZE": cap,
        "DEBUG_EXPORT": "0",
        "EXIT_STATUS": "0",
        "SOLVER_EXIT_STATUS": "0",
    }.items():
        if run_info.get(key) != value:
            raise ValueError(f"{key}: {run_info.get(key)!r} != {value!r}")
    if run_info.get("PHYDLL_WAIT", wait) != wait:
        raise ValueError(f"PHYDLL_WAIT: {run_info.get('PHYDLL_WAIT')!r} != {wait!r}")
    if "CASE" in run_info and run_info["CASE"] != CASE[wait]:
        raise ValueError(f"CASE: {run_info['CASE']!r} != {CASE[wait]!r}")
    if Path(run_info.get("MODEL", "")).name != MODELS[model]:
        raise ValueError(f"MODEL: {run_info.get('MODEL')!r} != {MODELS[model]!r}")
    try:
        wall = float(run_info["BENCHMARK_SOLVER_WALL_SECONDS"])
    except (KeyError, ValueError):
        raise ValueError("missing or invalid BENCHMARK_SOLVER_WALL_SECONDS")
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("invalid BENCHMARK_SOLVER_WALL_SECONDS")
    if not any(key.startswith("NODE_RAM_PEAK_MIB_") for key in run_info):
        raise ValueError("missing node RAM telemetry")
    if not any(key.startswith("GPU_VRAM_PEAK_MIB_") for key in run_info):
        raise ValueError("missing GPU VRAM telemetry")

    solver = (run / "solver.log").read_text(errors="replace")
    steps = re.findall(r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)", solver)
    if tuple(int(step) for step, _ in steps) != STEPS:
        raise ValueError(f"unexpected inference steps: {[s for s, _ in steps]}")
    timings = {int(step): float(value) for step, value in steps}
    if not all(math.isfinite(value) and value > 0 for value in timings.values()):
        raise ValueError("invalid inference timings")
    work = workload(solver)
    if set(work) != WORKLOAD_KEYS:
        raise ValueError(f"workload keys: {sorted(work)}")
    if work["solver_ranks"] != RANKS[placement]:
        raise ValueError(f"solver_ranks: {work['solver_ranks']} != {RANKS[placement]}")
    if work["overlap"] != int(OVERLAP[model]):
        raise ValueError(f"workload overlap: {work['overlap']} != {OVERLAP[model]}")

    original = row["cell"].replace("-cap50000", "")
    reference = canonical.get(original)
    if reference is None:
        raise ValueError(f"canonical row missing: {original}")
    for key, column in CANONICAL_FIELDS.items():
        if work[key] != int(reference[column]):
            raise ValueError(f"workload {key}: {work[key]} != {reference[column]}")
    if work["solver_ranks"] != int(reference["solver_ranks"]):
        raise ValueError("workload solver_ranks differs from canonical")
    reference_dir = resolve(reference["run_dir"])
    if (reference_dir / "solver.log").is_file():
        reference_work = workload((reference_dir / "solver.log").read_text(errors="replace"))
        if work != reference_work:
            raise ValueError(f"workload differs from canonical: {work} != {reference_work}")
        reference_props = reference_dir / "properties.toml"
        run_props = run / "properties.toml"
        if reference_props.is_file() and run_props.is_file():
            with reference_props.open("rb") as handle:
                expected_props = tomllib.load(handle)
            with run_props.open("rb") as handle:
                actual_props = tomllib.load(handle)
            if actual_props != expected_props:
                raise ValueError("properties.toml differs from canonical")

    measured = sum(timings[step] for step in MEASURED) / len(MEASURED)
    return {
        "cell": row["cell"],
        "jobid": row["jobid"],
        "status": "ok",
        "reason": "",
        "measured_mean_ms": f"{measured:.6f}",
        "wall_seconds": f"{wall:.0f}",
        "cubes_per_rank": work["cubes_per_rank"],
        "samples_per_rank": work["samples_per_rank"],
        "solver_ranks": work["solver_ranks"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical", type=Path,
                        default=ROOT / "results/cmi_bench_consolidated/runs.csv")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("ERROR: empty manifest", file=sys.stderr)
        return 1
    if len({row["cell"] for row in rows}) != len(rows):
        print("ERROR: duplicate cells in manifest", file=sys.stderr)
        return 1
    canonical = load_canonical(args.canonical)
    results = []
    failures = 0
    for row in rows:
        try:
            results.append(validate(row, canonical))
            print(f"[ok]   {row['cell']}")
        except (ValueError, KeyError, OSError) as error:
            failures += 1
            results.append({
                "cell": row["cell"], "jobid": row["jobid"], "status": "fail",
                "reason": str(error), "measured_mean_ms": "", "wall_seconds": "",
                "cubes_per_rank": "", "samples_per_rank": "", "solver_ranks": "",
            })
            print(f"[FAIL] {row['cell']}: {error}")
    report = args.report or args.manifest.parent / "validation.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{len(results) - failures}/{len(results)} passed; report: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
