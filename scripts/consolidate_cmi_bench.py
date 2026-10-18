#!/usr/bin/env python3
"""Consolidate the explicit grouped CMI campaign and its ordered retry ledgers.

No directory age, performance, pilot ledger, or failed-run fallback participates
in selection. CSV latencies retain their logged decimal strings; statistics use
Python floats without presentation rounding. RAM is whole-node usage and VRAM
is device-wide usage, not process allocations (MiB and binary GiB).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


# CPU-side solver ranks, GPU-side solver ranks, configured inference devices.
PLACEMENTS = {
    "24c1g": (24, 0, 1), "24p24c1g": (24, 24, 1),
    "96c4g": (96, 0, 4), "96p96c4g": (96, 96, 4),
}
MODELS = {"transformer": "transformer_inference_scripted_fw2.pt",
          "mlp": "mmcp_test_mlp_m5_cpu.pt"}
STEPS = (15, 44, 73, 102, 131, 160, 189)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP_RE = re.compile(r"ML_STEP_TIMING step=(\d+) ml_ms=(" + NUMBER + r")(?=\s|$)")
WORKLOAD_RE = re.compile(
    r"ML_WORKLOAD_LOG cubes_per_rank=(\d+) samples_per_rank=(\d+) "
    r"cubes_aggregate=(\d+) samples_aggregate=(\d+).*?solver_ranks=(\d+)")
AIXMEM_RE = re.compile(
    r"AIX_GPU_MEM device=(\d+) batch_dim=(-?\d+) batchsize=(\d+) "
    r"num_batches=(\d+) used_mib=(\d+) total_mib=(\d+) call=(\d+)")
WORKLOAD_FIELDS = ("cubes_per_rank", "samples_per_rank", "cubes_aggregate",
                   "samples_aggregate", "workload_solver_ranks")
CONFIG_FIELDS = ("placement", "model", "provider", "layout", "comm", "wait")
JOB_RE = re.compile(r"_([0-9]+)_[A-Za-z0-9]+$")


def expected_matrix() -> dict[str, dict]:
    """Return the explicit 288 cell identities and intended configurations."""
    configs = [("aix", layout, comm, "")
               for layout in ("default", "affine", "numa")
               for comm in ("coll", "p2p1", "p2pfull")]
    configs += [("phydll", "", "", wait) for wait in ("default", "readiness")]
    configs += [("smartsim", "shared", "", "")]
    cells = {}
    for placement, (cpu, gpu, devices) in PLACEMENTS.items():
        for model in MODELS:
            for rep in (1, 2, 3):
                group = f"{placement}-{model}-r{rep}"
                for provider, layout, comm, wait in configs:
                    mode = "-".join(v for v in (layout, comm, wait) if v)
                    cell = f"{provider}-{placement}-{mode}-{model}-r{rep}"
                    cells[cell] = dict(cell=cell, group=group, placement=placement,
                                       model=model, provider=provider, layout=layout,
                                       comm=comm, wait=wait, rep=rep,
                                       overlap=0 if model == "transformer" else 4,
                                       solver_ranks=cpu + gpu, cpu_solver_ranks=cpu,
                                       gpu_solver_ranks=gpu, gpu_count=devices)
    return cells


def read_kv(path: Path) -> dict[str, str]:
    """Read raw metadata, rejecting conflicting repeated keys."""
    result = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in result and result[key] != value:
            raise ValueError(f"{path}: conflicting metadata key {key}")
        result[key] = value
    return result


def source_rows(root: Path, name: str, sources: dict, header: bool = True) -> list:
    """Read one ledger/status file and record its exact content hash."""
    path = root / "results" / name
    data = path.read_bytes()
    sources[str(path.relative_to(root))] = {
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    lines = data.decode("utf-8").splitlines()
    reader = csv.DictReader(lines) if header else csv.reader(lines)
    rows = list(reader)
    if header and (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)):
        raise ValueError(f"{name}: missing or duplicate headers")
    for row in rows:
        if header and (None in row or any(v is None for v in row.values())):
            raise ValueError(f"{name}: malformed CSV row {row}")
    return rows


def job_groups(root: Path, name: str, prefix: str, groups: set, sources: dict) -> dict:
    result = {}
    for row in source_rows(root, name, sources, header=False):
        if len(row) != 2:
            raise ValueError(f"{name}: expected group,jobid: {row}")
        label, jobid = row
        group = label.removeprefix(prefix)
        if not label.startswith(prefix) or group not in groups or group in result:
            raise ValueError(f"{name}: duplicate/unexpected group {label}")
        if not jobid.isdigit() or int(jobid) <= 0:
            raise ValueError(f"{name}: invalid job ID {jobid}")
        result[group] = jobid
    return result


def select_cells(root: Path, matrix: dict, sources: dict) -> dict:
    """Apply only the documented ledgers, in precedence order, with lineage."""
    groups = {m["group"] for m in matrix.values()}
    original = job_groups(root, "cmi_bench_grouped_jobids.txt", "grp-", groups, sources)
    if set(original) != groups:
        raise ValueError(f"original ledger missing groups: {sorted(groups - original.keys())}")
    selected = {}
    for cell, spec in matrix.items():
        jid = original[spec["group"]]
        selected[cell] = dict(jobid=jid, original_jobid=jid, selection="original",
                              retry_reason="", lineage=[{"selection": "original", "jobid": jid}])
    quota = job_groups(root, "cmi_bench_quota_retry_jobids.txt", "quota-retry-grp-", groups, sources)
    stages = (
        ("smartsim_retry", "cmi_bench_smartsim_retry_jobids.csv", "original_group_jobid"),
        ("quota_retry", "cmi_bench_quota_retry_cells.csv", "original_jobid"),
        ("shutdown_retry", "cmi_bench_shutdown_retry_cells.csv", "failed_jobid"),
        ("final_retry", "cmi_bench_final_retry_cells.csv", "original_jobid"),
    )
    for stage, name, parent_key in stages:
        seen = set()
        stage_jobs = {}
        for row in source_rows(root, name, sources):
            cell = row["cell"]
            if cell not in matrix or cell in seen:
                raise ValueError(f"{name}: duplicate/unexpected cell {cell}")
            seen.add(cell)
            if stage == "smartsim_retry" and matrix[cell]["provider"] != "smartsim":
                raise ValueError(f"{name}: non-SmartSim cell {cell}")
            previous = selected[cell]
            if row[parent_key] != previous["jobid"]:
                raise ValueError(f"{name}: {cell}: lineage {row[parent_key]} != selected {previous['jobid']}")
            jobid = quota[matrix[cell]["group"]] if stage == "quota_retry" else row["retry_jobid"]
            if not jobid.isdigit() or int(jobid) <= 0 or jobid == previous["jobid"]:
                raise ValueError(f"{name}: invalid retry job ID {jobid}")
            previous.update(jobid=jobid, selection=stage, retry_reason=row["reason"])
            previous["lineage"].append(dict(row, selection=stage, jobid=jobid, source=name))
            group = matrix[cell]["group"]
            if group in stage_jobs and stage_jobs[group] != jobid:
                raise ValueError(f"{name}: conflicting retry jobs for group {group}")
            stage_jobs[group] = jobid
        if stage == "quota_retry" and {matrix[c]["group"] for c in seen} != set(quota):
            raise ValueError("quota manifest and job group ledger differ")
        # These redundant submission receipts are evidence, not selection inputs.
        receipt = f"cmi_bench_{stage}_jobids.txt"
        if stage in ("shutdown_retry", "final_retry") and (root / "results" / receipt).exists():
            jobs = job_groups(root, receipt, stage.replace("_", "-") + "-grp-", groups, sources)
            if jobs != stage_jobs:
                raise ValueError(f"{receipt}: submission receipt differs from cell ledger")
    # Status files contain no job IDs: associate them only via the selected ledger.
    cache = {}
    for cell, selection in selected.items():
        stage = selection["selection"]
        if stage == "smartsim_retry":
            selection["status_source"] = "run.info (standalone SmartSim retry)"
            continue
        prefix = "" if stage == "original" else stage + "_"
        group = matrix[cell]["group"]
        name = prefix + "group_" + group.replace("-", "_") + "_cells.csv"
        if name not in cache:
            statuses = {}
            for row in source_rows(root, name, sources, header=False):
                if len(row) != 2:
                    raise ValueError(f"{name}: malformed status row {row}")
                identity, status = row
                if identity in statuses or identity not in matrix or matrix[identity]["group"] != group:
                    raise ValueError(f"{name}: duplicate/unexpected cell {identity}")
                statuses[identity] = status
            cache[name] = statuses
        if cache[name].get(cell) != "0":
            raise ValueError(f"{name}: {cell}: selected status is {cache[name].get(cell)!r}, not 0")
        selection["status_source"] = "results/" + name
    return selected


def index_runs(root: Path) -> dict:
    """Index only immediate run directories by (CELL metadata, directory job ID)."""
    index = defaultdict(list)
    for base in (root / "scratch/cmi_bench", root / "scratch/cmi_smokes"):
        for directory in sorted(base.iterdir()):
            match = JOB_RE.search(directory.name)
            if directory.is_dir() and match and (directory / "run.info").is_file():
                # Irrelevant pilot metadata must not prevent campaign selection.
                cells = [line.partition("=")[2].strip()
                         for line in (directory / "run.info").read_text().splitlines()
                         if line.partition("=")[0].strip() == "CELL"]
                for cell in set(cells):
                    index[(cell, match[1])].append(directory)
    return index


def parse_selected_run(root: Path, directory: Path, spec: dict, selection: dict) -> tuple[dict, list]:
    """Validate a designated run; return its canonical row and seven call rows."""
    info = read_kv(directory / "run.info")
    def require(key, expected):
        if info.get(key) != str(expected):
            raise ValueError(f"{spec['cell']}: {key}={info.get(key)!r}, expected {expected!r}")

    require("CELL", spec["cell"])
    require("REP", spec["rep"])
    require("SOLVER_EXIT_STATUS", 0)
    require("EXIT_STATUS", 0)
    require("ML_CUBE_OVERLAP", spec["overlap"])
    require("RUN_STEPS", 200)
    require("SOLUTION_INTERVAL", 10000)
    if Path(info.get("MODEL", "")).name != MODELS[spec["model"]]:
        raise ValueError(f"{spec['cell']}: missing/mismatched MODEL")
    provider = spec["provider"]
    hetero = bool(spec["gpu_solver_ranks"])
    cases = {"aix": "aix-collective" if spec["comm"] == "coll" else "aix-" + spec["comm"],
             "phydll": "phydll-blocking" if spec["wait"] == "default" else "phydll-readiness",
             "smartsim": "smartsim"}
    if hetero:
        require("PROVIDER", provider.upper())
        require("HET_CPU_TASKS", spec["cpu_solver_ranks"])
        require("HET_GPU_TASKS", spec["gpu_solver_ranks"])
        require("HET_GPUS", spec["gpu_count"])
    else:
        require("CASE", cases[provider])
        require("SOLVER_RANKS", spec["solver_ranks"])
        require("DL_RANKS", spec["gpu_count"] if provider == "phydll" else 0)
    if "PROVIDER" in info:
        require("PROVIDER", provider.upper())
    batch = ""
    if provider == "aix":
        require("AIX_COMM", "collective" if spec["comm"] == "coll" else "pipelined")
        credits = {"coll": "unset", "p2p1": "1", "p2pfull": str(spec["solver_ranks"])}[spec["comm"]]
        require("AIX_CREDITS", credits)
        layout = info.get("AIX_LAYOUT_LIB", "")
        if spec["layout"] == "default":
            require("AIX_LAYOUT_LIB", "default")
        elif Path(layout).name != "INSTALL-" + spec["layout"].upper():
            raise ValueError(f"{spec['cell']}: mismatched AIX layout {layout!r}")
        batch = 50000 if spec["model"] == "transformer" else 1000000
        require("MAIA_AIX_BATCHSIZE", batch)
    elif provider == "phydll":
        if hetero:
            require("PHYDLL_WAIT", spec["wait"])
            require("NP_DL", spec["gpu_count"])
        batch = 2048 if spec["model"] == "transformer" else 1000000
        require("MAIA_PHYDLL_BATCHSIZE", batch)
    else:
        require("MAIA_SMARTSIM_NUM_GPUS", spec["gpu_count"])
    log = (directory / "solver.log").read_text()
    calls = STEP_RE.findall(log)
    if tuple(int(step) for step, _ in calls) != STEPS:
        raise ValueError(f"{spec['cell']}: expected exactly steps {STEPS}, got {calls}")
    if log.count("ML_STEP_TIMING step=") != len(calls):
        raise ValueError(f"{spec['cell']}: malformed inference timing")
    values = [float(value) for _, value in calls]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{spec['cell']}: invalid latency")
    workloads = WORKLOAD_RE.findall(log)
    if not workloads or len(set(workloads)) != 1:
        raise ValueError(f"{spec['cell']}: missing/conflicting critical workload log")
    workload = dict(zip(WORKLOAD_FIELDS, map(int, workloads[0])))
    if workload["workload_solver_ranks"] != spec["solver_ranks"] or any(v <= 0 for v in workload.values()):
        raise ValueError(f"{spec['cell']}: invalid workload {workload}")
    wall = info.get("BENCHMARK_SOLVER_WALL_SECONDS", "")
    if not wall or not math.isfinite(float(wall)) or float(wall) <= 0:
        raise ValueError(f"{spec['cell']}: missing/invalid solver wall time")
    measured = values[2:]
    batch_records = [dict(zip(("device", "batch_dim", "batchsize", "num_batches",
                               "used_mib", "total_mib", "call"), values))
                     for values in AIXMEM_RE.findall(log)]
    row = {**spec, **{k: v for k, v in selection.items() if k != "lineage"},
            **workload, "run_dir": directory.relative_to(root).as_posix(),
           "solver_exit_status": int(info["SOLVER_EXIT_STATUS"]),
           "exit_status": int(info["EXIT_STATUS"]),
           "batch_size": batch, "aix_credits": info.get("AIX_CREDITS", "") if provider == "aix" else "",
           "aix_gpu_batch_records": json.dumps(batch_records, separators=(",", ":")) if batch_records else "",
           "wall_seconds": wall, "warmup_ms": ";".join(v for _, v in calls[:2]),
           "measured_ms": ";".join(v for _, v in calls[2:]),
           "n_inferences": 7, "n_measured": 5,
           "measured_mean_ms": statistics.mean(measured),
           "measured_median_ms": statistics.median(measured),
           "measured_stdev_ms": statistics.stdev(measured),
           "measured_min_ms": min(measured), "measured_max_ms": max(measured)}
    inferences = [dict(cell=spec["cell"], jobid=selection["jobid"], call=i + 1,
                       phase="warmup" if i < 2 else "measured", step=int(step), latency_ms=value)
                  for i, (step, value) in enumerate(calls)]
    return row, inferences


def extract_resources(root: Path, directory: Path, run: dict) -> tuple[list, list]:
    """Keep every raw per-host/device peak and baseline; audit expected coverage.

    Configured devices are indexed 0..N-1 on the ML host by these samplers.
    Other observed devices/hosts are retained but do not count toward coverage.
    Missing/zero/invalid peaks never become zero-valued successful samples.
    """
    info = read_kv(directory / "run.info")
    cpu_host = info.get("BATCH_HOST", "")
    ml_host = info.get("HET_GPU_NODELIST", "") if run["gpu_solver_ranks"] else cpu_host
    # This campaign uses one exact hostname for each heterogeneous component.
    valid_host = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
    warnings = []
    for role, host in (("solver", cpu_host), ("ml", ml_host)):
        if not host or host == "unknown" or not valid_host.fullmatch(host):
            warnings.append(f"{run['cell']} job {run['jobid']}: unresolved {role} host {host!r}")
            if role == "solver":
                cpu_host = "<missing_solver_host>"
            else:
                ml_host = "<missing_ml_host>"
    nodes, gpus, baselines = {}, {}, {}
    for key, value in info.items():
        if key.startswith("NODE_RAM_PEAK_MIB_"):
            nodes[key.removeprefix("NODE_RAM_PEAK_MIB_")] = value
        for prefix, target in (("GPU_VRAM_PEAK_MIB_", gpus), ("GPU_VRAM_BASE_MIB_", baselines)):
            if key.startswith(prefix):
                host, sep, device = key.removeprefix(prefix).rpartition("_gpu")
                if not sep or not device.isdigit():
                    warnings.append(f"{run['cell']}: malformed telemetry key {key}")
                    continue
                target[(host, int(device))] = value
    expected_nodes = {cpu_host, ml_host}
    expected_gpus = {(ml_host, i) for i in range(run["gpu_count"])}
    rows = []
    for metric, observed, expected in (("node_ram", nodes, expected_nodes),
                                       ("gpu_vram", gpus, expected_gpus)):
        identities = set(observed) | expected
        if metric == "gpu_vram":
            identities |= set(baselines)
        for identity in sorted(identities):
            host, device = identity if metric == "gpu_vram" else (identity, "")
            raw = observed.get(identity, "")
            base = baselines.get(identity, "") if metric == "gpu_vram" else ""
            def number(value):
                try:
                    result = float(value)
                    return result if math.isfinite(result) and result >= 0 else None
                except ValueError:
                    return None
            peak, baseline = number(raw), number(base)
            status = "missing" if raw == "" else "invalid" if peak is None else "zero" if peak == 0 else "ok"
            role = "shared" if host == cpu_host == ml_host else "solver" if host == cpu_host else "ml" if host == ml_host else "other"
            is_expected = identity in expected
            if status != "ok" and is_expected:
                warnings.append(f"{run['cell']} job {run['jobid']}: {status} {metric} host={host} device={device}")
            if base and baseline is None:
                warnings.append(f"{run['cell']}: invalid GPU baseline host={host} device={device}")
            rows.append(dict(cell=run["cell"], jobid=run["jobid"], run_dir=run["run_dir"],
                             metric=metric, scope="whole_node" if metric == "node_ram" else "device_wide",
                             host=host, host_role=role, device=device, expected=is_expected,
                             status=status, peak_mib=raw, peak_gib=peak / 1024 if peak is not None else "",
                             baseline_mib=base, baseline_gib=baseline / 1024 if baseline is not None else "",
                             source="run.info"))
    return rows, warnings


def summarize(runs: list[dict]) -> list[dict]:
    """Aggregate three independent rep means, never pool the 15 measured calls."""
    groups = defaultdict(list)
    for run in runs:
        groups[tuple(run[k] for k in CONFIG_FIELDS)].append(run)
    rows = []
    for identity, reps in sorted(groups.items()):
        if sorted(r["rep"] for r in reps) != [1, 2, 3]:
            raise ValueError(f"{identity}: expected reps 1,2,3")
        means = [r["measured_mean_ms"] for r in reps]
        medians = [r["measured_median_ms"] for r in reps]
        rows.append(dict(zip(CONFIG_FIELDS, identity), n_reps=len(reps),
                         mean_ms=statistics.mean(means), sample_sd_ms=statistics.stdev(means),
                         min_ms=min(means), max_ms=max(means),
                         mean_rep_median_ms=statistics.mean(medians),
                         sample_sd_rep_median_ms=statistics.stdev(medians),
                         mean_wall_seconds=statistics.mean(float(r["wall_seconds"]) for r in reps)))
    return rows


def consolidate(project_root: Path, out_dir: Path) -> dict:
    """Write CSVs only after fatal validation passes; always write an audit report.

    Returns the report (status 'ok' or 'error'); main maps that status to exit code.
    There are no timestamps or machine-clock-dependent fields in the outputs.
    """
    root = Path(project_root).resolve()
    out = Path(out_dir)
    out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=True)
    matrix = expected_matrix()
    report = dict(status="error", expected=dict(runs=288, inferences=2016, configs=96),
                  selected=0, errors=[], warnings=[], sources={}, provenance={})
    runs, inferences, resources, summary = [], [], [], []
    try:
        selections = select_cells(root, matrix, report["sources"])
        report["selected"] = len(selections)
        report["provenance"] = selections
        index = index_runs(root)
        workloads = {}
        for cell, spec in matrix.items():
            selection = selections[cell]
            candidates = index.get((cell, selection["jobid"]), [])
            if len(candidates) != 1:
                report["errors"].append(f"{cell} job {selection['jobid']}: expected exactly one designated run, found {len(candidates)}: {[str(p.relative_to(root)) for p in candidates]}")
                continue
            directory = candidates[0]
            try:
                run, calls = parse_selected_run(root, directory, spec, selection)
                key = (spec["placement"], spec["model"])
                workload = tuple(run[k] for k in WORKLOAD_FIELDS)
                if key in workloads and workloads[key] != workload:
                    raise ValueError(f"{cell}: workload differs within placement/model: {workload} != {workloads[key]}")
                workloads[key] = workload
                memory, warnings = extract_resources(root, directory, run)
                runs.append(run)
                inferences.extend(calls)
                resources.extend(memory)
                report["warnings"].extend(warnings)
                report["provenance"][cell]["run_dir"] = run["run_dir"]
                report["provenance"][cell]["artifacts"] = {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("run.info", "solver.log")}
            except (OSError, ValueError, KeyError) as exc:
                report["errors"].append(f"{cell}: {exc}")
        if not report["errors"]:
            summary = summarize(runs)
            if (len(runs), len(inferences), len(summary)) != (288, 2016, 96):
                raise ValueError("validated output cardinality differs from expected matrix")
            report["status"] = "ok"
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(str(exc))
    report["validated"] = dict(runs=len(runs), inferences=len(inferences), configs=len(summary))
    report["selection_counts"] = dict(Counter(r["selection"] for r in runs))
    report["resource_coverage"] = {
        metric: dict(expected=sum(r["expected"] for r in resources if r["metric"] == metric),
                     valid=sum(r["expected"] and r["status"] == "ok" for r in resources if r["metric"] == metric),
                     statuses=dict(Counter(r["status"] for r in resources if r["metric"] == metric and r["expected"])))
        for metric in ("node_ram", "gpu_vram")}
    report["measurement_notes"] = {
        "latency_unit": "ms; first two calls warmup, last five measured",
        "summary_unit": "one per-rep measured mean; sample SD across three reps",
        "memory_units": "MiB; GiB = MiB / 1024; whole-node RAM and device-wide VRAM",
        "gpu_coverage": "configured gpu_count on ML host, indices 0..gpu_count-1; not SLURM_JOB_GPUS",
        "batch_counts": "aix_gpu_batch_records preserves logged per-device/call AIX_GPU_MEM fields when available; blank means not logged",
        "selection_order": ["original", "smartsim_retry", "quota_retry", "shutdown_retry", "final_retry"],
    }
    if report["status"] == "ok":
        for name, rows in (("runs", runs), ("inferences", inferences),
                           ("resources", resources), ("summary", summary)):
            with (out / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    else:
        # Do not leave stale successful datasets beside a failed audit.
        for name in ("runs", "inferences", "resources", "summary"):
            (out / (name + ".csv")).unlink(missing_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("results/cmi_bench_consolidated"))
    args = parser.parse_args(argv)
    report = consolidate(args.project_root, args.out_dir)
    out = args.out_dir if args.out_dir.is_absolute() else args.project_root / args.out_dir
    print(f"{report['status']}: {report['validated']}; {len(report['warnings'])} warnings, {len(report['errors'])} errors; output: {out.resolve()}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
