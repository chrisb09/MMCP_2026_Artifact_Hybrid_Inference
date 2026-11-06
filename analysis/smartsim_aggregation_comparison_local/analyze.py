#!/usr/bin/env python3
"""Reproduce the SmartSim aggregation versus canonical shared comparison."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate_smartsim_aggregation_reruns as validator

MANIFESTS = [ROOT / "results/smartsim_aggregation_reruns" / name / "manifest.csv" for name in (
    "campaign_20261103T181527Z_7tpvSF", "campaign_20261103T181527Z_VGwZo0")]
CANON = ROOT / "results/cmi_bench_consolidated"
SCRATCH = ROOT / "scratch/smartsim_aggregation_reruns"
PLACEMENTS = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
MODELS = ("transformer", "mlp")
STEPS = (15, 44, 73, 102, 131, 160, 189)
PARAMS = {"transformer": ("7000", "7000", "5"), "mlp": ("45000", "45000", "1")}
T95_DF2 = 4.302652729911275


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(path):
    text = path.read_text(errors="strict")
    found = re.findall(r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)", text)
    require(text.count("ML_STEP_TIMING step=") == len(found) == 7, f"bad timing records: {path}")
    require(tuple(int(s) for s, _ in found) == STEPS, f"bad timing steps: {path}")
    values = [float(v) for _, v in found]
    require(all(math.isfinite(v) and v > 0 for v in values), f"bad timing values: {path}")
    return values


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in (*MANIFESTS, CANON / "runs.csv", CANON / "inferences.csv", CANON / "report.json")}
    canonical_rows = csv_rows(CANON / "runs.csv")
    canonical = {r["cell"]: r for r in canonical_rows}
    require(len(canonical) == len(canonical_rows), "duplicate canonical cell")
    inferences = csv_rows(CANON / "inferences.csv")
    report = json.loads((CANON / "report.json").read_text())
    require(not report["errors"], "canonical report contains errors")
    manifest = sum((csv_rows(p) for p in MANIFESTS), [])
    expected = {f"smartsim-{p}-agg-{m}-r{r}" for m in MODELS for p in PLACEMENTS for r in (1, 2, 3)}
    require(len(manifest) == 24 and {r["cell"] for r in manifest} == expected, "manifest coverage or duplicate error")
    require(len({r["jobid"] for r in manifest}) == 24, "aggregation job IDs not unique")
    audit = []
    values = {}
    baseline_jobs = set()
    for row in manifest:
        model, placement, rep = row["model"], row["placement"], row["rep"]
        require(row["cell"] == f"smartsim-{placement}-agg-{model}-r{rep}" and rep in ("1", "2", "3"), "manifest identity mismatch")
        require(row["dry_run"] == "0" and tuple(row[k] for k in ("batch_size", "min_batch_size", "timeout_ms")) == PARAMS[model], "unexpected aggregation parameters")
        validator.validate(row, canonical, SCRATCH)
        agg_dir = validator.find_run(SCRATCH, row["cell"], row["jobid"])
        base_cell = row["cell"].replace("-agg-", "-shared-")
        base = canonical[base_cell]
        require(base["provider"] == "smartsim" and base["layout"] == "shared" and base["placement"] == placement and base["model"] == model and base["rep"] == rep, f"canonical identity: {base_cell}")
        require(base["jobid"] not in baseline_jobs, "baseline job IDs not independent")
        baseline_jobs.add(base["jobid"])
        base_dir = ROOT / base["run_dir"]
        prov = report["provenance"][base_cell]
        require(prov["jobid"] == base["jobid"] and prov["run_dir"] == base["run_dir"], f"canonical provenance mismatch: {base_cell}")
        for filename in ("run.info", "solver.log"):
            require(digest(base_dir / filename) == prov["artifacts"][filename], f"canonical hash mismatch: {base_cell}/{filename}")
        for directory, cell, job in ((base_dir, base_cell, base["jobid"]), (agg_dir, row["cell"], row["jobid"])):
            info = validator.info(directory / "run.info")
            require(info.get("CELL") == cell and info.get("REP") == rep, f"run identity: {directory}")
            require(info.get("MAIA_ML_STEP_TIMING_SYNC") == "1" and info.get("MAIA_ML_STEP_TIMING_PRINT") == "1", f"timing flags: {directory}")
            require(info.get("SOLVER_EXIT_STATUS") == "0" and info.get("EXIT_STATUS") == "0", f"exit status: {directory}")
            for filename in ("run.info", "solver.log", "config_smartsim.toml", "properties.toml"):
                path = directory / filename
                inputs[str(path.relative_to(ROOT))] = digest(path)
        bconfig = tomllib.loads((base_dir / "config_smartsim.toml").read_text())
        aconfig = tomllib.loads((agg_dir / "config_smartsim.toml").read_text())
        for key in ("batch_size", "min_batch_size", "min_batch_timeout"):
            require(bconfig["library"].pop(key) == 0, f"baseline aggregation enabled: {base_cell}")
            aconfig["library"].pop(key)
        require(bconfig == aconfig, f"non-batching config differs: {base_cell}")
        bvals, avals = calls(base_dir / "solver.log"), calls(agg_dir / "solver.log")
        cinf = sorted((r for r in inferences if r["cell"] == base_cell), key=lambda r: int(r["call"]))
        require(len(cinf) == 7 and all(int(r["call"]) == i + 1 and int(r["step"]) == STEPS[i] and r["phase"] == ("warmup" if i < 2 else "measured") and r["jobid"] == base["jobid"] and float(r["latency_ms"]) == bvals[i] for i, r in enumerate(cinf)), f"canonical inference mismatch: {base_cell}")
        require(base["n_measured"] == "5" and math.isclose(float(base["measured_mean_ms"]), statistics.mean(bvals[2:]), rel_tol=1e-12), f"canonical mean mismatch: {base_cell}")
        for kind, cell, job, directory, vals in (("baseline", base_cell, base["jobid"], base_dir, bvals), ("aggregation", row["cell"], row["jobid"], agg_dir, avals)):
            mean = statistics.mean(vals[2:])
            values.setdefault((model, placement, kind), []).append(mean)
            audit.append(dict(model=model, placement=placement, variant=kind, rep=rep, cell=cell, jobid=job, run_dir=str(directory.relative_to(ROOT)), all_calls_ms=";".join(map(str, vals)), measured_mean_ms=f"{mean:.6f}"))
    require(not (baseline_jobs & {r["jobid"] for r in manifest}), "baseline and aggregation jobs overlap")
    summary = []
    for model in MODELS:
        for placement in PLACEMENTS:
            b, a = (values[model, placement, kind] for kind in ("baseline", "aggregation"))
            require(len(b) == len(a) == 3, "wrong repetition count")
            bm, am = statistics.mean(b), statistics.mean(a)
            bc, ac = (T95_DF2 * statistics.stdev(v) / math.sqrt(3) for v in (b, a))
            summary.append(dict(model=model, placement=placement, n=3, baseline_mean_ms=f"{bm:.3f}", baseline_ci95_half_ms=f"{bc:.3f}", aggregation_mean_ms=f"{am:.3f}", aggregation_ci95_half_ms=f"{ac:.3f}", change_pct=f"{100 * (am / bm - 1):+.3f}"))
    write_csv(OUT / "runs.csv", audit)
    write_csv(OUT / "summary.csv", summary)
    (OUT / "inputs.sha256.json").write_text(json.dumps(dict(sorted(inputs.items())), indent=2) + "\n")
    print("| Model | Placement | Shared mean ± 95% CI (ms) | Aggregation mean ± 95% CI (ms) | Change |")
    print("|---|---|---:|---:|---:|")
    for r in summary:
        print(f"| {r['model']} | {r['placement']} | {r['baseline_mean_ms']} ± {r['baseline_ci95_half_ms']} | {r['aggregation_mean_ms']} ± {r['aggregation_ci95_half_ms']} | {r['change_pct']}% |")
    print("Validated 24 aggregation and 24 canonical runs; n=3 independent runs per cell.")


if __name__ == "__main__":
    main()
