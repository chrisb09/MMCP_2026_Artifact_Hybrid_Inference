#!/usr/bin/env python3
"""Render baseline-only Chapter 6 drafts from the consolidated CSV snapshot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path.home() / "Master-Thesis/gitlab/sourcecode"))
import thesis_style as ts

ts.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import consolidate_cmi_bench as consolidator


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("transformer", "mlp")
HARDWARE = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
HW_LABELS = ("24c/1GPU", "24+24c/1GPU", "96c/4GPUs", "96+96c/4GPUs")
PROVIDERS = ("aix", "phydll", "smartsim")
COLORS = ("#577787", "#C97AA7", "#8064A2")
LABELS = ("AIx (default, collective)", "PhyDLL (default)", "SmartSim (sharedDB)")
T95 = 4.302652729911275
MODEL_LABELS = {"mlp": "MLP (ov4)", "transformer": "Transformer (ov0)"}
CI_NOTE = "Mean of 3 repetition means (5 steady calls each).\nError bars: 95% Student-t CI, df=2."
MEM_NOTE = "Bars: maximum observed peak, not sums.\nNo CI for capacity metrics; no imputation."
WIDTH_MM = 150
HALF_WIDTH_MM = 73.5
AXIS_HEIGHT_MM = 32
SUPPLEMENT = ROOT / "results/cmi_bench_telemetry_retry_validation.json"
SUPPLEMENT_CELL = "aix-24c1g-default-coll-transformer-r2"


def supplemental_peak(selected, resources, report, manifest_path=SUPPLEMENT):
    """Validate the designated repeat without invoking canonical selection."""
    manifest = json.loads(manifest_path.read_text())
    if (manifest["cell"], manifest["original_jobid"], manifest["measurement_jobid"]) != (
            SUPPLEMENT_CELL, "5009522", "5106286") or manifest["status"] != "ok":
        raise ValueError("Unexpected supplemental identity/status")
    original = next(r for r in selected if r["cell"] == SUPPLEMENT_CELL)
    if original["jobid"] != manifest["original_jobid"]:
        raise ValueError("Supplement original job mismatch")
    directory = ROOT / manifest["run_dir"]
    match = consolidator.JOB_RE.search(directory.name)
    if not match or match[1] != manifest["measurement_jobid"]:
        raise ValueError("Supplement directory job mismatch")
    required = {"results/cmi_bench_telemetry_retry_cells.csv",
                "results/telemetry_retry_group_24c1g_transformer_r2_cells.csv"}
    required.update(f"{d}/{name}" for d in (original["run_dir"], manifest["run_dir"])
                    for name in ("run.info", "solver.log"))
    if set(manifest["hashes"]) != required:
        raise ValueError("Incomplete supplemental artifact hashes")
    for path, digest in manifest["hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Supplement hash mismatch: {path}")
    for name, digest in report["provenance"][SUPPLEMENT_CELL]["artifacts"].items():
        if manifest["hashes"][f"{original['run_dir']}/{name}"] != digest:
            raise ValueError("Supplement original artifact differs from canonical report")
    ledger = read_csv(ROOT / "results/cmi_bench_telemetry_retry_cells.csv")
    if len(ledger) != 1 or any(ledger[0][k] != v for k, v in (
            ("cell", SUPPLEMENT_CELL), ("original_jobid", "5009522"), ("retry_jobid", "5106286"))):
        raise ValueError("Supplement ledger mismatch")
    status = ROOT / ledger[0]["group_summary"]
    if status.read_text().splitlines() != [f"{SUPPLEMENT_CELL},0"]:
        raise ValueError("Supplement group status is not successful")
    spec = consolidator.expected_matrix()[SUPPLEMENT_CELL]
    repeat, _ = consolidator.parse_selected_run(ROOT, directory, spec, {"jobid": "5106286"})
    parsed_original, _ = consolidator.parse_selected_run(
        ROOT, ROOT / original["run_dir"], spec, {"jobid": "5009522"})
    for key in (*consolidator.CONFIG_FIELDS, *consolidator.WORKLOAD_FIELDS,
                "rep", "overlap", "batch_size", "gpu_count", "solver_ranks"):
        if str(repeat[key]) != original[key] or repeat[key] != parsed_original[key]:
            raise ValueError(f"Supplement configuration/workload mismatch: {key}")
    original_info = consolidator.read_kv(ROOT / original["run_dir"] / "run.info")
    info = consolidator.read_kv(directory / "run.info")
    for key in ("MAIA", "MODEL", "MAIA_ML_STEP_TIMING_SYNC", "MAIA_ML_STEP_TIMING_PRINT",
                "RUN_STEPS", "SOLUTION_INTERVAL", "AIX_LAYOUT_LIB", "AIX_COMM", "AIX_CREDITS"):
        if not info.get(key) or info[key] != original_info.get(key):
            raise ValueError(f"Supplement raw configuration mismatch: {key}")
    rows, warnings = consolidator.extract_resources(ROOT, directory, repeat)
    peaks = [r for r in rows if r["metric"] == "gpu_vram" and r["expected"]]
    if warnings or len(peaks) != 1:
        raise ValueError("Invalid supplemental resource coverage")
    peak = peaks[0]
    if (peak["scope"] != "device_wide" or peak["host_role"] != "shared"
            or peak["device"] != 0 or peak["status"] != "ok"
            or peak["baseline_gib"] == "" or not np.isfinite(peak["baseline_gib"])):
        raise ValueError("Invalid supplemental device-wide peak/baseline")
    for key, value in manifest["peak"].items():
        if str(peak[key]) != str(value):
            raise ValueError(f"Supplement manifest peak mismatch: {key}")
    missing = [r for r in resources if r["cell"] == SUPPLEMENT_CELL and r["metric"] == "gpu_vram"
               and r["expected"] == "True"]
    if len(missing) != 1 or missing[0]["status"] != "missing" or missing[0]["peak_gib"]:
        raise ValueError("Supplement may only fill the designated missing device")
    if missing[0]["scope"] != peak["scope"] or missing[0]["device"] != str(peak["device"]):
        raise ValueError("Supplement device/scope mismatch")
    return {**peak, "expected": "True", "device": "0", "observation_kind": "supplemental_repeat",
            "source_jobid": "5106286", "measurement_jobid": "5106286", "original_jobid": "5009522"}


def figure_panels(count=1, width=WIDTH_MM):
    # Model pairs share a row; the three memory scopes remain a vertical plate.
    rows, cols = (1, 2) if count == 2 else (count, 1)
    top = 20 if width < WIDTH_MM else 11 if count == 2 else 15 if count > 1 else 10
    height = top + 12 + rows * AXIS_HEIGHT_MM + (rows - 1) * 13
    return plt.subplots(rows, cols, figsize=(width / 25.4, height / 25.4), squeeze=False)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline(row):
    return ((row["provider"] == "aix" and row["layout"] == "default" and row["comm"] == "coll")
            or (row["provider"] == "phydll" and row["wait"] == "default")
            or (row["provider"] == "smartsim" and row["layout"] == "shared"))


def collect(data_dir, supplement=SUPPLEMENT):
    runs = read_csv(data_dir / "runs.csv")
    resources = read_csv(data_dir / "resources.csv")
    report = json.loads((data_dir / "report.json").read_text())
    if report["errors"] or report["status"] != "ok":
        raise ValueError("Consolidation report is not clean")
    selected = [r for r in runs if baseline(r)]
    if len(selected) != 72 or len({r["cell"] for r in selected}) != 72:
        raise ValueError("Expected exactly 72 unique baseline runs")
    by_cell = {r["cell"]: r for r in selected}
    expected = [r for r in resources if r["cell"] in by_cell and r["expected"] == "True"]
    peak = supplemental_peak(selected, resources, report, supplement) if supplement and supplement.exists() else None
    # One effective row per expected resource; canonical rows remain untouched.
    plot_resources = [{**r, "observation_kind": "original", "measurement_jobid": r["jobid"]}
                      for r in expected]
    if peak:
        for i, r in enumerate(plot_resources):
            if r["cell"] == peak["cell"] and r["metric"] == "gpu_vram" and r["device"] == peak["device"]:
                plot_resources[i] = {**r, **{f"original_{k}": r[k] for k in ("status", "peak_gib", "host", "run_dir")},
                                     **peak, "jobid": r["jobid"]}
    devices = []
    for r in expected:
        if r["metric"] == "gpu_vram":
            run = by_cell[r["cell"]]
            devices.append({**{k: run[k] for k in ("model", "placement", "provider", "rep")}, **r,
                            "observation_kind": "original", "source_jobid": r["jobid"],
                            "measurement_jobid": r["jobid"], "original_jobid": r["jobid"]})
    if peak:
        devices.append({**{k: by_cell[peak["cell"]][k] for k in ("model", "placement", "provider", "rep")}, **peak})
    latency, memory = [], []
    for model in MODELS:
        for hw in HARDWARE:
            for provider in PROVIDERS:
                group = sorted((r for r in selected if (r["model"], r["placement"], r["provider"])
                                == (model, hw, provider)), key=lambda r: int(r["rep"]))
                if [r["rep"] for r in group] != ["1", "2", "3"]:
                    raise ValueError(f"Incomplete repetitions: {model}, {hw}, {provider}")
                for r in group:
                    calls = np.array([float(v) for v in r["measured_ms"].split(";")])
                    if (r["n_measured"] != "5" or len(calls) != 5 or not np.isfinite(calls).all()
                            or not np.isclose(calls.mean(), float(r["measured_mean_ms"]))
                            or r["exit_status"] != "0" or r["solver_exit_status"] != "0"):
                        raise ValueError(f"Invalid steady measurements: {r['cell']}")
                reps = np.array([float(r["measured_mean_ms"]) / 1000 for r in group])
                mean, sd = float(reps.mean()), float(reps.std(ddof=1))
                half = T95 * sd / np.sqrt(3)
                key = dict(model=model, placement=hw, provider=provider)
                latency.append({**key, "mean_s": mean, "sd_s": sd, "ci_low_s": mean-half,
                                "ci_high_s": mean+half, "n_reps": 3,
                                **{f"rep{i+1}_s": v for i, v in enumerate(reps)}})
                for metric, role in (("node_ram", "shared" if "p" not in hw else "solver"),
                                     ("node_ram", "ml"), ("gpu_vram", "ml" if "p" in hw else "shared")):
                    values, effective, observed, required, repeats = [], [], 0, 0, 0
                    for run in group:
                        rows = [r for r in expected if r["cell"] == run["cell"]
                                and r["metric"] == metric and r["host_role"] == role]
                        n_expected = int(run["gpu_count"]) if metric == "gpu_vram" else int(role != "ml" or "p" in hw)
                        if len(rows) != n_expected:
                            raise ValueError(f"Unexpected resource coverage: {run['cell']} {metric} {role}")
                        peaks = [float(r["peak_gib"]) for r in rows if r["status"] == "ok" and r["peak_gib"]]
                        if not np.isfinite(peaks).all():
                            raise ValueError("Nonfinite resource peak")
                        values.append(max(peaks) if peaks else "")
                        effective_rows = [r for r in plot_resources if r["cell"] == run["cell"]
                                          and r["metric"] == metric and r["host_role"] == role]
                        effective_peaks = [float(r["peak_gib"]) for r in effective_rows if r["status"] == "ok" and r["peak_gib"]]
                        effective.append(max(effective_peaks) if effective_peaks else "")
                        repeats += sum(r["observation_kind"] == "supplemental_repeat" for r in effective_rows)
                        observed += len(effective_peaks)
                        required += n_expected
                    present = [v for v in values if v != ""]
                    memory.append({**key, "metric": metric, "host_role": role,
                                    "max_gib": max(v for v in effective if v != "") if any(v != "" for v in effective) else "",
                                    **{f"rep{i+1}_gib": v for i, v in enumerate(values)},
                                    **{f"effective_rep{i+1}_gib": v for i, v in enumerate(effective)},
                                    "supplemental_observations": repeats,
                                    "supplemental_jobid": peak["measurement_jobid"] if repeats else "",
                                    "original_missing_jobid": peak["original_jobid"] if repeats else "",
                                   "n_reps_observed": len(present), "n_reps_expected": 3 if required else 0,
                                   "observations": observed, "expected_observations": required})
    return runs, selected, resources, report, latency, memory, devices


def style_axis(ax, title, ylabel):
    if len(ax.figure.axes) > 1:
        paired = ax.get_subplotspec().get_gridspec().ncols == 2
        ax.set_title(title, loc="left", fontsize=10, pad=3 if paired else 9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(4), [label.replace("/", "/\n") for label in HW_LABELS])
    ax.tick_params(axis="x", labelsize=8, length=0, pad=7)
    ax.tick_params(axis="y", width=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.5)
    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=.45, color=".83")
    ax.set_xlim(-.55, 3.55)


def finish(fig, out, name):
    width, height = fig.get_size_inches() * 25.4
    paired = fig.axes[0].get_subplotspec().get_gridspec().ncols == 2
    narrow = width < WIDTH_MM
    handles = [Patch(facecolor=c, edgecolor=".25", linewidth=.5, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=1 if narrow else 3, frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=1)
    top = 20 if narrow else 11 if paired else 15 if len(fig.axes) > 1 else 10
    if paired or narrow:
        # Each 73.5 mm model slot has its own ylabel margin, with a 3 mm gap.
        fig.subplots_adjust(top=1-top/height, bottom=12/height,
                            left=13/width, right=1-2/width, wspace=18/58.5)
    else:
        fig.subplots_adjust(top=1-top/height, bottom=12/height,
                            left=.105, right=.985, hspace=13/AXIS_HEIGHT_MM)
    for ax in fig.axes:
        if not ax.get_subplotspec().is_last_row():
            ax.tick_params(axis="x", labelbottom=False)
    ts.save(fig, out / name)


def latency_panel(ax, rows, model):
    narrow = ax.figure.get_figwidth() * 25.4 / ax.get_subplotspec().get_gridspec().ncols < 100
    subset = [r for r in rows if r["model"] == model]
    for p, (provider, color) in enumerate(zip(PROVIDERS, COLORS)):
        for h, hw in enumerate(HARDWARE):
            row = next(r for r in subset if r["provider"] == provider and r["placement"] == hw)
            x, y = h + (p-1)*.24, row["mean_s"]
            ax.bar(x, y, width=.21, color=color, edgecolor=".25", linewidth=.5)
            ax.errorbar(x, y, yerr=row["ci_high_s"]-y, fmt="none", color=".18", capsize=2, linewidth=.7)
            ax.annotate(f"{y:.2f}", (x, row["ci_high_s"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=6.5, color=".35",
                        rotation=90 if narrow else 0)
    low = min(r["ci_low_s"] for r in subset)
    high = max(r["ci_high_s"] for r in subset)
    if low < 0:
        warnings.warn(f"{model}: negative CI lower bound {low:.6g} s retained")
    ax.set_ylim(min(0, low - .025*high) if low < 0 else 0, high*(1.35 if narrow else 1.17))
    style_axis(ax, MODEL_LABELS[model], "Steady-call latency (s)")


def memory_panel(ax, rows, model, panel):
    narrow = ax.figure.get_figwidth() * 25.4 / ax.get_subplotspec().get_gridspec().ncols < 100
    titles = ("Shared / CPU-side RAM", "GPU-side host RAM (ML host)", "Maximum per-GPU VRAM")
    for p, (provider, color) in enumerate(zip(PROVIDERS, COLORS)):
        for h, hw in enumerate(HARDWARE):
            role = ("shared" if "p" not in hw else "solver") if panel == 0 else "ml"
            if panel == 2:
                role = "ml" if "p" in hw else "shared"
            metric = "gpu_vram" if panel == 2 else "node_ram"
            row = next(r for r in rows if (r["model"], r["placement"], r["provider"], r["metric"], r["host_role"])
                       == (model, hw, provider, metric, role))
            x, y = h + (p-1)*.24, row["max_gib"]
            if y == "":
                ax.text(x, .025, "N/A" if not row["expected_observations"] else "missing",
                        transform=ax.get_xaxis_transform(), ha="center", fontsize=6, color=".45", rotation=90)
                continue
            ax.bar(x, y, width=.21, color=color, edgecolor=".25", linewidth=.5)
            if row.get("supplemental_observations") or row["n_reps_observed"] < 3:
                ax.annotate("*", (x, y), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=7, color=".25")
    ax.set_ylim(bottom=0)
    ax.margins(y=.19)
    if narrow:
        ax.set_ylim(top=ax.dataLim.ymax * 1.4)
    title = MODEL_LABELS[model] + ("\n" if narrow else ": ") + titles[panel]
    paired = ax.get_subplotspec().get_gridspec().ncols == 2
    if paired:
        style_axis(ax, MODEL_LABELS[model], "Max. per-GPU VRAM (GiB)")
    else:
        style_axis(ax, title, "Device-wide GiB" if panel == 2 else "Whole-node GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/cmi_bench_consolidated")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/ch6_gpu_draft")
    args = parser.parse_args()
    runs, selected, resources, report, latency, memory, devices = collect(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = figure_panels(2)
    for ax, model in zip(axes.flat, MODELS):
        latency_panel(ax, latency, model)
    finish(fig, args.out_dir, "latency_baselines")
    for model in MODELS:
        fig, axes = figure_panels()
        latency_panel(axes[0, 0], latency, model)
        finish(fig, args.out_dir, f"{model}_latency")
        fig, axes = figure_panels(width=HALF_WIDTH_MM)
        latency_panel(axes[0, 0], latency, model)
        finish(fig, args.out_dir, f"{model}_latency_halfwidth")
        fig, axes = figure_panels(3)
        for panel, ax in enumerate(axes.flat):
            memory_panel(ax, memory, model, panel)
        finish(fig, args.out_dir, f"{model}_memory")
        for panel, kind in enumerate(("cpu_ram", "ml_host_ram", "vram")):
            fig, axes = figure_panels()
            memory_panel(axes[0, 0], memory, model, panel)
            finish(fig, args.out_dir, f"{model}_{kind}")
            fig, axes = figure_panels(width=HALF_WIDTH_MM)
            memory_panel(axes[0, 0], memory, model, panel)
            finish(fig, args.out_dir, f"{model}_{kind}_halfwidth")
    fig, axes = figure_panels(2)
    for ax, model in zip(axes.flat, MODELS):
        memory_panel(ax, memory, model, 2)
    finish(fig, args.out_dir, "gpu_device_peaks")
    write_csv(args.out_dir / "plot_values.csv", latency)
    write_csv(args.out_dir / "memory_values.csv", memory)
    write_csv(args.out_dir / "device_peaks.csv", devices)
    metadata = {
        "inputs": {name: {"path": str(args.data_dir / name),
                          "sha256": hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()}
                   for name in ("runs.csv", "resources.csv", "report.json")},
        "counts": {"input_runs": len(runs), "baseline_runs": len(selected), "latency_bars": len(latency),
                   "steady_calls": 5*len(selected), "input_resource_rows": len(resources),
                   "memory_audit_rows": len(memory), "memory_applicable_bars": sum(r["max_gib"] != "" for r in memory),
                    "expected_device_observations": sum(r["observation_kind"] == "original" for r in devices),
                    "supplemental_device_observations": sum(r["observation_kind"] == "supplemental_repeat" for r in devices),
                    "original_observed_device_peaks": sum(bool(r["peak_gib"]) and r["observation_kind"] == "original" for r in devices),
                   "observed_device_peaks": sum(bool(r["peak_gib"]) for r in devices),
                   "missing_device_peaks": sum(not r["peak_gib"] for r in devices),
                     "figure_triplets": 20},
        "figure_dimensions_mm": {"width": WIDTH_MM, "single_height": 54,
                                  "two_panel_height": 55, "three_panel_height": 149,
                                  "paired_layout": "1x2", "model_slot_width": HALF_WIDTH_MM,
                                  "halfwidth": HALF_WIDTH_MM, "halfwidth_height": 64,
                                 "axes_height": AXIS_HEIGHT_MM},
        "student_t_critical_df2": T95, "consolidation_warnings": report["warnings"],
        "supplement": {"path": str(SUPPLEMENT), "sha256": hashlib.sha256(SUPPLEMENT.read_bytes()).hexdigest()} if SUPPLEMENT.exists() else None,
        "workloads": [{k: r[k] for k in ("cell", "samples_aggregate", "samples_per_rank", "solver_ranks", "overlap", "batch_size")}
                      for r in selected],
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# Chapter 6 GPU Drafts\n\nBaseline comparisons authored at thesis page width (150 mm).\n\n"
        "## Figures\n\n"
        "| Draft | Preview | Vector |\n| --- | --- | --- |\n"
        "| Combined latency | [PNG](latency_baselines.png) | [PDF](latency_baselines.pdf) |\n"
        "| Transformer latency | [PNG](transformer_latency.png) | [PDF](transformer_latency.pdf) |\n"
        "| MLP latency | [PNG](mlp_latency.png) | [PDF](mlp_latency.pdf) |\n"
        "| Transformer memory | [PNG](transformer_memory.png) | [PDF](transformer_memory.pdf) |\n"
        "| MLP memory | [PNG](mlp_memory.png) | [PDF](mlp_memory.pdf) |\n"
        "| Per-GPU observed maxima | [PNG](gpu_device_peaks.png) | [PDF](gpu_device_peaks.pdf) |\n\n"
        "Every table entry also has a same-stem PGF export. Individual memory panels are available as "
        "`{transformer,mlp}_{cpu_ram,ml_host_ram,vram}.{png,pdf,pgf}`. "
         "The 12 original figures remain 150 mm wide: standalones are 54 mm high, two-model plates "
         "are one horizontal row (1x2), 55 mm high, and three-scope single-model memory plates "
         "remain stacked, 149 mm high. Paired model slots are 73.5 mm wide with a 3 mm gap. "
         "Eight additional `{transformer,mlp}_{latency,cpu_ram,ml_host_ram,vram}_halfwidth.{png,pdf,pgf}` "
         "exports are authored at 73.5 x 64 mm, with a vertical legend, for independent subfigures. "
         "Each plotting axis is 32 mm high, roughly half the "
        "previous physical y-axis height. Include at authored width, without further shrinking. "
        "Overview model labels identify Transformer (ov0) and MLP (ov4). Independent panels have no "
        "embedded titles or subcaptions in any format; supply model/overlap and memory scope in LaTeX "
        "subcaptions. cpu_ram is shared/CPU-side RAM, ml_host_ram is GPU-side ML-host RAM, and vram "
        "is maximum per-GPU VRAM. Statistical and coverage explanations are caption/README-only.\n\n"
        "## Font Style\n\n"
        "All formats use thesis_style's pdflatex/Latin Modern typography. Axis labels are 9 pt, "
        "ticks 8 pt, legends 7.5 pt and latency values 6.5 pt; overview headers are 9--10 pt. "
        "Removing explanatory footers and independent titles trims canvas margins only, without "
        "enlarging fonts or stretching the 32 mm plotting axes. Paired overviews use an 11 mm "
        "top band and 3 pt model-header padding; measured legend/header bounds must not overlap "
        "and their gap must be at most 3 mm. The paired VRAM overview identifies maximum per-GPU "
        "VRAM in its y-axis labels; the metric remains device-wide. Narrow three-row legends "
        "retain their existing 20 mm top band.\n\n"
        "## Methodology\n\n"
        "Baseline only: AIx default/coll, PhyDLL default, SmartSim sharedDB; 72 runs, 24 latency bars. "
        "Hardware groups are placements, not a fixed-work strong-scaling experiment.\n\n"
        + CI_NOTE + " SD uses ddof=1; CI = mean +/- 4.302652729911275 * SD / sqrt(3). "
        "The 15 calls per bar are NOT 15 independent observations. Negative CI bounds are retained and warned.\n\n"
        + MEM_NOTE + " RAM is whole-node usage, not CPU-process RSS; GPU-side RAM is the separate ML host "
        "in heterogeneous placements and N/A for shared-host placements. VRAM is device-wide usage, "
        "maximized across expected ML-host devices within each repetition, then across repetitions, never summed. "
        "Background usage is not subtracted. No missing observations are zero-filled. "
        "AIx Transformer 24c/1GPU repetition 2 VRAM is missing "
        "(aix-24c1g-default-coll-transformer-r2, job 5009522): historically n=2 original observations; "
        "the original rep2_gib remains blank in the audit. A validated supplemental repeat, job 5106286, "
        "now contributes only its sampled device-wide peak to the maximum, marked with an availability '*'. "
        "That bar has 2 original observations + 1 supplemental repeat, "
        "not three original repetitions. effective_rep2_gib is plot-derived, not a canonical replacement. "
        "device_peaks.csv retains the explicit original missing row plus a separate supplemental_repeat row "
        "with source_jobid/measurement_jobid/original_jobid provenance, "
        "and excludes incidental solver-host GPUs. Raw device/repetition values remain in the CSV only; "
        "there are no dots or scatter overlays in any timing or memory figure. "
        "Observed maxima are capacity summaries, not estimators of a mean: no misleading CI or "
        "range whisker is attached to them.\n\n"
        "The heterogeneous surrogate and variable workload require caution: solver rank counts, overlap, "
        "aggregate samples and batch sizes vary across placements/models/providers. See metadata.json workloads; "
        "these are not workload-normalized speedups or application-level performance claims.\n\n"
        "## Coverage And Confounds\n\n"
        "The 12 SmartSim DB-per-GPU measurements for 96c4g and 96p96c4g are NOT LOCATED "
        "(2 models x 2 four-GPU placements x 3 repetitions). This is missing measurement coverage, "
        "not a plot-code omission: the current 288-run matrix contains only shared c0 SmartSim. "
        "No DB-per-GPU bars are depicted or fabricated. Available per-device memory "
        "peaks are not perGPU timing evidence. No cross-library GPU Score-P phase dataset is available; "
        "no phase plots are fabricated.\n\n"
        "All selected shared SmartSim runs use DB=1, intra=4, inter=1. "
        "Transformer 24c1g and 96c4g use TPQ=4, while heterogeneous 24p24c1g and 96p96c4g "
        "use TPQ=1; MLP uses TPQ=1 throughout. This TPQ confound prevents attributing "
        "shared/heterogeneous Transformer differences solely to placement.\n\n"
        "Canonical inputs are runs.csv, resources.csv and report.json in the consolidated snapshot. "
        "If present, the fixed results/cmi_bench_telemetry_retry_validation.json explicitly designates "
        "job 5106286; raw artifact and ledger hashes, original identity, configuration, seven steps, "
        "workload and expected device scope are validated before aggregation. There is no latest/fastest "
        "discovery and no automatic canonical replacement. Timing and all other memory observations are unchanged. "
        "SHA256 hashes and counts are in metadata.json. CSV audit units are seconds and GiB. "
        "PNG is rasterized from the PDF at 300 dpi; PDF/PGF use the thesis_style pdflatex/Latin Modern "
        "pipeline without tight bounding-box resizing. No thesis LaTeX is modified.\n\n"
        "The linear MLP axis preserves large variability but compresses fast AIx bars. "
        "Memory panels keep host roles separate; raw imbalance can be audited in device_peaks.csv, "
        "but is not shown by maximum-only bars.\n"
    )
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Rendered drafts: {args.out_dir}")


if __name__ == "__main__":
    main()
