#!/usr/bin/env python3
"""Render MAIA GPU readiness timings from the canonical snapshot; never run jobs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/cmi_bench_consolidated"
OUTPUT = ROOT / "figures/ch7_readiness_draft"
MODELS = ("mlp", "transformer")
MODEL_LABELS = {"mlp": "MLP (ov4)", "transformer": "Transformer (ov0)"}
HARDWARE = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
HW_LABELS = ("24c\n1GPU", "24+24c\n1GPU", "96c\n4GPUs", "96+96c\n4GPUs")
VARIANTS = ("aix_default_coll", "phydll_default", "phydll_readiness")
LABELS = ("AIx", "PhyDLL default", "PhyDLL readiness")
COLORS = ("#577787", "#C97AA7", "#009E73")
T95 = 4.30265
PANEL_MM = 73.5
HEIGHT_MM = 65


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def variant(row):
    if row["provider"] == "aix" and row["layout"] == "default" and row["comm"] == "coll":
        return VARIANTS[0]
    if row["provider"] == "phydll" and row["wait"] in ("default", "readiness"):
        return "phydll_" + row["wait"]
    return None


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected three finite nonnegative independent repetition means")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    return mean, sd, T95 * sd / math.sqrt(3)


def collect(data_dir=DATA):
    report = json.loads((data_dir / "report.json").read_text())
    if report["status"] != "ok" or report["errors"]:
        raise ValueError("Consolidation report is not clean")
    runs = [r for r in read_csv(data_dir / "runs.csv") if variant(r)]
    if len(runs) != 72 or len({r["cell"] for r in runs}) != 72:
        raise ValueError("Expected exactly 72 unique readiness/reference runs")
    calls = read_csv(data_dir / "inferences.csv")
    values = []
    for model in MODELS:
        for hardware in HARDWARE:
            for config in VARIANTS:
                group = sorted((r for r in runs if (r["model"], r["placement"], variant(r))
                                == (model, hardware, config)), key=lambda r: int(r["rep"]))
                if [r["rep"] for r in group] != ["1", "2", "3"]:
                    raise ValueError(f"Incomplete repetitions: {model}/{hardware}/{config}")
                for run in group:
                    schedule = sorted((c for c in calls if c["cell"] == run["cell"]),
                                      key=lambda c: int(c["call"]))
                    if (len(schedule) != 7 or [c["call"] for c in schedule] != list("1234567")
                            or [c["phase"] for c in schedule] != ["warmup"]*2 + ["measured"]*5
                            or [int(c["step"]) for c in schedule] != [15, 44, 73, 102, 131, 160, 189]
                            or any(c["jobid"] != run["jobid"] for c in schedule)):
                        raise ValueError(f"Invalid inference schedule: {run['cell']}")
                    measured = np.array([float(c["latency_ms"]) for c in schedule[2:]])
                    logged = np.array([float(v) for v in run["measured_ms"].split(";")])
                    if (run["exit_status"] != "0" or run["solver_exit_status"] != "0"
                            or run["n_measured"] != "5" or run["n_inferences"] != "7"
                            or run["overlap"] != ("4" if model == "mlp" else "0")
                            or measured.shape != logged.shape or not np.array_equal(measured, logged)
                            or not np.isfinite(measured).all() or (measured < 0).any()
                            or not math.isclose(float(measured.mean()), float(run["measured_mean_ms"]),
                                                rel_tol=1e-12, abs_tol=1e-9)):
                        raise ValueError(f"Invalid measured calls: {run['cell']}")
                reps = [float(r["measured_mean_ms"]) / 1000 for r in group]
                mean, sd, half = mean_ci(reps)
                values.append(dict(model=model, placement=hardware, variant=config,
                                   mean_s=mean, sd_s=sd, ci_half_s=half,
                                   ci_low_s=mean-half, ci_high_s=mean+half,
                                   n_reps=3, calls_per_rep=5, df=2, t95=T95,
                                   **{f"rep{i+1}_s": v for i, v in enumerate(reps)},
                                   cells=";".join(r["cell"] for r in group),
                                   jobids=";".join(r["jobid"] for r in group)))
    return values, runs


def style():
    plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
                         "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
                         "font.family": "serif", "font.size": 8,
                         "axes.labelsize": 9, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .5, "savefig.bbox": None})


def panel(fig, rows, model, start_mm, width_mm=PANEL_MM):
    width, height = fig.get_size_inches() * 25.4
    ax = fig.add_axes([(start_mm+10)/width, 12/height, (width_mm-12)/width, 40/height])
    subset = [r for r in rows if r["model"] == model]
    for j, (config, color) in enumerate(zip(VARIANTS, COLORS)):
        group = [next(r for r in subset if r["placement"] == hw and r["variant"] == config)
                 for hw in HARDWARE]
        ax.bar(np.arange(4)+(j-1)*.25, [r["mean_s"] for r in group], width=.22,
               color=color, edgecolor=".25", linewidth=.4,
               yerr=[r["ci_half_s"] for r in group], capsize=1.3,
               error_kw={"linewidth": .65, "capthick": .65})
    low = min(0, min(r["ci_low_s"] for r in subset))
    high = max(r["ci_high_s"] for r in subset)
    ax.set_ylim(low*1.05, high*1.08)
    ax.set_xlim(-.55, 3.55)
    ax.set_xticks(range(4), HW_LABELS)
    ax.tick_params(length=2, pad=2)
    ax.set_ylabel("ML-step time (s)", labelpad=3)
    ax.set_title(MODEL_LABELS[model], loc="left", pad=5)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".85", linestyle="--", linewidth=.4)
    # One readable horizontal legend per panel, below the hardware labels.
    ax.legend(handles=[Patch(facecolor=c, edgecolor=".25", linewidth=.4, label=l)
                       for c, l in zip(COLORS, LABELS)],
              loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3,
              frameon=False, fontsize=6.5, handlelength=.9,
              handletextpad=.35, columnspacing=.65, borderaxespad=0)
    return ax


def figure(rows, models=MODELS):
    width = 150 if len(models) == 2 else PANEL_MM
    fig = plt.figure(figsize=(width/25.4, HEIGHT_MM/25.4))
    for i, model in enumerate(models):
        panel(fig, rows, model, i*(PANEL_MM+3))
    return fig


def save(fig, path):
    fig.savefig(path.with_suffix(".pgf"), bbox_inches=None)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=None,
                metadata={"CreationDate": None, "ModDate": None})
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "300",
                    str(path.with_suffix(".pdf")), str(path)], check=True)
    plt.close(fig)


NOTES = """# GPU Readiness Draft

MAIA GPU data only; no CPU or terrain measurements or substitutions.
Regenerate from the repository root using the thesis Python environment:

```sh
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 scripts/plot_cmi_ch7_readiness.py
```

Requires matplotlib, numpy, pdflatex with T1/Latin Modern, and pdftocairo on PATH.
The renderer uses the thesis guide's PGF/pdflatex pipeline; PDFs embed Latin Modern
Type 1 fonts and PNGs are rasterized from those PDFs at 300 dpi.

## Figures

- [Combined preview](readiness_latency_1x2.png): 150 x 65 mm, independent y scales.
- [MLP panel](mlp_latency.pgf): 73.5 x 65 mm; PDF and PNG companions share the stem.
- [Transformer panel](transformer_latency.pgf): 73.5 x 65 mm; PDF and PNG companions share the stem.

Panels are independently authored at 73.5 mm (0.49 textwidth), separated by 3 mm
in the full-width preview, not scaled-down full-size plots. Include at native size.
Axis labels/titles are 9 pt, ticks 8 pt, local legend 6.5 pt. No repetition dots,
bar-value annotations, figure captions or subfigure letters are baked in. Exact
values are in plot_values.csv; LaTeX owns the captions. PGFs need the pgf package;
use T1 font encoding and lmodern in the including document. No thesis files edited.

## Statistics

Bars: arithmetic mean of three independent repetition means. Each repetition
uses ONLY five measured calls (calls 3-7, steps 73, 102, 131, 160, 189); calls
1-2 are warmup. Whiskers: two-sided 95% Student-t confidence interval, df=2,
t=4.30265, half-width = t * sample SD(rep means) / sqrt(3). The 15 calls are
not 15 independent repetitions. Intervals are not clipped at zero.
Timing is the synchronized solver ML-step latency, not whole-job wall time,
isolated model forward time, or a phase decomposition.

The three variants are AIx default layout/collective (reference), PhyDLL default
(blocking), and PhyDLL readiness. Labels: MLP (ov4), Transformer (ov0).
All four placements are retained: 24c1g, 24p24c1g, 96c4g, 96p96c4g.
The plus sign denotes heterogeneous CPU-side + GPU-side solver ranks (48/192
total), using c23g+c23g surrogate nodes, not CPU-only measurements. Workload
aggregates differ across placements: comparisons within a hardware group are
matched; across-group bars must not be interpreted as fixed-work strong scaling.

## Audit

plot_values.csv contains all 24 bar means, sample SDs, CI bounds, repetition means,
cell identities and job IDs. selected_runs.csv preserves all 72 selected source
rows, measured calls, workload sizes, overlap, paths and retry lineage.
manifest.json records SHA-256 and byte counts for the three input files, selection
counts, statistical constants and authored dimensions. The loader rejects missing
or duplicate repetitions, failed runs, wrong overlap, inconsistent per-call
timings, and an invalid seven-call schedule. It never selects fastest runs.
The consolidation report's missing AIx VRAM observation is unrelated to timing;
no memory measurements are plotted here. Chaining coverage is a separate audit
in ../ch7_smartsim_chaining_draft, not part of this readiness renderer.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows, runs = collect(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    save(figure(rows), args.out_dir / "readiness_latency_1x2")
    for model in MODELS:
        save(figure(rows, (model,)), args.out_dir / f"{model}_latency")
    write_csv(args.out_dir / "plot_values.csv", rows)
    write_csv(args.out_dir / "selected_runs.csv", runs)
    manifest = dict(sources={name: dict(path=str((args.data_dir/name).resolve()),
                                       sha256=hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest(),
                                       bytes=(args.data_dir/name).stat().st_size)
                             for name in ("runs.csv", "inferences.csv", "report.json")},
                    bars=24, independent_runs=72, measured_calls=360, calls_per_run=5,
                    student_t_critical=T95, df=2, combined_width_mm=150,
                    panel_width_mm=PANEL_MM, height_mm=HEIGHT_MM,
                    fonts_pt=dict(axis=9, ticks=8, legend=6.5),
                    consolidation_warnings=json.loads((args.data_dir/"report.json").read_text())["warnings"])
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.out_dir / "README.md").write_text(NOTES)
    print(f"Rendered 3 PGF/PDF/PNG figures and 24 audited bars in {args.out_dir}")


if __name__ == "__main__":
    main()
