#!/usr/bin/env python3
"""Phase-only AIx collective GPU drafts from the canonical selected runs."""

import argparse
import itertools
import json
from pathlib import Path
import statistics

import plot_cmi_aix_timeline as raw
import plot_cmi_ch6 as style

plt, ts = style.plt, style.ts
ROOT = raw.ROOT
LAYOUTS = ("default", "affine", "numa")
HOST = ("gather_collective_ms", "device_cpu_ms", "scatter_collective_ms")
FIVE = ("gather_collective_ms", "h2d_ms", "forward_ms", "d2h_ms", "scatter_collective_ms")
LABELS = {"gather_collective_ms": "Gather (host)", "device_cpu_ms": "Device (host)",
          "scatter_collective_ms": "Scatter (host)", "h2d_ms": "H2D (CUDA)",
          "forward_ms": "Forward (CUDA)", "d2h_ms": "D2H (CUDA)"}
COLORS = {"gather_collective_ms": "#577787", "device_cpu_ms": "#C97AA7",
          "scatter_collective_ms": "#8064A2", "h2d_ms": "#67A9A0",
          "forward_ms": "#C97AA7", "d2h_ms": "#D4A45A"}
SCOPES = {"host3": HOST, "measured5": FIVE}


def select_workgroup(rows, phases):
    """Keep all components of one WG, not independent per-phase maxima."""
    return max(rows, key=lambda r: (sum(r[p] for p in phases), -r["workgroup"]))


def summarize(calls):
    result = []
    for model, hw, layout, scope in itertools.product(style.MODELS, style.HARDWARE, LAYOUTS, SCOPES):
        phases = SCOPES[scope]
        subset = [r for r in calls if (r["model"], r["placement"], r["layout"], r["scope"])
                  == (model, hw, layout, scope)]
        if len(subset) != 15 or {(int(r["rep"]), r["call"]) for r in subset} != set(
                itertools.product(range(1, 4), range(2, 7))):
            raise ValueError(f"Incomplete repetitions/calls: {model}/{hw}/{layout}/{scope}")
        for phase in phases:
            means = [statistics.mean(r[phase] for r in subset if int(r["rep"]) == rep)
                     for rep in range(1, 4)]
            result.append(dict(model=model, placement=hw, layout=layout, scope=scope, phase=phase,
                               mean_ms=statistics.mean(means), sd_run_means_ms=statistics.stdev(means),
                               rep1_ms=means[0], rep2_ms=means[1], rep3_ms=means[2], n_reps=3,
                               measured_calls_per_rep=5))
    return result


def collect(data_dir=raw.DATA):
    runs = raw.read_csv(data_dir / "runs.csv")
    inferences = raw.read_csv(data_dir / "inferences.csv")
    report = json.loads((data_dir / "report.json").read_text())
    if report["status"] != "ok" or report["errors"]:
        raise ValueError("Canonical consolidation report is not clean")
    if len(runs) != 288 or len({r["cell"] for r in runs}) != 288:
        raise ValueError("Expected the canonical 288 unique selected runs")
    selected = [r for r in runs if r["provider"] == "aix" and r["comm"] == "coll"]
    expected = set(itertools.product(style.MODELS, style.HARDWARE, LAYOUTS, ("1", "2", "3")))
    if len(selected) != 72 or {(r["model"], r["placement"], r["layout"], r["rep"])
                               for r in selected} != expected:
        raise ValueError("Expected all 72 collective runs, three layouts and three repetitions")
    sources = {data_dir / n for n in ("runs.csv", "inferences.csv", "report.json")}
    sources.update((Path(__file__), Path(raw.__file__), Path(style.__file__), Path(ts.__file__)))
    sources.update(ROOT / p for p in (
        "slurm/run_cmi_smoke_gpu.sh", "slurm/cmi_bench_het_engine.sh",
        "CPP-ML-Interface/dl_clients/phydll_dl_client.py",
        "CPP-ML-Interface/dl_clients/smartsim_controller.py",
        "CPP-ML-Interface/extern/AIxeleratorService/src/aixeleratorService/aixeleratorService.cpp",
        "CPP-ML-Interface/extern/AIxeleratorService/src/inferenceStrategy/torchInference/torchInference.cpp",
        "CPP-ML-Interface/extern/AIxeleratorService/src/distributionStrategy/roundRobinDistribution.cpp"))
    groups, calls, coverage = [], [], []
    for run in runs:
        directory = ROOT / run["run_dir"]
        profiles = sorted(str(p.relative_to(ROOT)) for pattern in ("*.cubex", "*.otf2")
                          for p in directory.rglob(pattern))
        for name in ("solver.log", "controller.log", "run.info"):
            if (directory / name).is_file():
                sources.add(directory / name)
        rendered = run in selected
        reason = ("Native controller host stages and CUDA durations available" if rendered else
                  "P2P request timelines exist; no collective gather/scatter five-stage scope" if run["provider"] == "aix" else
                  "DL Score-P regions instrumented, but selected GPU campaign disables profiling; no matching phase export"
                  if run["provider"] == "phydll" else
                  "Controller log records deployment/lifecycle, not measured per-call phase durations")
        if run["provider"] != "aix" and profiles:
            raise ValueError(f"New provider profile evidence requires review: {run['cell']}: {profiles}")
        coverage.append(dict(cell=run["cell"], provider=run["provider"], model=run["model"],
                             placement=run["placement"], layout=run["layout"], comm=run["comm"],
                             wait=run["wait"], rep=run["rep"], jobid=run["jobid"], run_dir=run["run_dir"],
                             rendered=rendered, status="available" if rendered else "not_rendered",
                             reason=reason, matching_profiles=";".join(profiles)))
    for run in selected:
        if (run["n_measured"] != "5" or run["n_inferences"] != "7"
                or run["overlap"] != ("0" if run["model"] == "transformer" else "4")):
            raise ValueError(f"Invalid steady-call/model scope: {run['cell']}")
        signatures = None
        for call in range(2, 7):
            metrics, _, paths = raw.load_run(run, inferences, call)
            sources.update(paths)
            members = [int(rank) for r in metrics for rank in r["members"].split(";")]
            signature = {(r["workgroup"], r["controller"], r["gpu_ordinal"], r["members"]) for r in metrics}
            if (len(metrics) != int(run["gpu_count"]) or sorted(members) != list(range(int(run["solver_ranks"])))
                    or len({r["gpu_ordinal"] for r in metrics}) != int(run["gpu_count"])
                    or (signatures is not None and signature != signatures)):
                raise ValueError(f"Incomplete/conflicting WG/GPU mapping: {run['cell']}")
            signatures = signature
            for row in metrics:
                if row["diagnostic_barriers"] != 0 or row["gather_arrival_wait_ms"] or row["scatter_arrival_wait_ms"]:
                    raise ValueError("Enabled diagnostic barriers/arrival waits are outside this scope")
                if row["device_batches"] <= 0:
                    raise ValueError("Missing device batches")
                row["host_stage_sum_ms"] = sum(row[p] for p in HOST)
                row["measured_five_sum_ms"] = sum(row[p] for p in FIVE)
                row["device_minus_cuda_ms"] = row["device_cpu_ms"] - row["gpu_sum_ms"]
                groups.append(row)
            for scope, phases in SCOPES.items():
                winner = select_workgroup(metrics, phases)
                calls.append(dict(scope=scope, **winner))
    values = summarize(calls)
    metadata = dict(cells={r["cell"]: dict(sha256=raw.cell_hash(r), selected_row=r) for r in selected},
                    sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): raw.sha256(p)
                             for p in sorted(sources)},
                    counts=dict(canonical_runs=len(runs), collective_runs=len(selected),
                                workgroup_call_rows=len(groups), selected_call_scope_rows=len(calls),
                                plot_phase_values=len(values), figure_triplets=6),
                    aggregation="For each scope/call choose one WG maximizing that scope's stage sum; "
                                "tie: lowest WG. Average five calls per run, then three run means. Never sum GPUs.",
                    residual_definition="device_cpu_ms - (h2d_ms + forward_ms + d2h_ms); signed clock/scope discrepancy, "
                                        "not identified overhead, never clipped or stacked",
                    negative_residual_workgroup_calls=sum(r["device_minus_cuda_ms"] < 0 for r in groups),
                    scopes={k: list(v) for k, v in SCOPES.items()},
                    dimensions_mm=dict(width=150, combined_height=62, standalone_height=62),
                    consolidation_warnings=report["warnings"])
    return groups, calls, values, coverage, metadata


def render(values, scope, models):
    fig, axes = plt.subplots(1, len(models), figsize=(150/25.4, 62/25.4), squeeze=False)
    phases = SCOPES[scope]
    for ax, model in zip(axes.flat, models):
        for h, hw in enumerate(style.HARDWARE):
            for l, layout in enumerate(LAYOUTS):
                bottom = 0
                for phase in phases:
                    row = raw.unique([r for r in values if (r["model"], r["placement"], r["layout"], r["scope"], r["phase"])
                                      == (model, hw, layout, scope, phase)], "plot phase")
                    y = row["mean_ms"] / 1000
                    ax.bar(h+(l-1)*.25, y, .22, bottom=bottom, color=COLORS[phase],
                           edgecolor=".25", linewidth=.35)
                    bottom += y
        if len(models) > 1:
            ax.set_title(style.MODEL_LABELS[model], loc="left", fontsize=9, pad=7)
        ax.set_ylabel("Measured stage sum (s)", fontsize=8)
        ax.set_xticks(range(4), [s.replace("/", "/\n") for s in style.HW_LABELS], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False, axis="x")
        ax.grid(axis="y", color=".85", linewidth=.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(bottom=0)
        ax.margins(y=.12)
    handles = [style.Patch(facecolor=COLORS[p], label=LABELS[p]) for p in phases]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, fontsize=7, frameon=False, columnspacing=1.2, handlelength=1.2)
    fig.subplots_adjust(left=.085 if len(models) == 2 else .115, right=.985,
                        bottom=.20, top=.78, wspace=.35)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=raw.DATA)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/gpu_phases_draft")
    args = parser.parse_args(argv)
    groups, calls, values, coverage, metadata = collect(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for scope in SCOPES:
        for models in (style.MODELS, ("transformer",), ("mlp",)):
            name = "aix_collective_" + scope + "_" + "_".join(models)
            ts.save(render(values, scope, models), args.out_dir / name)
            gallery.extend((f"## {name}", "", f"[PGF]({name}.pgf) | [PDF]({name}.pdf)",
                            f"![{name}]({name}.png)", ""))
    for name, rows in (("workgroup_calls.csv", groups), ("selected_calls.csv", calls),
                       ("plot_values.csv", values), ("coverage.csv", coverage)):
        raw.write_csv(args.out_dir / name, rows)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    notes = ["# GPU Phase-Only Drafts", "",
             "Six figures, each as PGF/PDF/PNG: host3 and measured5, each as a 150 mm two-model row "
              "and two 150 mm standalones. All canvases are 150 x 62 mm. "
              "Transformer ov0; MLP ov4. No memory or ML-step bars.", "",
              "Graphics contain no explanatory footers or subcaptions in any format. Standalones are title-free; "
              "two-model rows retain model headings for panel identification. "
              "The embedding LaTeX caption owns titles and explanatory notes. "
              "Within each hardware group, bars run left to right: default, affine, NUMA.", "",
             "## Scope And Aggregation", "",
             "All 72 canonical AIx collective runs: 2 models x 4 hardware placements x 3 layouts x 3 repetitions. "
             "Diagnostic calls 2..6 (one-based inference calls 3..7) are measured; calls 0,1 are excluded. "
             "Each stack retains all phases of the SAME controller/WG maximizing the sum for that scope/call. "
             "Host3 and measured5 may therefore select different WGs. Ties choose the lowest WG. "
             "Five call vectors are averaged per run, then three run vectors are averaged. "
             "No sum across GPUs, no independent phase maxima, no global critical-path claim, no CI dots/whiskers.", "",
             "Host3 uses steady_clock gather/device/scatter durations on a controller; device_cpu includes GPU work. "
             "These are native CPU-clock durations during GPU runs, NOT CPU-only benchmark observations. "
             "Interstage gaps and other solver/library operations are omitted.", "",
             "Measured5 uses host gather, CUDA-event H2D/forward/D2H batch sums, and host scatter. "
             "It is a mixed-clock measured-stage sum, NOT continuous elapsed time or total ML-step latency. "
             "The signed residual device_cpu - sum(CUDA phases) is retained in workgroup_calls.csv and selected_calls.csv. "
             "It can be negative, is never clipped, and cannot be called measured overhead or stacked as an exclusive phase. "
             "Do not add device_cpu to CUDA durations. No fictitious Other phase is inferred.", "",
             "## Provider Coverage", "",
             "coverage.csv audits all 288 selected cells. AIx: 72 collective runs rendered; 144 P2P runs have "
             "request timelines but not this collective five-stage scope. PhyDLL: 48 runs, no matching selected "
             "phase profiles; DL-side Score-P regions exist in source, but current production runners disable "
             "profiling. SmartSim: 24 runs, controller.log gives deployment/DB/thread/lifecycle diagnostics, "
             "not per-call put/model-request/get/unpack timings. Selected directories are searched recursively "
             "for .cubex/.otf2 artifacts; any new non-AIx profile evidence stops rendering for review. "
             "This bounded audit does not assert that unrelated historical profiles cannot exist.", "",
             "CPU renderer reference: ../smartsim/cpu_benchmark/plot_cpu_comparison.py phase_panel and "
             "solver_summary(phase_only=True): phase-only grouping and physical-width thesis exports. "
             "No CPU/terrain observations, historical unpaired profiles, total-latency fractions, or zeros "
             "substitute for missing provider phases. Consequently no cross-provider phase figure is rendered.", "",
             "## Reproduction", "", "`python scripts/plot_cmi_gpu_phases.py`", "",
             "plot_values.csv contains exact plotted phase means, three run means and their sample SDs. "
             "selected_calls.csv records each selected controller/WG; workgroup_calls.csv retains all candidates, "
             "mapping, batch counts and signed residuals. metadata.json hashes the canonical inputs, raw logs, "
             "controller diagnostics, renderer/style and instrumentation sources, and records selected rows. "
             "No changes are made to canonical results or results/cmi_aix_stage_tables.", "", *gallery]
    (args.out_dir / "README.md").write_text("\n".join(notes)+"\n")
    print(f"Rendered 6 phase-only PGF/PDF/PNG triplets; {len(groups)} WG calls; "
          f"{len(coverage)} coverage rows; {metadata['negative_residual_workgroup_calls']} signed negative residuals")


if __name__ == "__main__":
    main()
