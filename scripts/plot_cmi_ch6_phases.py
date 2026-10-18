#!/usr/bin/env python3
"""Partial Chapter 6 provider-phase drafts; missing observations are never zeros."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import plot_cmi_gpu_phases as phases

raw, style = phases.raw, phases.style
ROOT = raw.ROOT
CPU = ROOT.parent / "smartsim/cpu_benchmark/plot_cpu_comparison.py"
PROVIDERS = ("aix", "phydll", "ss_shared", "ss_pergpu")
LABELS = ("AIx", "PhyDLL", "SS shared", "SS per-GPU")
# Reuse the CPU palette; CUDA Forward keeps the exact CPU Forward color.
COLORS = ("#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2")
PHASE_LABELS = ("Gather (host)", "H2D (CUDA)", "Forward (CUDA)", "D2H (CUDA)", "Scatter (host)")
SEMANTICS = {
    "cpu_reference": {
        "AIx": {"Inference": "#8064A2"},
        "PhyDLL": {"Input receive": "#4C78A8", "Forward": "#F58518",
                   "Output send": "#54A24B", "Other": "#BAB0AC"},
        "SmartSim": {"Put": "#72B7B2", "Model request": "#E45756",
                     "Retrieval/unpack": "#B279A2", "Other": "#BAB0AC"}},
    "gpu_aix_extension": dict(zip(phases.FIVE, COLORS))}


def inventory():
    """Bounded read-only discovery, not automatic admission of unpaired profiles."""
    neighbor = ROOT.parent / "smartsim"
    scopes = [ROOT / "results", ROOT / "scratch", neighbor / "mini_app/results",
              neighbor / "mini_app/scratch", neighbor / "scratch",
              neighbor / "cpu_benchmark/mmcp_benchmark/results",
              neighbor / "cpu_benchmark/mmcp_benchmark/scratch",
              neighbor / "scorep_runs"]
    scopes.extend(sorted(ROOT.glob("scorep-*")))
    scopes.extend(sorted(neighbor.glob("scorep-*")))
    rows, bounds = [], []
    for scope in scopes:
        count, pruned = 0, 0
        if scope.is_dir():
            for directory, dirs, files in os.walk(scope):
                count += 1
                if count > 20000:
                    raise ValueError(f"Discovery directory limit exceeded: {scope}")
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
                if len(Path(directory).relative_to(scope).parts) >= 10:
                    pruned += len(dirs)
                    dirs[:] = []
                for name in sorted(files):
                    p = Path(directory) / name
                    manifest = name in ("cases.csv", "manifest.csv", "summary_combined.csv")
                    if name != "run.info" and not manifest and p.suffix not in (".cubex", ".otf2"):
                        continue
                    info = {}
                    if name == "run.info":
                        info = dict(line.split("=", 1) for line in p.read_text(errors="replace").splitlines()
                                    if "=" in line)
                    model_source = info.get("MODEL", "")
                    if manifest:
                        model_source = ";".join(sorted({r.get("model", "") for r in raw.read_csv(p)} - {""}))
                    context = " ".join((model_source, *(info.get(k, "") for k in ("MAIA", "CELL", "CASE"))))
                    model = (model_source if manifest else
                             "terrain" if any(s in context.lower() for s in ("watercnn", "giantmlp")) else
                             "transformer" if "transformer" in context.lower() else
                             "mlp" if "mlp" in context.lower() else "unknown")
                    rows.append(dict(path=str(p), kind="manifest" if manifest else "run_info" if info else "profile",
                                     sha256=raw.sha256(p), size_bytes=p.stat().st_size,
                                     mtime_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                                     model_hint=model, model_source=model_source,
                                     maia_source=info.get("MAIA", ""), cell=info.get("CELL", ""),
                                     decision="discovery only; no matching GPU phase admission"))
        bounds.append(dict(path=str(scope), exists=scope.is_dir(), visited_directories=count,
                           depth_limit=10, pruned_directories=pruned))
    return rows, bounds


def collect(data_dir=raw.DATA):
    groups, calls, values, _, metadata = phases.collect(data_dir)
    groups = [r for r in groups if r["layout"] == "default"]
    calls = [r for r in calls if r["layout"] == "default" and r["scope"] == "measured5"]
    values = [dict(r, provider="aix", status="measured") for r in values
              if r["layout"] == "default" and r["scope"] == "measured5"]
    runs = raw.read_csv(data_dir / "runs.csv")
    coverage = []
    for model in style.MODELS:
        for hw in style.HARDWARE:
            for provider in PROVIDERS:
                if provider == "ss_pergpu" and "1g" in hw:
                    continue
                group = [r for r in runs if r["model"] == model and r["placement"] == hw
                         and style.baseline(r) and r["provider"] == {
                             "ss_shared": "smartsim", "ss_pergpu": "absent"}.get(provider, provider)]
                if provider != "ss_pergpu" and sorted(r["rep"] for r in group) != ["1", "2", "3"]:
                    raise ValueError(f"Incomplete baseline repetitions: {model}/{hw}/{provider}")
                reason = {"aix": "Native host gather/scatter + CUDA H2D/forward/D2H; default collective",
                          "phydll": "Selected production no-Score-P; no matching phase export",
                          "ss_shared": "Selected controller logs are lifecycle-only, not per-call phases",
                          "ss_pergpu": "Experiment absent: three repetitions needed for this model/hardware"}[provider]
                coverage.append(dict(model=model, placement=hw, provider=provider,
                                     status="measured" if provider == "aix" else "missing_phase_data",
                                     mean_ms="", measured_phase_count=5 if provider == "aix" else 0,
                                     selected_runs=len(group), required_runs=3, reason=reason,
                                     cells=";".join(r["cell"] for r in group),
                                     run_dirs=";".join(r["run_dir"] for r in group)))
    instrumentation = [ROOT / p for p in (
        "CPP-ML-Interface/dl_clients/dl_client.cpp",
        "CPP-ML-Interface/dl_clients/phydll_dl_runtime.cpp",
        "CPP-ML-Interface/dl_clients/CMakeLists.txt",
        "CPP-ML-Interface/include/provider/ml_coupling_provider_phydll.hpp")]
    metadata["sources"].update({str(p): raw.sha256(p) for p in (CPU, Path(__file__), *instrumentation)})
    metadata["counts"] = dict(canonical_runs=288, baseline_runs=72, measured_aix_runs=24,
                              workgroup_calls=len(groups), selected_calls=len(calls),
                              measured_phase_values=len(values), coverage_slots=len(coverage),
                              measured_slots=8, missing_slots=20, absent_pergpu_runs=12, figure_triplets=8)
    metadata["dimensions_mm"] = dict(width=150, height=60)
    metadata["plot_scope"] = "Adjacent individual phase columns within provider groups; all bar bottoms zero"
    metadata["negative_residual_workgroup_calls"] = sum(r["device_minus_cuda_ms"] < 0 for r in groups)
    metadata["scopes"] = {"measured5": list(phases.FIVE)}
    metadata["cells"] = {k: v for k, v in metadata["cells"].items()
                         if v["selected_row"]["layout"] == "default"}
    metadata["phase_semantics_and_colors"] = SEMANTICS
    return groups, calls, values, coverage, metadata


def render(values, coverage, model, hw):
    fig, ax = style.plt.subplots(figsize=(150/25.4, 60/25.4))
    providers = PROVIDERS if "4g" in hw else PROVIDERS[:3]
    subset = [r for r in values if r["model"] == model and r["placement"] == hw]
    for x, provider in enumerate(providers):
        status = raw.unique([r for r in coverage if (r["model"], r["placement"], r["provider"])
                             == (model, hw, provider)], "coverage slot")["status"]
        if status != "measured":
            # Coverage annotations have no duration meaning or data-limit effect.
            ax.text(x, .10, "No phase data", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=".4")
            continue
        for j, (phase, color) in enumerate(zip(phases.FIVE, COLORS)):
            row = raw.unique([r for r in subset if r["phase"] == phase and r["provider"] == provider], "measured phase")
            y = raw.number(row["mean_ms"])/1000
            pos = x + (j - (len(phases.FIVE)-1)/2)*.19
            bars = ax.bar(pos, y, .19*.9, bottom=0, color=color, edgecolor=".25", linewidth=.4)
            bars[0].set_gid("measured-phase")
            ax.annotate(f"{y:.3g}", (pos, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=7, color=".25")
    ax.set_ylabel("Phase duration (s)", fontsize=9)
    ax.set_xticks(range(len(providers)), LABELS[:len(providers)], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(axis="y", color=".85", linestyle="--", linewidth=.4)
    ax.set_xlim(-.6, len(providers)-.4)
    ax.set_ylim(0, max(r["mean_ms"] for r in subset)/1000*1.30)
    handles = [style.Patch(facecolor=c, label=l) for c, l in zip(COLORS, PHASE_LABELS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1), ncol=3,
               frameon=False, fontsize=8, handlelength=1.3)
    fig.subplots_adjust(left=.11, right=.98, top=.80, bottom=.17)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=raw.DATA)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/ch6_phases_draft")
    args = parser.parse_args(argv)
    groups, calls, values, coverage, metadata = collect(args.data_dir)
    discovery, bounds = inventory()
    metadata["discovery"] = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), scopes=bounds,
                                 counts=dict(Counter(r["kind"] for r in discovery)),
                                 policy="Bounded inventory only; unpaired profiles never admitted by filename/model hint")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery = []
    for model in style.MODELS:
        for hw in style.HARDWARE:
            name = f"{model}_{hw}_phases"
            style.ts.save(render(values, coverage, model, hw), args.out_dir / name)
            gallery.extend([f"## {name}", f"[PGF]({name}.pgf) | [PDF]({name}.pdf)", f"![{name}]({name}.png)", ""])
    for name, rows in (("workgroup_calls", groups), ("selected_calls", calls), ("plot_values", values),
                       ("coverage", coverage), ("discovery", discovery)):
        raw.write_csv(args.out_dir / f"{name}.csv", rows)
    metadata["artifact_sha256"] = {p.name: raw.sha256(p) for p in sorted(args.out_dir.iterdir())
                                   if p.suffix in (".pgf", ".pdf", ".png", ".csv")}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    discovery_summary = ("## Fresh Inventory Counts\n\n" +
                         ", ".join(f"{v} {k}" for k, v in sorted(metadata["discovery"]["counts"].items())) +
                         ". See discovery.csv for paths, dates, model provenance and hashes.\n\n")
    (args.out_dir / "README.md").write_text(README + "\n" + discovery_summary + "\n".join(gallery))
    print(json.dumps(metadata["counts"], indent=2))


README = """# Chapter 6 Provider Phases: Partial Draft

Eight 150 x 60 mm phase-only figures, PGF/PDF/PNG. This is NOT a complete
cross-provider phase comparison. Each has one measured AIx group with five
adjacent, zero-based phase columns; 20 provider groups lack phase measurements.
"No phase data" is a coverage annotation in axis coordinates, not data or zero bars.
The legend contains only the five measured AIx phases.
Graphics have no titles, subcaptions or explanatory footers in any format;
the embedding LaTeX caption owns model/hardware identification and caveats.
On 1 GPU shared DB and per-GPU DB are the same topology, shown only once.
On 4 GPUs the per-GPU dimension is absent: 2 models x 2 placements x 3 reps = 12 runs.

## Measured Scope

24 selected AIx default-collective runs, Transformer ov0 and MLP ov4, four
placements, three repetitions, five steady calls each (diagnostic 2..6,
one-based inference 3..7). The source loader validates all 72 collective runs
across layouts; only default is plotted here. Each call keeps the SAME WG
maximizing gather + H2D + forward + D2H + scatter (ties: lowest WG).
Average five vectors per run, then three run means. Never sum GPUs or take
independent phase maxima. The sum is used ONLY to select one WG per call, not as
a plotted duration. Each column shows its own phase mean, starting at zero.
Host and CUDA phases have distinct clocks/scopes; neither individual phases nor
their sum establish total ML-step latency or a global critical path. device_cpu
is not plotted alongside its CUDA subphases. Its signed discrepancy remains in the audit
tables; no clipping, invented Other, total-latency fractions, or CI.

## CPU Reference

Reference: ../smartsim/cpu_benchmark/figures/cpu_final/main/ and
plot_cpu_comparison.py, phase_panel / solver_summary(phase_only=True).
The CPU middle chart (transformer_comparison.png, visually inspected) uses
adjacent phase columns within provider groups, not stacks. This draft follows
that layout, zero baseline, rotated value labels, physical-width thesis output
and semantic palette, NOT CPU measurements. No stacked provider columns are used.
Exact CPU schemas: AIx Inference (#8064A2); PhyDLL Input receive (#4C78A8:
dl_recv + dl_frame_copy), Forward (#F58518: dl_torch_forward), Output send
(#54A24B: dl_send_output), Other (#BAB0AC: allocation/unpack/H2D/D2H/reorder).
SmartSim Put (#72B7B2: smartsim_put_tensor), Model request (#E45756:
smartsim_run_model), Retrieval/unpack (#B279A2: smartsim_unpack_tensor), Other
(#BAB0AC: smartsim_chunk_plan). These are NOT universal pack/transfer/wait bins.
GPU AIx extends the palette: host gather #4C78A8, H2D #72B7B2, CUDA forward
#F58518, D2H #54A24B, host scatter #B279A2. Similar colors express broad transfer
roles, not identical measurement scopes. No absent provider phase is in the
measured legend or plot_values.csv.

## Availability And Provenance

Selected PhyDLL production runs disable profiling; DL Score-P source regions do
not constitute measurements. Selected SmartSim controller logs describe lifecycle,
deployment and threading, not per-call phases. The loader rechecks selected raw
directories for .cubex/.otf2 and stops if new non-AIx profile evidence appears.
discovery.csv is a fresh bounded inventory of run.info, cases/manifest/combined
summary CSVs and .cubex/.otf2 in workspace
results/scratch, neighboring mini_app results/scratch, CPU campaign results/scratch,
and top-level historical Score-P roots. metadata.json records absent scopes,
depth pruning, timestamps, hashes and counts. Model hints come only from run.info
MODEL/MAIA/CELL/CASE or manifest model columns, not profile filenames.
Discovery is not proof of a matching
GPU run; unrelated CPU MAIA or GPU terrain profiles must not be substituted.
Coverage is limited to the explicitly listed roots, depth 10; not a filesystem-wide
claim. Raw source hashes and selected-cell hashes are in metadata.json; discovery
files and generated outputs have hashes too. Canonical results are unchanged.

Newer mini_app GPU campaigns (including gpu_refresh_20261013_optimized_profile_retry1)
have explicit cases.csv/manifest.csv model identities watercnn8k and giantmlp288;
retest_smartsim/summary_combined.csv likewise identifies those terrain models.
They do not provide MAIA Transformer/MLP GPU evidence. Nearby CPU steady-profile
run.info files identify build_cpu_steady_20261014_scorep and the CPU MLP model;
historical unpaired profiles remain unadmitted, not silently reclassified as GPU.

## Needed Capture Plan

A true full-provider measured phase graph is NOT possible from the selected
snapshot. Additional captures are BLOCKED pending parent authorization;
this script submits no jobs and changes no instrumentation or production config.

1. PhyDLL: 24 default GPU runs (2 models x 4 placements x 3 reps). The actual
   GPU launchers (slurm/run_cmi_smoke_gpu.sh and cmi_bench_het_engine.sh) execute
   the C++ phydll_dl_client, built from CPP-ML-Interface/dl_clients/dl_client.cpp
   by dl_clients/CMakeLists.txt, NOT the Python client. dl_client.cpp already has
   USE_SCOREP regions dl_input_allocate/dl_output_allocate, dl_input_unpack,
   dl_h2d, dl_torch_forward, dl_d2h, dl_output_reorder and dl_send_output.
   dl_clients/phydll_dl_runtime.cpp supplies dl_recv/dl_frame_copy/dl_send.
   include/provider/ml_coupling_provider_phydll.hpp supplies solver
   phydll_prepack/send/recv/unpack. Enable a separate Score-P build/capture or
   add per-call exports to these C++ sources after authorization. Existing
   regions measure host elapsed time, not CUDA-event durations: H2D/forward
   synchronize the current stream when PHYDLL_HAS_CUDA_STREAM is enabled, and
   dl_torch_forward includes output assembly. Add CUDA events for genuine
   device-only H2D/forward/D2H attribution; do not relabel host regions as such.
   Retain rank, call, GPU mapping and model hash; exclude two warmups. Do not
   add overlapping solver waits to DL work or count dl_send_output and its
   dl_send descendant twice. Selected production captures remain unavailable.
2. SmartSim shared: 24 instrumented GPU runs. Add per-call host timers around
   chunk planning, put, model request, and get/unpack in
   CPP-ML-Interface/include/provider/ml_coupling_provider_smartsim.hpp.
   Controller lifecycle logging alone cannot supply these. RedisAI/backend tracing
   is additionally required for genuine GPU forward/H2D/D2H attribution; client
   model-request elapsed time is NOT forward time. Preserve all selected TPQ,
   intra/inter-op, batching, overlap and sequential-put settings.
3. SmartSim per-GPU: 12 new runs on the two 4-GPU placements with both models,
   three reps; include the same phase instrumentation and DB-to-GPU provenance.
   Do not duplicate the 1-GPU shared measurements as independent observations.
4. For all captures: retain two warmups + five steady calls, explicit per-call
   scope, rank/GPU identity, model/config/binary hashes, exit status, and unprofiled
   controls to assess instrumentation perturbation. Keep new captures separate
   from the selected production snapshot. Provider-semantic adjacent phase
   columns become possible after capture; a common critical-path decomposition
   needs aligned causal traces and cannot be inferred by renaming these phases.

## Reproduce

`python scripts/plot_cmi_ch6_phases.py`

`python -m unittest discover -s scripts -p 'test_plot_cmi_ch6_phases.py'`

plot_values.csv contains only measured AIx phases, exact run means and sample SD.
coverage.csv has blank duration fields for missing providers. selected_calls.csv
and workgroup_calls.csv retain the measured vectors and candidate WGs.
"""


if __name__ == "__main__":
    main()
