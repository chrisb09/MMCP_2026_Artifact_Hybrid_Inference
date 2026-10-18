#!/usr/bin/env python3
"""Chapter 7 collective-derived ideal pipeline prediction, not a makespan model."""

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import plot_cmi_aix_timeline as base

HARDWARE = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
HW_LABELS = ("24c/1GPU", "24+24c/1GPU", "96c/4GPUs", "96+96c/4GPUs")
MODELS = ("mlp", "transformer")
MODES = ("coll", "predicted", "p2p1", "p2pfull")
LABELS = ("Collective", "Predicted (ideal)", "P2P (1)", "P2P (full)")
COLORS = ("#577787", "#C99449", "#9784B4", "#568775")
STAGES = ("send_input", "h2d", "forward", "d2h", "send_output")
T95 = 4.302652729911275
THESIS = Path("/home/ro092286/Master-Thesis/gitlab/thesis/chapters/chapter_optimizations.tex")
GUIDE = Path("/home/ro092286/Master-Thesis/notes/thesis-figures-guide.md")
WIDTH_MM = 150
AXIS_HEIGHT_MM = 3.15 * 25.4 * .52


def pipeline(stages, n):
    """A is collective aggregate work, tau=A/N; N includes the controller."""
    if len(stages) != 5 or not isinstance(n, int) or n < 1:
        raise ValueError("Expected five stages and positive integer WG membership")
    a = [base.number(v) for v in stages]
    total, largest = sum(a), max(a)
    predicted = total / n + (n - 1) / n * largest
    return dict(stage_sum_ms=total, stage_max_ms=largest, pipeline_ms=predicted,
                savings_ms=(1 - 1 / n) * (total - largest))


def collective_stages(row):
    if (row["is_controller"] != "1" or int(row["workgroup_rank"]) != 0
            or int(row["diagnostic_barriers"]) != 0
            or base.number(row["gather_arrival_wait_s"])
            or base.number(row["scatter_arrival_wait_s"])):
        raise ValueError("Expected controller, WG rank zero, and disabled arrival barriers")
    return [1000 * base.number(row["gather_collective_s"]),
            *[base.number(row[p + "_gpu_ms"]) for p in base.PHASES],
            1000 * base.number(row["scatter_collective_s"])]


def job_prediction(groups, observed_ms):
    """Independent zero-origin WG phase envelopes; never sum GPU savings."""
    if not groups:
        raise ValueError("Missing workgroups")
    before = max(g["stage_sum_ms"] for g in groups)
    after = max(g["pipeline_ms"] for g in groups)
    saving = before - after
    observed = base.number(observed_ms)
    predicted = observed - saving
    if predicted < 0:
        raise ValueError(f"Negative predicted full ML-step: {predicted} ms (no clamping)")
    return dict(before_envelope_ms=before, after_envelope_ms=after, savings_ms=saving,
                residual_ms=observed-before, predicted_ms=predicted,
                before_critical_wg=max(groups, key=lambda g: g["stage_sum_ms"])["workgroup"],
                after_critical_wg=max(groups, key=lambda g: g["pipeline_ms"])["workgroup"])


def summarize(values):
    if len(values) != 3:
        raise ValueError("Exactly three independent repetition means required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T95 * sd / math.sqrt(3)
    return dict(mean_ms=mean, sd_ms=sd, ci_low_ms=mean-half, ci_high_ms=mean+half,
                n_reps=3, calls_per_rep=5, **{f"rep{i}_ms": v for i, v in enumerate(values, 1)})


def collect(layouts=("default", "numa"), data_dir=base.DATA):
    runs = base.read_csv(data_dir / "runs.csv")
    inferences = base.read_csv(data_dir / "inferences.csv")
    report = json.loads((data_dir / "report.json").read_text())
    if (len(runs) != 288 or len({r["cell"] for r in runs}) != 288
            or len(inferences) != 2016 or report["errors"] or report["status"] != "ok"):
        raise ValueError("Expected clean canonical 288-run / 2016-inference snapshot")
    if not layouts or len(set(layouts)) != len(layouts) or set(layouts) - {"default", "affine", "numa"}:
        raise ValueError("Layouts must be unique default/affine/numa selections")
    paths = {data_dir / f for f in ("runs.csv", "inferences.csv", "report.json")}
    paths.update((Path(__file__), Path(base.__file__), THESIS, GUIDE))
    selected, observations, calls, predictions = [], [], [], []
    for model, hw, layout, rep in itertools.product(MODELS, HARDWARE, layouts, range(1, 4)):
        reference = None
        for mode in ("coll", "p2p1", "p2pfull"):
            run = base.unique([r for r in runs if
                (r["provider"], r["model"], r["placement"], r["layout"], r["rep"], r["comm"])
                == ("aix", model, hw, layout, str(rep), mode)], "canonical run")
            if (run["exit_status"] != "0" or run["solver_exit_status"] != "0"
                    or int(run["overlap"]) != (4 if model == "mlp" else 0)):
                raise ValueError("Unsuccessful run or wrong model overlap")
            selected.append(run)
            directory = base.ROOT / run["run_dir"]
            log = directory / "solver.log"
            paths.add(log)
            # Validate that diagnostics are from the canonical consolidation's selected job.
            provenance = report["provenance"][run["cell"]]
            if (provenance["jobid"] != run["jobid"] or provenance["run_dir"] != run["run_dir"]
                    or provenance["artifacts"]["solver.log"] != base.sha256(log)):
                raise ValueError("Canonical solver provenance mismatch")
            mapping = base.parse_mapping(log, run)
            members = [r for g in mapping.values() for r in g["members"]]
            signature = sorted((g["controller"], g["gpu"], tuple(g["members"])) for g in mapping.values())
            if (sorted(members) != list(range(int(run["solver_ranks"])))
                    or sorted(g["gpu"] for g in mapping.values()) != list(range(int(run["gpu_count"])))):
                raise ValueError("Incomplete/duplicate workgroup membership or GPU mapping")
            if reference is not None and signature != reference:
                raise ValueError("Collective/P2P workgroup mapping mismatch")
            reference = signature
            diagnostics = {}
            for wg, group in mapping.items():
                ctrl = group["controller"]
                name = f"aix_rank_{ctrl}.csv" if mode == "coll" else f"aix_p2p_timeline_rank_{ctrl}.csv"
                path = directory / "aix_gpu_diagnostics" / name
                paths.add(path)
                diagnostics[wg] = (path, base.read_csv(path))
            run_values = []
            for call in range(2, 7):
                inference = base.unique([r for r in inferences if r["cell"] == run["cell"]
                                        and int(r["call"]) == call+1], "matched inference")
                if (inference["phase"] != "measured" or int(inference["step"]) != base.STEPS[call]
                        or inference["jobid"] != run["jobid"]):
                    raise ValueError("Invalid measured-call identity")
                observed = base.number(inference["latency_ms"])
                run_values.append(observed)
                common = dict(cell=run["cell"], jobid=run["jobid"], model=model, placement=hw,
                              layout=layout, rep=rep, mode=mode, call=call, inference_call=call+1,
                              maia_step=base.STEPS[call], observed_mlstep_ms=observed)
                calls.append(dict(**common, value_ms=observed))
                group_rows = []
                for wg, group in sorted(mapping.items()):
                    path, rows = diagnostics[wg]
                    rows = [r for r in rows if int(r["call" if mode == "coll" else "step"]) == call]
                    if not rows or any(r["is_controller"] != "1" or int(r["world_rank"]) != group["controller"]
                                       or int(r["workgroup_rank"]) != 0 for r in rows):
                        raise ValueError(f"Missing/invalid controller diagnostics: {path}, call {call}")
                    raw = {k: "" for k in ("gather_collective_s", "device_cpu_s", "h2d_gpu_ms",
                        "forward_gpu_ms", "d2h_gpu_ms", "scatter_collective_s", "device_batches")}
                    adapted = {k: "" for k in (*[s+"_ms" for s in STAGES],
                        *["tau_"+s+"_ms" for s in STAGES], "stage_sum_ms", "stage_max_ms", "pipeline_ms", "savings_ms")}
                    if mode == "coll":
                        row = base.unique(rows, "collective controller call")
                        raw.update({k: row[k] for k in raw})
                        base.number(row["device_cpu_s"])
                        a = collective_stages(row)
                        gpu = a[1:4]
                        adapted.update(dict(zip((s+"_ms" for s in STAGES), a)))
                        adapted.update(dict(zip(("tau_"+s+"_ms" for s in STAGES),
                                                (v/len(group["members"]) for v in a))))
                        adapted.update(pipeline(a, len(group["members"])))
                    else:
                        intervals = base.pair_intervals(rows, call, group["controller"])
                        gpu = [sum(r["duration_ms"] for r in intervals if r["event"] == "torch_"+p)
                               for p in base.PHASES]
                    record = dict(**common, workgroup=wg, controller=group["controller"],
                        gpu_ordinal=group["gpu"], n_members=len(group["members"]),
                        members=";".join(map(str, group["members"])), source_path=str(path.relative_to(base.ROOT)),
                        **raw, **adapted, **{p+"_work_ms": v for p, v in zip(base.PHASES, gpu)},
                        gpu_work_sum_ms=sum(gpu))
                    observations.append(record)
                    group_rows.append(record)
                if mode == "coll":
                    prediction = job_prediction(group_rows, observed)
                    predictions.append(dict(**common, **prediction))
                    calls.append(dict(common, mode="predicted", value_ms=prediction["predicted_ms"]))
            if (run["n_measured"] != "5" or run_values != [float(v) for v in run["measured_ms"].split(";")]
                    or not math.isclose(statistics.mean(run_values), float(run["measured_mean_ms"]))):
                raise ValueError("Run/inference steady-call mismatch")
    values = []
    for model, hw, layout, mode in itertools.product(MODELS, HARDWARE, layouts, MODES):
        reps = []
        for rep in range(1, 4):
            rows = [r for r in calls if (r["model"], r["placement"], r["layout"], r["mode"], r["rep"])
                    == (model, hw, layout, mode, rep)]
            if sorted(r["call"] for r in rows) != list(range(2, 7)):
                raise ValueError("Incomplete five-call repetition")
            reps.append(statistics.mean(r["value_ms"] for r in rows))
        values.append(dict(model=model, placement=hw, layout=layout, mode=mode, **summarize(reps)))
    metadata = dict(canonical_runs=288, selected_runs=selected, layouts=list(layouts),
        sources={str(p): base.sha256(p) for p in sorted(paths)}, student_t_df2=T95,
        counts=dict(selected_runs=len(selected), workgroup_calls=len(observations),
                    call_values=len(calls), call_predictions=len(predictions), plot_values=len(values)),
        negative_predictions=0, negative_residuals=sum(p["residual_ms"] < 0 for p in predictions),
        negative_ci_bounds=sum(v["ci_low_ms"] < 0 for v in values))
    return observations, predictions, calls, values, metadata


def render(values, layouts, out):
    sys.path.insert(0, str(Path.home() / "Master-Thesis/gitlab/sourcecode"))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = []
    for layout in layouts:
        for models, name in ((MODELS, f"{layout}_combined_preview"),
                             (("mlp",), f"{layout}_mlp"), (("transformer",), f"{layout}_transformer")):
            preview = len(models) == 2
            top_mm = 19.2 if preview else 15
            height_mm = top_mm + AXIS_HEIGHT_MM + 12
            fig, axes = plt.subplots(1, len(models), squeeze=False,
                                     figsize=(WIDTH_MM/25.4, height_mm/25.4))
            for ax, model in zip(axes[0], models):
                subset = [v for v in values if v["layout"] == layout and v["model"] == model]
                for h, hw in enumerate(HARDWARE):
                    for m, mode in enumerate(MODES):
                        v = base.unique([v for v in subset if v["placement"] == hw and v["mode"] == mode], "bar")
                        x = h + (m-1.5)*.20
                        ax.bar(x, v["mean_ms"]/1000, width=.18, color=COLORS[m],
                               edgecolor=".25", linewidth=.4, hatch="///" if mode == "predicted" else None)
                        ax.errorbar(x, v["mean_ms"]/1000, yerr=(v["ci_high_ms"]-v["mean_ms"])/1000,
                                    color=".15", capsize=1.5, linewidth=.65)
                if preview:
                    ax.set_title(("MLP (overlap 4)" if model == "mlp" else "Transformer (overlap 0)")
                                 + f" / {layout}", fontsize=9, loc="left")
                ax.set_xticks(range(4), [s.replace("/", "/\n") for s in HW_LABELS])
                ax.tick_params(labelsize=6.5 if preview else 9)
                ax.set_ylabel("Full ML-step time (s)", fontsize=8 if preview else 10)
                low = min(0, min(v["ci_low_ms"] for v in subset)/1000)
                ax.set_ylim(low*1.05, max(v["ci_high_ms"] for v in subset)/1000*1.08)
                ax.set_xlim(-.55, 3.55)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
            handles = [Patch(facecolor=c, label=l, hatch="///" if i == 1 else None)
                       for i, (c, l) in enumerate(zip(COLORS, LABELS))]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99),
                       ncol=2, fontsize=8, handlelength=1.5)
            fig.subplots_adjust(left=.10, right=.985, bottom=12/height_mm,
                                top=1-top_mm/height_mm, wspace=.40)
            ts.save(fig, out / name)
            for suffix in ("pgf", "pdf", "png"):
                if not (out / f"{name}.{suffix}").is_file():
                    raise RuntimeError(f"Missing rendered {name}.{suffix}")
            names.append(name)
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=("default", "numa"), default=["default", "numa"])
    parser.add_argument("--data-dir", type=Path, default=base.DATA)
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "figures/ch7_pipeline_prediction_draft")
    args = parser.parse_args(argv)
    observations, predictions, calls, values, metadata = collect(args.layouts, args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = render(values, args.layouts, args.out_dir)
    for filename, rows in (("workgroup_call_stages.csv", observations), ("call_predictions.csv", predictions),
                           ("call_values.csv", calls), ("plot_values.csv", values)):
        base.write_csv(args.out_dir / filename, rows)
    metadata["figures"] = names
    metadata["figure_dimensions_mm"] = dict(width=WIDTH_MM, axes_height=AXIS_HEIGHT_MM,
        standalone_height=15+AXIS_HEIGHT_MM+12, preview_height=19.2+AXIS_HEIGHT_MM+12)
    metadata["sources"][str(Path.home()/"Master-Thesis/gitlab/sourcecode/thesis_style.py")] = base.sha256(
        Path.home()/"Master-Thesis/gitlab/sourcecode/thesis_style.py")
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    readme = r"""# Chapter 7 Pipeline Prediction Draft

## Definition
Thesis chapter_optimizations.tex lines 274--405, especially equations tau_step,
tau_max, collective and pipelined, is the model source (hashed in metadata.json).
For N workgroup members INCLUDING the controller (not GPU count),
tau_step = sum(tau), tau_max = max(tau), T_collective = N sum(tau),
T_pipeline = sum(tau) + (N-1) max(tau), savings = (N-1)(sum(tau)-max(tau)).
The synthetic N=10 example is 380 -> 218 ms.

## Collective Adapter
A_g = (1000 gather_collective_s, h2d_gpu_ms, forward_gpu_ms, d2h_gpu_ms,
1000 scatter_collective_s); tau_g = A_g/N_g.
Thus P_g = sum(A_g)/N_g + (N_g-1)/N_g max(A_g) and
savings_g = (1-1/N_g)(sum(A_g)-max(A_g)). No GPU-count scaling.
These are amortized hypothetical per-rank stages, NOT measured rank service times.
Only the mapped controller row is used; rank 0 is not an aggregate, and controller
identity is taken from base.parse_mapping(solver.log), never rank modulo GPU count.
The controller's local input is still counted in N, as in the thesis idealization.
Collective gather/scatter durations include protocol, waiting and possible imbalance:
amortizing them assumes baseline equivalence, homogeneous ranks and scalable service.
CUDA-event sums cover all serial collective sub-batches and exclude host overhead.
device_cpu_s is exported as a diagnostic ONLY: it is not added to the GPU stages,
and device_cpu minus GPU work is not assumed to be a precise CPU overhead measure.
GPU work in P2P can overlap; its phase SUM is diagnostic work, not elapsed time.
All modes' GPU diagnostics cover all 3 repetitions and all 5 measured calls.

## Job-Level Proxy
B = max_g sum(A_g), P = max_g P_g, savings_job = B-P.
predicted full ML-step = observed collective full ML-step - savings_job.
This is the difference of independent ZERO-ORIGIN WG phase envelopes, NOT a measured
global critical path or synchronized makespan. WG temporal origins are unknown;
the critical WG may switch (both identities exported). GPU savings are NEVER summed.
The observed residual R = observed collective - B is retained unchanged:
prediction = R + P. The observed baseline is never replaced by a stage sum.
Even negative residuals are retained and counted; they expose scope/instrumentation
differences rather than establish literal overhead. Negative predictions raise,
never clamp. This adapter is a conditional ideal scenario, not a guaranteed bound
on measured P2P speedup. It assumes homogeneous ranks, independent stages, unchanged
batch efficiency and non-stage residual, no inter-WG contention, and baseline
equivalence. It ignores the implementation's serialized input/output communication,
batch-size effects, local controller shortcuts and rank readiness differences.

## Scope and Statistics
Canonical consolidated 288 runs only; no retry-directory discovery. Successful AIx
default/numa coll,p2p1,p2pfull cells for MLP overlap 4 and Transformer overlap 0.
Each mode is paired by model, hardware, layout and repetition, with exact logged
WG/GPU membership equality. Diagnostic calls 2..6 map to canonical inference calls
3..7 and MAIA steps 73,102,131,160,189. First two calls are excluded.
Observed bars use canonical inferences.csv full MAIA ML-step elapsed, NOT controller
elapsed. Stages have controller-local single-call scope; no cross-clock rate or
throughput comparison is made. Hardware groups are not fixed-work strong scaling.
Prediction is computed per call, then five-call means per run, then across three
independent repetitions. CI = mean +/- 4.302652729911275 SD/sqrt(3), SD ddof=1,
95% Student t df=2. These CIs show repetition variability, not model uncertainty.
Calls and GPUs are not independent samples. Negative CI bounds are retained.
No repetition dots. Four modes times four hardware groups = 16 bars per model/layout.

## Outputs and Sizing
Each layout has a combined ONE-ROW preview (MLP left, Transformer right) plus two
independent panels intended for LaTeX. Every canvas is 150 mm wide; independent panels
are 68.6052 mm high and previews 72.8052 mm high. The measured plotting-axis height
remains 41.6052 mm; removed footer/title margins do not stretch the data area.
Independent panels use the FULL 150 mm thesis text width, not 73.5 mm half-width;
the combined preview necessarily uses smaller ticks and is for overview only.
Use independent PDFs at width=\textwidth; PGF requires the thesis Latin Modern
preamble. thesis_style supplies PGF + pdflatex PDF + PDF-derived 300 dpi PNG.
No tight bounding-box cropping, so physical width stays invariant.
Independent panels have no embedded titles or subcaptions in PNG, PDF or PGF.
Supply model, overlap and layout in LaTeX subcaptions: MLP overlap 4, Transformer
overlap 0; filenames identify default/numa layout. Combined previews retain only
these model/overlap/layout headers for identification, without (a)/(b) labels.
Statistical explanations and prediction-proxy caveats appear only in captions/README,
never inside graphics. The predicted-series legend and hatch remain data identifiers.

## Font Style
All formats use thesis_style's pdflatex/Latin Modern typography, without enlarging
fonts to fill removed margins. Independent ticks are 9 pt and axis labels 10 pt;
overview ticks are 6.5 pt, labels 8 pt and model headers 9 pt. Legends remain 8 pt.

workgroup_call_stages.csv: raw collective inputs, adapted A/tau, N, mapping,
all-mode GPU phase work and diagnostic source paths, one row per WG/call.
call_predictions.csv: before/after envelopes, critical WG switch, residual,
savings and observed/predicted full-step per collective call.
call_values.csv: observed and predicted call-level values for all bars.
plot_values.csv: three run means, SD, CI and plotted means (all in ms).
metadata.json: selected canonical rows, counts, SHA256 of all consumed inputs,
controller diagnostics, mappings, thesis, guide, helper and script/style sources.

## Gallery
"""
    for name in names:
        readme += f"\n### {name}\n[PDF]({name}.pdf) | [PGF]({name}.pgf)\n\n![{name}]({name}.png)\n"
    readme += "\n## Validation Counts\n```json\n" + json.dumps(metadata["counts"], indent=2) + "\n```\n"
    readme += (f"\nNegative residuals retained: {metadata['negative_residuals']}; negative CI bounds retained: "
               f"{metadata['negative_ci_bounds']}; negative predictions: 0.\n")
    (args.out_dir / "README.md").write_text(readme)
    print(json.dumps(dict(figures=len(names), artifacts=3*len(names), **metadata["counts"])))


if __name__ == "__main__":
    main()
