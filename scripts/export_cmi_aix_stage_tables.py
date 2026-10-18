#!/usr/bin/env python3
"""Export three auditable AIx stage tables; never render or change figures."""

import argparse
import itertools
import json
from pathlib import Path
import statistics

import plot_cmi_ch7_pipeline_prediction as prediction

base = prediction.base
LAYOUTS = ("default", "affine", "numa")
STAGE_FIELDS = tuple(s + "_ms" for s in prediction.STAGES)
METRICS = (*STAGE_FIELDS, "stage_sum_ms", "collective_mlstep_ms",
           "c1_ideal_pipeline_ms", "c1_ideal_mlstep_ms",
           "full_ideal_pipeline_ms", "full_ideal_mlstep_ms",
           "c1_measured_mlstep_ms", "full_measured_mlstep_ms",
           "c1_ready_range_window_ms", "full_ready_range_window_ms")
HEADERS = ("Model", "Hardware", "Send input", "H2D", "Forward", "D2H", "Send output",
           "Stage sum", "Coll. ML step", "c1 ideal pipeline", "c1 ideal ML step",
           "Full ideal pipeline", "Full ideal ML step", "c1 measured ML step",
           "Full measured ML step", "c1 ready-range window", "Full ready-range window")
NOTES = """All durations are in milliseconds. Each row averages three independent
repetition means, each from five measured calls. c1 and full-credit ideal columns
are aliases of ONE COMMON credit-independent prediction, not separate models.
Measured ML-step times are canonical MAIA full-step latencies. Ready-range windows
are measured controller-local spans, NOT five-stage elapsed times or global
makespans. See README.md for definitions, assumptions, and uncertainty.
"""


def ready_range_window(intervals):
    """Subtract on each controller's clock before taking any workgroup maximum."""
    ranges = [r for r in intervals if r["event"] == "range_inference"]
    if not ranges:
        raise ValueError("Missing ready-range intervals")
    return 1000 * (max(r["end_s"] for r in ranges) - min(r["start_s"] for r in ranges))


def collect(data_dir=base.DATA):
    observations, predictions, _, _, metadata = prediction.collect(LAYOUTS, data_dir)
    groups = {}
    cache = {}
    for row in observations:
        groups.setdefault((row["cell"], row["call"]), []).append(row)
        row["ready_range_window_ms"] = ""
        row["controller_mlstep_ms"] = ""
        if row["mode"] != "coll":
            path = base.ROOT / row["source_path"]
            if path not in cache:
                cache[path] = base.read_csv(path)
            rows = [r for r in cache[path] if int(r["step"]) == row["call"]]
            intervals = base.pair_intervals(rows, row["call"], row["controller"])
            row["ready_range_window_ms"] = ready_range_window(intervals)
            row["controller_mlstep_ms"] = base.unique(
                [r for r in intervals if r["event"] == "ml_step"], "controller ML step")["duration_ms"]

    runs = {(r["model"], r["placement"], r["layout"], int(r["rep"]), r["comm"]): r
            for r in metadata["selected_runs"]}
    call_rows = []
    for p in predictions:
        key = tuple(p[k] for k in ("model", "placement", "layout", "rep"))
        critical = base.unique([g for g in groups[p["cell"], p["call"]]
                                if g["workgroup"] == p["before_critical_wg"]], "critical collective WG")
        row = {k: p[k] for k in ("model", "placement", "layout", "rep", "call",
                                 "inference_call", "maia_step")}
        row.update(collective_cell=p["cell"], collective_jobid=p["jobid"],
                   before_critical_wg=p["before_critical_wg"], after_critical_wg=p["after_critical_wg"],
                   residual_ms=p["residual_ms"], savings_ms=p["savings_ms"],
                   **{k: critical[k] for k in STAGE_FIELDS}, stage_sum_ms=p["before_envelope_ms"],
                   collective_mlstep_ms=p["observed_mlstep_ms"])
        for mode, prefix in (("p2p1", "c1"), ("p2pfull", "full")):
            run = runs[(*key, mode)]
            selected = groups[run["cell"], p["call"]]
            row.update({prefix + "_cell": run["cell"], prefix + "_jobid": run["jobid"],
                        prefix + "_ideal_pipeline_ms": p["after_envelope_ms"],
                        prefix + "_ideal_mlstep_ms": p["predicted_ms"],
                        prefix + "_measured_mlstep_ms": selected[0]["observed_mlstep_ms"],
                        prefix + "_ready_range_window_ms": max(g["ready_range_window_ms"] for g in selected),
                        prefix + "_controller_mlstep_envelope_ms": max(g["controller_mlstep_ms"] for g in selected)})
        call_rows.append(row)

    tables, summaries = [], []
    for layout, model, hw in itertools.product(LAYOUTS, prediction.MODELS, prediction.HARDWARE):
        selected = [r for r in call_rows if (r["layout"], r["model"], r["placement"]) == (layout, model, hw)]
        table = dict(layout=layout, model=model, placement=hw)
        for metric in METRICS:
            reps = []
            for rep in range(1, 4):
                calls = [r for r in selected if r["rep"] == rep]
                if sorted(r["call"] for r in calls) != list(range(2, 7)):
                    raise ValueError("Expected five measured calls in each of three repetitions")
                reps.append(statistics.mean(r[metric] for r in calls))
            summary = prediction.summarize(reps)
            summaries.append(dict(layout=layout, model=model, placement=hw, metric=metric, **summary))
            table[metric] = summary["mean_ms"]
        tables.append(table)
    metadata["sources"][str(Path(__file__))] = base.sha256(Path(__file__))
    metadata["table_counts"] = dict(layouts=3, rows_per_layout=8, call_rows=len(call_rows),
                                    summary_rows=len(summaries), workgroup_calls=len(observations))
    metadata["diagnostics"] = dict(
        negative_residuals=sum(p["residual_ms"] < 0 for p in predictions),
        residual_min_ms=min(p["residual_ms"] for p in predictions),
        residual_max_ms=max(p["residual_ms"] for p in predictions),
        critical_wg_switches=sum(p["before_critical_wg"] != p["after_critical_wg"] for p in predictions),
        negative_summary_ci_bounds=sum(s["ci_low_ms"] < 0 for s in summaries),
        max_repetition_cv=max(s["sd_ms"] / s["mean_ms"] for s in summaries if s["mean_ms"] > 0),
        negative_ci_metrics=[{k: s[k] for k in ("layout", "model", "placement", "metric",
                                               "mean_ms", "sd_ms", "ci_low_ms", "ci_high_ms")}
                             for s in summaries if s["ci_low_ms"] < 0])
    return tables, summaries, call_rows, observations, predictions, metadata


def render(rows, layout, out):
    """One eight-row table per format; wide LaTeX uses a landscape page."""
    selected = [r for r in rows if r["layout"] == layout]
    if len(selected) != 8:
        raise ValueError("Expected eight model/hardware rows per layout")
    base.write_csv(out / f"{layout}.csv", selected)
    body = [[r["model"], prediction.HW_LABELS[prediction.HARDWARE.index(r["placement"])],
             *[f"{r[k]:.3f}" for k in METRICS]] for r in selected]
    markdown = f"# AIx {layout} Stage Table\n\n{NOTES}\n"
    markdown += "| " + " | ".join(HEADERS) + " |\n| " + " | ".join(["---"] * len(HEADERS)) + " |\n"
    markdown += "".join("| " + " | ".join(r) + " |\n" for r in body)
    (out / f"{layout}.md").write_text(markdown)
    latex = ("% Requires booktabs, graphicx, pdflscape. All values in ms.\n"
             "\\begin{landscape}\n\\begin{table}[p]\n\\centering\n"
             f"\\caption{{AIx {layout}: collective stages, shared credit-independent ideal predictions, "
             "and measured P2P timings (ms).}\n"
             f"\\label{{tab:aix-stages-{layout}}}\n"
             "\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n"
             "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{ll" + "r" * len(METRICS) + "}\n\\toprule\n")
    latex_headers = ["\\shortstack{" + h.replace(" ", "\\\\") + "}" for h in HEADERS]
    latex += " & ".join(latex_headers) + " \\\\\n\\midrule\n"
    latex += "".join(" & ".join(r) + " \\\\\n" for r in body)
    latex += ("\\bottomrule\n\\end{tabular}%\n}\n\\par\\smallskip\n"
              "\\begin{minipage}{\\linewidth}\\footnotesize\n" + NOTES.replace("README.md", "the accompanying README") +
              "\\end{minipage}\n\\end{table}\n\\end{landscape}\n")
    (out / f"{layout}.tex").write_text(latex)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.DATA)
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "results/cmi_aix_stage_tables")
    args = parser.parse_args(argv)
    tables, summaries, calls, observations, predictions, metadata = collect(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        render(tables, layout, args.out_dir)
    for name, rows in (("summary_statistics.csv", summaries), ("call_metrics.csv", calls),
                       ("workgroup_call_stages.csv", observations), ("call_predictions.csv", predictions)):
        base.write_csv(args.out_dir / name, rows)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    readme = """# AIx Stage Tables

Reproduce with `python scripts/export_cmi_aix_stage_tables.py`. No jobs are submitted;
no figures or thesis files are changed. Each of default, affine, numa has one
eight-row table (two models x four hardware placements), in CSV, Markdown and LaTeX.
LaTeX snippets require booktabs, graphicx and pdflscape; wide tables use landscape.

## Reading the Tables

""" + NOTES + """
CSV preserves full precision; presentation formats round to 0.001 ms. Each stage
comes from the SAME collective workgroup selected per call by argmax sum(A).
Thus the five displayed stage means sum to the displayed stage-sum mean (up to
presentation rounding). They are NOT independent per-stage workgroup maxima.
The selected workgroup can change across calls. Full collective ML-step remains
the observed canonical latency, NOT the five-stage sum.

## Exact Model and Adapter

Source: thesis chapter_optimizations.tex lines 274--405, equations tau_step,
tau_max, collective and pipelined (source SHA256 in metadata.json).
N is workgroup membership INCLUDING its controller, never GPU count.
tau_step = sum(tau); tau_max = max(tau);
T_collective = N sum(tau); T_pipeline = sum(tau) + (N-1) max(tau).
The N=10 synthetic example is 380 -> 218 ms.

For each collective controller, A = (1000*gather_collective_s, h2d_gpu_ms,
forward_gpu_ms, d2h_gpu_ms, 1000*scatter_collective_s), tau = A/N.
P_g = sum(A_g)/N_g + (N_g-1)/N_g max(A_g).
B = max_g sum(A_g); P = max_g P_g; R = observed collective ML step - B.
Ideal pipeline = P; ideal full ML step = R + P = observed collective - (B-P).
Before/after maxima are independent ZERO-ORIGIN workgroup envelopes; the critical
workgroup may switch. Never sum GPU savings or substitute stage sum for ML step.
The c1 and full ideal columns deliberately repeat P and R+P: the thesis ideal
formula has NO credit term. They are not credit-specific performance forecasts.

Homogeneous ranks, baseline equivalence, independent stages, unchanged batch
efficiency and non-stage residual, and no inter-workgroup contention are assumed.
Gather/scatter include protocol/waiting/imbalance. GPU stages are CUDA-event sums
over serial collective sub-batches; device_cpu_s is diagnostic, never added.
Negative residuals are retained as scope/instrumentation discrepancies, NOT
literal negative overhead. Negative predictions raise rather than clamp. This
conditional ideal proxy is not a strict bound on observed P2P speedup.

## Measured Pipeline Window

For each P2P controller/call, ready-range window = 1000 *
(last range_inference_end - first range_inference_start). Take the maximum of
these durations across workgroups only AFTER local subtraction, independent of
clock origin. This is a ready-range inference envelope, NOT true five-stage
elapsed time: it omits input readiness before the first range and may omit
communication outside the range events. It is not a synchronized global span.
It includes gaps between ranges. Controller ML-step envelope (max of locally
measured ml_step intervals) is additionally exported in call_metrics.csv and
per-workgroup controller elapsed in workgroup_call_stages.csv; neither replaces
the canonical full MAIA ML-step timing.

## Sampling and Audit

Canonical consolidated snapshot only: 288 runs, 2016 inferences, clean report.
Select 216 successful AIx runs across all layouts, all three repetitions, and
coll/p2p1/p2pfull; MLP overlap 4 and Transformer overlap 0. Provenance hashes and
exact logged workgroup/GPU memberships are validated by the prediction collector.
Diagnostic calls 2..6 match inference calls 3..7 (MAIA steps 73,102,131,160,189).
Compute per-call predictions/maxima BEFORE five-call run means, then average
three independent repetitions. Hardware rows are not fixed-work strong scaling.

summary_statistics.csv supplies all metric means, three run means, sample SD,
and 95% Student t CI (df=2, t=4.302652729911275). Calls and GPUs are not independent
samples. CIs quantify repetition variability, not model uncertainty; negative
CI bounds are retained. call_metrics.csv holds 360 paired collective/P2P calls;
call_predictions.csv preserves per-call before/after identities and residuals;
workgroup_call_stages.csv holds raw/adapted stages, mapping, sources, and P2P local
windows for every WG/call. metadata.json hashes all consumed sources and records
selected canonical runs. No cross-controller absolute timestamps are subtracted.

## Observed Diagnostics

"""
    readme += "```json\n" + json.dumps(metadata["diagnostics"], indent=2) + "\n```\n"
    readme += ("\nThe maximum coefficient of variation above is sample SD / mean across the\n"
               "three repetition means, over all exported table metrics. Negative residuals\n"
               "and wide/negative CIs are reported, not hidden or clamped.\n")
    (args.out_dir / "README.md").write_text(readme)
    print(json.dumps(metadata["table_counts"]))


if __name__ == "__main__":
    main()
