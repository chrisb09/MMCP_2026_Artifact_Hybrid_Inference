#!/usr/bin/env python3
"""Five-stage collective work versus ideal pipeline and local exchange envelopes."""

import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys

import plot_cmi_ch7_pipeline_prediction as prediction

base = prediction.base
LAYOUTS = ("default", "affine", "numa")
STAGE_FIELDS = tuple(s + "_ms" for s in prediction.STAGES)
METRICS = (*STAGE_FIELDS, "collective_ms", "c1_ideal_ms", "c1_measured_ms",
           "full_ideal_ms", "full_measured_ms")
SOURCE = base.ROOT / "CPP-ML-Interface/extern/AIxeleratorService/src/communicationStrategy/collectiveCommunication.cpp"
STYLE = Path.home() / "Master-Thesis/gitlab/sourcecode/thesis_style.py"


def exchange_span(rows, n, controller):
    """All endpoints belong to ONE controller/call; never align worker clocks."""
    if n < 2 or not rows or any(int(r["world_rank"]) != controller or
            r["is_controller"] != "1" or int(r["workgroup_rank"]) != 0 for r in rows):
        raise ValueError("Expected one mapped controller and at least two members")
    if len({r["step"] for r in rows}) != 1:
        raise ValueError("Mixed calls")
    start = base.unique([r for r in rows if r["event"] == "input_listen_start"], "input listen start")
    ml_start = base.unique([r for r in rows if r["event"] == "ml_step_start"], "ML start")
    ml_end = base.unique([r for r in rows if r["event"] == "ml_step_end"], "ML end")

    def peers(event, expected):
        selected = [r for r in rows if r["event"] == event]
        if sorted(int(r["peer_workgroup_rank"]) for r in selected) != list(expected):
            raise ValueError(f"Missing/duplicate {event} membership")
        return {int(r["peer_workgroup_rank"]): r for r in selected}

    grants = peers("input_credit_send", range(1, n))
    ready = peers("input_ready", range(n))
    outputs = peers("result_send_start", range(n))
    done = peers("result_send_complete", range(1, n))
    end = max(done.values(), key=lambda r: base.number(r["time_s"]))
    t0, t1 = (base.number(r["time_s"]) for r in (start, end))
    if not base.number(ml_start["time_s"]) <= t0 <= t1 <= base.number(ml_end["time_s"]):
        raise ValueError("Invalid exchange endpoint order")
    for rank in range(n):
        times = [t0]
        if rank:
            times.append(base.number(grants[rank]["time_s"]))
        times.extend(base.number(r["time_s"]) for r in (ready[rank], outputs[rank]))
        if rank:
            times.append(base.number(done[rank]["time_s"]))
        times.append(t1)
        if times != sorted(times):
            raise ValueError("Invalid per-peer exchange order")
    intervals = base.pair_intervals(rows, int(start["step"]), controller)
    covered = []
    for r in intervals:
        if r["event"] == "range_inference":
            first, stop = int(r["range_first_rank"]), int(r["range_end_rank"])
            if not 0 <= first < stop <= n or not t0 <= r["start_s"] <= r["end_s"] <= t1:
                raise ValueError("Invalid inference range")
            covered.extend(range(first, stop))
    if sorted(covered) != list(range(n)):
        raise ValueError("Missing/duplicate inference membership")
    # CUDA phase timestamps are host-anchored estimates, not a second clock to align.
    return dict(exchange_ms=1000 * (t1-t0), start_s=t0, end_s=t1,
                start_event=start["event"], end_event=end["event"],
                start_identity=json.dumps(start, sort_keys=True),
                end_identity=json.dumps(end, sort_keys=True),
                input_members=n, inference_members=len(covered), output_members=n,
                range_requests=sum(r["event"] == "range_inference" for r in intervals))


def collect(layouts=LAYOUTS, data_dir=base.DATA):
    observations, predictions, _, _, metadata = prediction.collect(layouts, data_dir)
    groups, cache = {}, {}
    for g in observations:
        groups.setdefault((g["cell"], g["call"]), []).append(g)
        span = dict(exchange_ms="", start_s="", end_s="", start_event="", end_event="",
                    start_identity="", end_identity="", input_members="", inference_members="",
                    output_members="", range_requests="")
        if g["mode"] != "coll":
            path = base.ROOT / g["source_path"]
            if path not in cache:
                cache[path] = base.read_csv(path)
            rows = [r for r in cache[path] if int(r["step"]) == g["call"]]
            span = exchange_span(rows, g["n_members"], g["controller"])
        g.update(span)
    runs = {(r["model"], r["placement"], r["layout"], int(r["rep"]), r["comm"]): r
            for r in metadata["selected_runs"]}
    calls = []
    for p in predictions:
        identity = tuple(p[k] for k in ("model", "placement", "layout", "rep"))
        critical = base.unique([g for g in groups[p["cell"], p["call"]]
                               if g["workgroup"] == p["before_critical_wg"]], "collective critical WG")
        row = {k: p[k] for k in ("model", "placement", "layout", "rep", "call", "inference_call", "maia_step")}
        row.update(collective_cell=p["cell"], collective_jobid=p["jobid"],
                   collective_ms=p["before_envelope_ms"], ideal_ms=p["after_envelope_ms"],
                   before_critical_wg=p["before_critical_wg"], after_critical_wg=p["after_critical_wg"],
                   **{k: critical[k] for k in STAGE_FIELDS})
        for mode, prefix in (("p2p1", "c1"), ("p2pfull", "full")):
            run = runs[(*identity, mode)]
            measured = max(groups[run["cell"], p["call"]], key=lambda g: g["exchange_ms"])
            row.update({prefix + "_cell": run["cell"], prefix + "_jobid": run["jobid"],
                        prefix + "_ideal_ms": p["after_envelope_ms"],
                        prefix + "_measured_ms": measured["exchange_ms"],
                        prefix + "_critical_wg": measured["workgroup"]})
        calls.append(row)
    summaries = []
    for layout, model, hw, metric in itertools.product(layouts, prediction.MODELS, prediction.HARDWARE, METRICS):
        selected = [r for r in calls if (r["layout"], r["model"], r["placement"]) == (layout, model, hw)]
        reps = []
        for rep in range(1, 4):
            values = [r for r in selected if r["rep"] == rep]
            if sorted(r["call"] for r in values) != list(range(2, 7)):
                raise ValueError("Expected five measured calls per repetition")
            reps.append(statistics.mean(r[metric] for r in values))
        summaries.append(dict(layout=layout, model=model, placement=hw, metric=metric,
                              **prediction.summarize(reps)))
    for path in (Path(__file__), SOURCE):
        metadata["sources"][str(path)] = base.sha256(path)
    # Do not inherit the other generator's full-ML-step plot counts/diagnostics.
    metadata = {k: metadata[k] for k in ("canonical_runs", "selected_runs", "layouts", "sources", "student_t_df2")}
    metadata.update(counts=dict(selected_runs=len(runs), workgroup_calls=len(observations),
                               measured_workgroup_calls=sum(g["mode"] != "coll" for g in observations),
                               paired_calls=len(calls), summary_rows=len(summaries)),
                    negative_ci_bounds=sum(s["ci_low_ms"] < 0 for s in summaries),
                    critical_wg_switches=sum(p["before_critical_wg"] != p["after_critical_wg"] for p in predictions),
                    measured_definition="controller input_listen_start -> last result_send_complete after output Waitall",
                    model_definition="P=max_g(sum(A_g)/N_g+(N_g-1)/N_g*max(A_g)); no credit term; no residual")
    return observations, calls, summaries, metadata


def render(summaries, layouts, out):
    sys.path.insert(0, str(STYLE.parent))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    colors = ("#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2")
    names = []
    for layout, model, hw in itertools.product(layouts, prediction.MODELS, prediction.HARDWARE):
        values = {s["metric"]: s for s in summaries if
                  (s["layout"], s["model"], s["placement"]) == (layout, model, hw)}
        height_mm = 20.592 + 5 + 8
        fig, (left, right) = plt.subplots(1, 2, figsize=(150/25.4, height_mm/25.4),
                                         gridspec_kw={"width_ratios": [45, 55]})
        for x, (metric, color) in enumerate(zip(STAGE_FIELDS, colors)):
            v = values[metric]
            left.bar(x, v["mean_ms"]/1000, width=.64, color=color, edgecolor=".25", linewidth=.35)
            left.errorbar(x, v["mean_ms"]/1000, yerr=(v["ci_high_ms"]-v["mean_ms"])/1000,
                          color=".15", capsize=2, linewidth=.7)
            left.annotate(f"{v['mean_ms']/1000:.3g}", (x, v["ci_high_ms"]/1000),
                          xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)
        left.set_xticks(range(5), ["Gather", "H2D", "Fwd", "D2H", "Scatter"])
        left.set_xlim(-.65, 4.65)
        metrics = ("collective_ms", "c1_ideal_ms", "c1_measured_ms", "full_ideal_ms", "full_measured_ms")
        positions = (0, 1.4, 2.4, 3.9, 4.9)
        for x, metric in zip(positions, metrics):
            v = values[metric]
            ideal = "ideal" in metric
            right.bar(x, v["mean_ms"]/1000, width=.64,
                      color="#C99449" if ideal else ("#577787" if metric == "collective_ms" else "#568775"),
                      hatch="///" if ideal else None, edgecolor=".25", linewidth=.4)
            right.errorbar(x, v["mean_ms"]/1000, yerr=(v["ci_high_ms"]-v["mean_ms"])/1000,
                           color=".15", capsize=2, linewidth=.7)
            right.annotate(f"{v['mean_ms']/1000:.3g}", (x, v["ci_high_ms"]/1000),
                           xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)
        right.set_xticks((0, 1.9, 4.4), ["Collective\n5-stage sum", "P2P\nCredit=1", "P2P\nFullcredit"])
        for ax, fields in ((left, STAGE_FIELDS), (right, metrics)):
            low = min(0, *(values[m]["ci_low_ms"]/1000 for m in fields))
            high = max(values[m]["ci_high_ms"]/1000 for m in fields)
            ax.set_ylim(low*1.05, high+.35*(high-low))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=6.5 if ax is right else 7)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.grid(False, axis="x")
            ax.set_axisbelow(True)
        left.set_ylabel("Phase time (s)", fontsize=9)
        right.set_ylabel("Inference time (s)", fontsize=9)
        fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors,
                   ("Gather", "H2D", "Forward", "D2H", "Scatter"))],
                   loc="upper left", bbox_to_anchor=(.105, .985), ncol=5, fontsize=5.5, handlelength=.8,
                   handleheight=.6, columnspacing=.7, handletextpad=.4, borderaxespad=0)
        fig.legend(handles=[Patch(facecolor="#577787", label="Collective 5-stage sum"),
                             Patch(facecolor="#C99449", hatch="///", edgecolor=".25", label="Predicted"),
                             Patch(facecolor="#568775", label="Measured")],
                   loc="upper right", bbox_to_anchor=(.985, .985), ncol=3, fontsize=5.5, handlelength=.8,
                   handleheight=.6, columnspacing=.7, handletextpad=.4, borderaxespad=0)
        fig.subplots_adjust(left=.105, right=.985, bottom=8/height_mm, top=1-5/height_mm, wspace=.30)
        name = f"{model}_{hw}_{layout}_pipeline_spans"
        ts.save(fig, out / name)
        for suffix in ("pgf", "pdf", "png"):
            if not (out / f"{name}.{suffix}").is_file():
                raise RuntimeError(f"Missing {name}.{suffix}")
        names.append(name)
        # Author independent subfigures at their inclusion width, without captions.
        fig.set_size_inches(73.5/25.4, height_mm/25.4)
        for index, (axis, suffix) in enumerate(((left, "stages"), (right, "comparison"))):
            for other in (left, right):
                other.set_visible(other is axis)
            axis.set_position((13/73.5, 8/height_mm, 1-15/73.5, 20.592/height_mm))
            for i, legend in enumerate(fig.legends):
                legend.set_visible(i == index)
                legend.set_bbox_to_anchor((13/73.5 if i == 0 else 1-2/73.5, .985))
            ts.save(fig, out / "panels" / f"{name}_{suffix}")
    return names


README = """# AIx Pipeline Spans Draft

Reproduce: `python scripts/plot_cmi_pipeline_spans.py`. All three layouts by default;
use `--layouts default affine numa` or a subset. No jobs, data edits, or commits.
Each model/hardware/layout has a 150 x 33.592 mm two-panel PGF/PDF/PNG figure.
Independent 73.5 x 33.592 mm subfigures are exported under panels/ as
<name>_stages and <name>_comparison, also in all three formats.
All exports are caption-free: no overall title, (a)/(b) headings, or explanatory
footers. Model, overlap, hardware and layout are identified by filenames and the
gallery headings so LaTeX can own captions. Interpretation and statistics below
belong in the accompanying thesis text or caption, not inside the graphic.
Plotting areas remain 20.592 mm high (60% of their original height). Both compact legends share one
row, left- and right-aligned respectively. Every bar is labeled with its mean in
seconds above the upper CI endpoint; labels are rounded to three significant figures.
The first eight gallery entries are the default-layout main figures.

## Definitions

Left: five collective stages from the SAME per-call WG selected by argmax sum(A),
consistent with export_cmi_aix_stage_tables.py, not independent stage maxima.
Five adjacent, zero-baseline columns show individual phase means and their own
95% repetition CIs, NOT stacked heights or the stage-sum CI. Short ticks identify
Gather, H2D, Fwd, D2H, Scatter; the legend spells out the phases. Colors match the
chapter-6 CPU phase palette: #4C78A8, #72B7B2, #F58518, #54A24B, #B279A2.
Panels use independent y scales: phase durations at left, full envelopes at right.
The reader-facing right-axis label is Inference time (s), including transfers;
the precise measured exchange scope remains defined below, not pure forward time.
The panel width ratio is 45:55. Right-hand groups are Collective 5-stage sum,
P2P Credit=1, and P2P Fullcredit; legend colors distinguish Predicted and Measured.
A = (1000*gather_collective_s, h2d_gpu_ms, forward_gpu_ms, d2h_gpu_ms,
1000*scatter_collective_s). Their means sum to the collective reference at right.
Gather/scatter are host-clock durations including wait/protocol/imbalance; the GPU
phases are CUDA-event sums over serial sub-batches. This mixed-instrument work sum
is not a directly measured five-stage elapsed interval. device_cpu_s is not added.

Right: collective B=max_g sum(A_g), then c1 ideal/measured and full ideal/measured.
Thesis chapter_optimizations.tex equations tau_step, tau_max, collective, pipelined:
P_g=sum(A_g)/N_g+(N_g-1)/N_g*max(A_g); P=max_g P_g. N includes the controller,
not GPU count. The ideal has NO credit term: the two ideal bars repeat exactly P,
not credit-specific forecasts. No residual is added, and no full MAIA ML-step time
is plotted. The predicted critical WG can differ from the collective critical WG.
Assumptions: rank homogeneity, operation independence, baseline equivalence,
unchanged batch efficiency, and no inter-WG contention. This is a conditional ideal
proxy, not a guaranteed bound on measured performance.

## Measured Exchange

Per controller/call: input_listen_start -> last result_send_complete. The former
is logged before receive posting (collectiveCommunication.cpp lines 296--312),
before credit grants (329--331), and before local input_ready (348--350). It is the
earliest explicit controller input activity, not a remote sender start timestamp.
The latter is logged per remote rank AFTER output MPI_Waitall (662--666), not at
individual worker receipt. The last log includes completion-marker loop overhead.
Local rank zero has input_ready and result_send_start but no result_send_complete;
its output is already local. We validate all N inputs, exactly-once inference-range
coverage (and paired H2D/forward/D2H request identities), all N output posts, and
N-1 grants/completions. No worker or controller-local request is dropped.
The local segment is already available; no local input-copy start is instrumented.
This envelope spans input/inference/output activity INCLUDING setup, scheduling,
waiting, logging and output completion protocol. It excludes initial function
setup before input_listen_start and final credit/ACK Waitalls after output done.
It is not pure stage service time, not a ready-range-only window, not the canonical
MAIA step, and not a synchronized global first-input-to-last-worker-receipt span.
There is no fallback to a silently broader ml_step interval: missing endpoints fail.

Subtract endpoints on EACH controller's own clock first, then take the maximum
local duration across WGs. This independent zero-origin envelope is not a global
makespan. No absolute timestamps from different clocks are subtracted, and no
causal clock alignment is assumed. Host-anchored CUDA timestamps are not used as
exchange endpoints. Comparing the mixed host/CUDA collective sum and ideal with
this host exchange is useful but instrument/scope differences remain explicit.

## Sampling And Audit

Canonical consolidated snapshot only: 288 runs / 2016 inference rows. All layouts
select 216 successful AIx runs, coll/c1/full, 2 models x 4 placements x 3 reps.
MLP ov4, Transformer ov0. Diagnostic calls 2..6 match inference calls 3..7 and MAIA
steps 73,102,131,160,189. Exact solver-log WG/GPU membership is validated across modes.
Per-call maxima/predictions precede five-call repetition means. Three independent
repetition means yield mean +/- 4.302652729911275*sample_SD/sqrt(3), 95% Student t,
df=2. Calls/GPUs are not independent samples. CIs express repetition variability,
not model uncertainty. Negative CI bounds remain visible, not clamped. Each left
CI is for that phase; right CIs are for envelope metrics. Hardware is not fixed-work scaling.

workgroup_calls.csv contains every WG/call with raw stages, memberships, sources,
and measured endpoints plus complete serialized endpoint row identities.
call_metrics.csv contains paired per-call reference/predictions/measurements and
all critical WG identities. summary_statistics.csv contains means, all three run
means, SD and CIs. metadata.json hashes consumed canonical files, diagnostics,
solver logs, thesis, guide, and code/style sources; hashes are recomputed on each run.
No existing exported CSVs, old figures, or metadata are used as cached results.

## Gallery
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--data-dir", type=Path, default=base.DATA)
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "figures/aix_pipeline_spans_draft")
    args = parser.parse_args(argv)
    observations, calls, summaries, metadata = collect(args.layouts, args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = render(summaries, args.layouts, args.out_dir)
    for name, rows in (("workgroup_calls.csv", observations), ("call_metrics.csv", calls),
                       ("summary_statistics.csv", summaries)):
        base.write_csv(args.out_dir / name, rows)
    metadata["figures"] = names
    metadata["sources"][str(STYLE)] = base.sha256(STYLE)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    gallery = sorted(names, key=lambda n: ("_default_" not in n, names.index(n)))
    text = README + "".join(f"\n### {name}\n[PDF]({name}.pdf) | [PGF]({name}.pgf)\n\n![{name}]({name}.png)\n"
                           for name in gallery)
    text += "\n## Coverage\n```json\n" + json.dumps(metadata["counts"], indent=2) + "\n```\n"
    (args.out_dir / "README.md").write_text(text)
    print(json.dumps(dict(figures=len(names), **metadata["counts"])))


if __name__ == "__main__":
    main()
