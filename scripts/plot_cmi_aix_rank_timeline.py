#!/usr/bin/env python3
"""24c1g aggregate and all-rank AIx timelines from selected measured calls."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import plot_cmi_aix_timeline as base
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SOURCE = base.ROOT / "CPP-ML-Interface/extern/AIxeleratorService/src/communicationStrategy/collectiveCommunication.cpp"
FAMILIES = {
    "ml_step": ("ml_step_start", "ml_step_end"),
    "credit_wait": ("input_credit_wait_start", "input_send_start"),
    "send_ack": ("input_send_start", "input_send_complete"),
    "result_wait": ("result_wait_start", "result_received"),
}
WORK_COLORS = {"credit_wait": "#e7c547", "send_ack": "#347db4", "result_wait": "#df873b"}
MARKERS = {"input_credit_send": ("o", "#887000"), "input_ready": ("|", "#23764d"),
           "result_send_start": ("^", "#222222"), "result_send_complete": ("v", "#89549b")}


def worker_intervals(rows):
    """One endpoint per family and call; endpoint request metadata deliberately differs."""
    endpoints = {event: base.number(base.unique([r for r in rows if r["event"] == event], event)["time_s"])
                 for events in FAMILIES.values() for event in events}
    ordered = [endpoints[e] for e in ("ml_step_start", "input_credit_wait_start", "input_send_start",
               "input_send_complete", "result_wait_start", "result_received", "ml_step_end")]
    if ordered != sorted(ordered):
        raise ValueError(f"Negative or out-of-order worker own-clock interval: {endpoints}")
    return [dict(family=family, start_s=endpoints[start], end_s=endpoints[end],
                 duration_ms=1000*(endpoints[end]-endpoints[start]))
            for family, (start, end) in FAMILIES.items()]


def causal_bounds(grant, ready, send_start, send_complete):
    """Controller time = worker time + delta; no common MPI_Wtime zero assumed."""
    low = max(grant-send_start, ready-send_complete)
    high = ready-send_start
    if low > high:
        raise ValueError(f"Infeasible causal clock bounds [{low}, {high}]; inspect diagnostics "
                         "or explicitly use --alignment local")
    return low, high, (low+high)/2


def controller_endpoint(rows, event, local):
    selected = [r for r in rows if r["event"] == event and int(r["peer_workgroup_rank"]) == local]
    row = base.unique(selected, f"controller {event} for workgroup-local peer {local}")
    if event in ("input_ready", "result_send_start") and (
            int(row["range_first_rank"]), int(row["range_end_rank"])) != (local, local+1):
        raise ValueError(f"Controller range/peer mismatch: {row}")
    return base.number(row["time_s"])


def collect(models, layouts, modes, rep=1, call=6, alignment="causal"):
    comparisons, metrics, events, metadata = base.collect(models, ["24c1g"], layouts, modes, rep, call)
    intervals, alignments = [], []
    for comparison in comparisons:
        p = base.unique(comparison["p2p"], "24c1g workgroup")
        run = metadata["cells"][p["cell"]]["selected_row"]
        mapping = base.parse_mapping(base.ROOT / p["solver_log"], run)[p["workgroup"]]
        if len(mapping["members"]) != 24 or int(run["solver_ranks"]) != 24:
            raise ValueError("Expected exactly 24 mapped members")
        rank_rows, rank_local, paths = {}, {}, {}
        for rank in mapping["members"]:
            path = base.ROOT / p["run_dir"] / "aix_gpu_diagnostics" / f"aix_p2p_timeline_rank_{rank}.csv"
            rows = [r for r in base.read_csv(path) if int(r["step"]) == call]
            locals_seen = {int(r["workgroup_rank"]) for r in rows}
            if len(locals_seen) != 1 or any(int(r["world_rank"]) != rank or
                    int(r["is_controller"]) != int(rank == p["controller"]) for r in rows):
                raise ValueError(f"Invalid rank membership/diagnostic call: {path}")
            rank_local[rank] = locals_seen.pop()
            rank_rows[rank], paths[rank] = rows, str(path.relative_to(base.ROOT))
            metadata["sources"][paths[rank]] = base.sha256(path)
        if set(rank_local.values()) != set(range(24)) or rank_local[p["controller"]] != 0:
            raise ValueError(f"Invalid workgroup-local mapping: {rank_local}")
        comparison["rank_mapping"] = rank_local
        controller = rank_rows[p["controller"]]
        for rank, local in sorted(rank_local.items()):
            if local == 0:
                continue
            own = worker_intervals(rank_rows[rank])
            times = {r["event"]: base.number(r["time_s"]) for r in rank_rows[rank]}
            marks = {event: controller_endpoint(controller, event, local) for event in MARKERS}
            if not (marks["input_credit_send"] <= marks["input_ready"] <= marks["result_send_start"]
                    <= marks["result_send_complete"]):
                raise ValueError(f"Invalid controller endpoint order for rank {rank}")
            low, high, delta = (causal_bounds(marks["input_credit_send"], marks["input_ready"],
                                times["input_send_start"], times["input_send_complete"])
                                if alignment == "causal" else (None, None, None))
            uncertainty = (high-low)*500 if alignment == "causal" else 0
            origin = p["controller_start_s"] if alignment == "causal" else times["ml_step_start"]
            shift = delta if alignment == "causal" else 0
            common = dict(comparison=comparison["name"], cell=p["cell"], call=call,
                          maia_step=base.STEPS[call], world_rank=rank, workgroup_rank=local,
                          controller=p["controller"], alignment=alignment, source_path=paths[rank])
            alignments.append(dict(**common, delta_est_s=delta, delta_low_s=low, delta_high_s=high,
                uncertainty_ms=uncertainty, display_origin_s=origin,
                method="same-call grant/send/ready/ACK causal bounds" if alignment == "causal" else "rank-own ml_step_start zero; unaligned",
                controller_source_path=paths[p["controller"]], c_grant_s=marks["input_credit_send"],
                c_ready_s=marks["input_ready"], w_send_start_s=times["input_send_start"],
                w_send_complete_s=times["input_send_complete"], **{k+"_s": v for k, v in marks.items()}))
            for interval in own:
                intervals.append(dict(**common, **interval, own_start_ms=1000*(interval["start_s"]-times["ml_step_start"]),
                    own_end_ms=1000*(interval["end_s"]-times["ml_step_start"]),
                    display_start_ms=1000*(interval["start_s"]+shift-origin),
                    display_end_ms=1000*(interval["end_s"]+shift-origin), uncertainty_ms=uncertainty))
    metadata.update(alignment=alignment, rank_files_per_comparison=24, worker_intervals=len(intervals),
                    alignment_rows=len(alignments), alignment_assumption="Constant additive clock offset within selected call; no drift model",
                    causal_evidence="collectiveCommunication.cpp: grant log before MPI_Isend (329-331); input_ready before ACK (404-409); completion after Waitall (662-666)")
    for path in (Path(__file__), SOURCE):
        metadata["sources"][str(path.relative_to(base.ROOT))] = base.sha256(path)
    return comparisons, metrics, events, metadata, intervals, alignments


def render_detail(comparison, events, intervals, alignments, out):
    p, c = comparison["p2p"][0], comparison["collective"][0]
    selected = [r for r in events if r["comparison"] == comparison["name"]]
    workers = [r for r in intervals if r["comparison"] == comparison["name"]]
    offsets = sorted([r for r in alignments if r["comparison"] == comparison["name"]], key=lambda r: r["workgroup_rank"])
    causal = offsets[0]["alignment"] == "causal"
    fig = base.plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 1, height_ratios=[3.1, 1.1, 11], hspace=.42)
    gpu, collective, ax = [fig.add_subplot(grid[i]) for i in range(3)]
    fig.suptitle(base.title(comparison), fontsize=12, y=.985)
    gpu.set_title(f"Controller rank {p['controller']} (WG-local 0): own observed host/GPU lanes", loc="left", fontsize=10)
    for row in selected:
        if row["kind"] == "interval" and row["event"].startswith("torch_"):
            lane = base.PHASES.index(row["event"][6:])
            gpu.broken_barh([(row["start_ms"], row["duration_ms"])], (3-lane-.25, .5),
                            facecolors=base.COLORS[lane], edgecolors=base.COLORS[lane], linewidth=.3)
        if row["kind"] == "observation" and row["event"] in MARKERS:
            marker, color = MARKERS[row["event"]]
            gpu.plot(row["start_ms"], 0, marker=marker, color=color, markersize=4, linestyle="none")
    gpu.axvline(p["controller_mlstep_ms"], color="black", linestyle=":", linewidth=.7)
    gpu.set_yticks([3, 2, 1, 0], ["H2D", "Forward", "D2H", "Grant / ready / post / done"])
    gpu.set_ylim(-.5, 3.6)
    gpu.set_xlabel("ms since controller ml_step_start", fontsize=8)
    g, d, s = (c[k] for k in ("gather_collective_ms", "device_cpu_ms", "scatter_collective_ms"))
    for start, width, color, label in ((0, g, "#829b91", "G"), (g, d, "#b9adc9", "D"), (g+d, s, "#8ca3b8", "S")):
        collective.broken_barh([(start, width)], (-.3, .6), facecolors=color, edgecolors=color, linewidth=.4)
        collective.text(start+width/2, 0, label, ha="center", va="center", fontsize=8)
    collective.set_title("Schematic collective reference (gaps omitted, independent zero)", loc="left", fontsize=9)
    collective.set_yticks([0], ["Collective G / D / S"])
    collective.set_ylim(-.7, .7)
    collective.set_xlabel("host durations concatenated from 0; NOT observed collective timestamps", fontsize=8)
    for row in workers:
        lane = row["workgroup_rank"]
        if row["family"] != "ml_step":
            color = WORK_COLORS[row["family"]]
            ax.broken_barh([(row["display_start_ms"], row["duration_ms"])], (lane-.27, .54),
                           facecolors=color, edgecolors=color, linewidth=.3)
        else:
            for endpoint in ("display_start_ms", "display_end_ms"):
                ax.plot(row[endpoint], lane, "|", color="black", markersize=8, markeredgewidth=.8)
                if causal:
                    ax.errorbar(row[endpoint], lane, xerr=row["uncertainty_ms"], color=".35",
                                capsize=2, linewidth=.65, alpha=.7)
    if causal:
        for row in offsets:
            for event, (marker, color) in MARKERS.items():
                ax.plot(1000*(row[event+"_s"]-p["controller_start_s"]), row["workgroup_rank"],
                        marker=marker, color=color, markersize=4, linestyle="none")
    ax.set_yticks([r["workgroup_rank"] for r in offsets],
                 [f"rank {r['world_rank']} / local {r['workgroup_rank']}" +
                  (f"  +/- {r['uncertainty_ms']:.2f} ms" if causal else "  own zero") for r in offsets])
    ax.set_ylim(23.7, .3)
    ax.set_title("23 workers: causal midpoint placement; whiskers = offset bounds" if causal else
                 "23 workers: independent own zeros; controller markers ONLY on top strip", loc="left", fontsize=10)
    ax.set_xlabel("ms since controller ml_step_start (estimated worker placement)" if causal else
                  "ms since EACH worker's own ml_step_start (not synchronized)")
    # Include both full ML spans and their offset uncertainty, including negative starts.
    xmin = min([0]+[r["display_start_ms"]-r["uncertainty_ms"] for r in workers])
    xmax = max([g+d+s, p["controller_mlstep_ms"]]+[r["display_end_ms"]+r["uncertainty_ms"] for r in workers])
    pad = (xmax-xmin)*.015
    for axis in (gpu, collective, ax):
        axis.set_xlim(xmin-pad, xmax+pad)
        axis.tick_params(labelsize=8)
        axis.grid(axis="x", alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
        for x in (g, g+d):
            axis.axvline(x, color=".45", linestyle="--", linewidth=.65)
    collective.text(g, .4, "gather done", ha="left", fontsize=7)
    collective.text(g+d, .4, "scatter start", ha="right", fontsize=7)
    handles = [Patch(facecolor=WORK_COLORS[k], label=label) for k, label in
               (("credit_wait", "Credit wait + logging gap"), ("send_ack", "Blocking send + ACK"), ("result_wait", "Result wait + receive"))]
    handles += [Line2D([], [], marker=marker, color=color, linestyle="none", label=label) for
                (marker, color), label in zip(MARKERS.values(), ("Grant logged", "Ready (before ACK)", "Result send post", "Done (after Waitall)"))]
    handles += [Line2D([], [], marker="|", color="black", linestyle="none", label="Own ML start/end"),
                Line2D([], [], color=".45", linestyle="--", label="Schematic G end / S start")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(.55, .015))
    fig.subplots_adjust(left=.205, right=.98, top=.94, bottom=.14)
    base.save(fig, out, comparison["name"]+"_rank_timeline")


def render_aggregate(comparison, events, out):
    save = base.save

    def spaced_save(fig, directory, name):
        # The base's compact single-WG canvas overlaps title/footer text; only adjust layout.
        fig.set_size_inches(16, 6)
        fig.subplots_adjust(top=.80, bottom=.27)
        for text, y in zip(fig.texts, (.97, .91, .04)):
            text.set_y(y)
        c = comparison["collective"][0]
        axis = fig.axes[0]
        gather_end = c["gather_collective_ms"]
        scatter_start = gather_end + c["device_cpu_ms"]
        for boundary, label in ((gather_end, "Coll gather end (ref.)"),
                                (scatter_start, "Coll scatter start (ref.)")):
            axis.axvline(boundary, color=".4", linestyle="--", linewidth=.7)
            axis.text(boundary, 1.01, label, transform=axis.get_xaxis_transform(),
                      ha="left" if boundary == gather_end else "right", fontsize=7)
        low, high = axis.get_xlim()
        axis.set_xlim(low, max(high, scatter_start * 1.015))
        fig.text(.5, .015, "Dashed collective boundaries: concatenated host durations from independent zero; gaps omitted.",
                 ha="center", fontsize=8)
        save(fig, directory, name)

    with patch.object(base, "save", spaced_save):
        base.render_timeline(comparison, events, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["transformer", "mlp"])
    parser.add_argument("--layouts", nargs="+", choices=["default", "affine", "numa"], default=["default", "numa"])
    parser.add_argument("--modes", nargs="+", choices=["p2p1", "p2pfull"], default=["p2p1", "p2pfull"])
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--call", type=int, choices=range(7), default=6)
    parser.add_argument("--alignment", choices=["causal", "local"], default="causal")
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "figures/aix_rank_timeline_draft")
    args = parser.parse_args(argv)
    comparisons, metrics, events, metadata, intervals, alignments = collect(
        args.models, args.layouts, args.modes, args.rep, args.call, args.alignment)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for comparison in comparisons:
        render_aggregate(comparison, [r for r in events if r["comparison"] == comparison["name"]], args.out_dir)
        render_detail(comparison, events, intervals, alignments, args.out_dir)
    for name, rows in (("workgroup_metrics", metrics), ("events", events),
                       ("worker_intervals", intervals), ("alignment", alignments)):
        base.write_csv(args.out_dir / (name+".csv"), rows)
    metadata["max_uncertainty_ms"] = max(r["uncertainty_ms"] for r in alignments)
    metadata["min_display_with_uncertainty_ms"] = min(r["display_start_ms"]-r["uncertainty_ms"] for r in intervals)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    lines = ["# AIx All-Rank Timeline Draft", "", f"Alignment: `{args.alignment}`. One selected measured call, not repetition averages.", "",
        "Aggregate plots use plot_cmi_aix_timeline.render_timeline, including its unchanged per-WG table and data; "
        "a save-time canvas/spacing adjustment prevents single-WG title/footer collisions. "
        "Dashed lines in aggregate views mark the independently zeroed collective duration reference, not recorded timestamps. "
        "Detailed plots show one controller (GPU lanes and observations) plus all 23 mapped workers. "
        "World-rank membership comes from the solver log; workgroup-local peer IDs come from each actual rank CSV, never sorted-member indices.", "",
        "## Clock Evidence", "",
        "Causal placement assumes a constant additive offset during this call: controller time = worker time + delta. "
        "Bounds are low=max(c_grant-w_send_start, c_ready-w_send_complete), high=c_ready-w_send_start; "
        "the midpoint is displayed, not a synchronized timestamp. Whiskers on both own-ML endpoints and rank-label +/- values "
        "show half the feasible interval; the same uncertainty shifts every bar endpoint. No exact transfer arrows are drawn. "
        "Infeasible bounds fail, never silently switch modes. No cross-call tightening or drift estimate is used.", "",
        "Source: collectiveCommunication.cpp lines 239-260, 329-331, 398-409, 662-666. "
        "Grant is logged before MPI_Isend; worker send starts after receiving credit; input_ready follows input receipt "
        "and precedes posting ACK; worker send-complete follows ACK receipt. These orderings support the bounds. "
        "The source SHA256 is recorded. Logged grant is not a measured credit-receipt timestamp; no such event exists.", "",
        "Local mode independently zeros each worker at its own ml_step_start and the controller at its own start. "
        "Controller marks stay exclusively on the top strip in that mode; row overlap implies no concurrency. "
        "Local alignment CSV delta bounds are empty, uncertainty is 0 for display only, not a claim of synchronization.", "",
        "## Interpretation", "",
        "Yellow spans credit-wait start to input-send start, including the receipt instrumentation gap. "
        "Blue includes blocking MPI_Send AND ACK wait. Orange includes result wait AND receive. Black ticks mark each rank's own ML endpoints. "
        "Controller symbols use the very same selected call's events: grant, ready before ACK, result post, and "
        "done logged after the ALL-output MPI_Waitall, never individual transfer completion. Symbols may overplot. "
        "CUDA-event durations are host anchored; inter-request overlap is approximate.", "",
        "Collective G/D/S uses gather_collective, host device_cpu, scatter_collective durations concatenated from zero. "
        "Arrival waits and gaps are omitted. Dashed lines show schematic gather done and scatter start only, "
        "NOT observed collective timestamps or cross-run synchronization. The independent reference shares only the millisecond scale. "
        "All bars retain true widths. Axis limits include full worker ML spans plus offset uncertainty, including negative starts.", "",
        "[Worker intervals](worker_intervals.csv): own-clock seconds, own-zero milliseconds, displayed milliseconds. "
        "[Alignment](alignment.csv): estimates/bounds, raw causal evidence, endpoint times and source paths. "
        "[Workgroup metrics](workgroup_metrics.csv), [controller events](events.csv), "
        "[metadata and SHA256 provenance](metadata.json), including all 24 rank files per comparison and inherited base provenance.", "", "## Gallery", ""]
    for comparison in comparisons:
        lines += [f"### {base.title(comparison)}", ""]
        for kind in ("timeline", "rank_timeline"):
            name = comparison["name"]+"_"+kind
            lines += [f"[{kind} PDF]({name}.pdf)", f"![{kind}]({name}.png)", ""]
    (args.out_dir / "README.md").write_text("\n".join(lines)+"\n")
    print(f"Rendered {len(comparisons)} comparisons / {len(comparisons)*4} PNG+PDF files; "
          f"{len(intervals)} worker intervals, {len(alignments)} alignments; "
          f"max uncertainty +/-{metadata['max_uncertainty_ms']:.3f} ms; "
          f"minimum extent {metadata['min_display_with_uncertainty_ms']:.3f} ms; {args.out_dir}")


if __name__ == "__main__":
    main()
