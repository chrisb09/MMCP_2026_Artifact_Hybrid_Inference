#!/usr/bin/env python3
"""Selected-run AIx controller-local timelines and GPU work (no Score-P)."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/cmi_bench_consolidated"
STEPS = (15, 44, 73, 102, 131, 160, 189)
PHASES = ("h2d", "forward", "d2h")
COLORS = ("#2686a5", "#b96332", "#7956a0")
IDENTITY = ("range_first_rank", "range_end_rank", "sample_start", "sample_count",
            "peer_workgroup_rank")
CAVEAT = ("Each controller has its own zero; rows are NOT globally synchronized. CUDA-event durations are reliable;\n"
          "host-anchored inter-request overlap is approximate. GPU work SUM is not elapsed time or a critical path.")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative duration/time: {value}")
    return value


def unique(rows, description):
    if len(rows) != 1:
        raise ValueError(f"Expected one {description}, got {len(rows)}")
    return rows[0]


def parse_mapping(log, run):
    """Return group -> controller/GPU/members from this run, never rank intersection."""
    text = Path(log).read_text(errors="replace")
    def records(pattern):
        result = {}
        for match in re.finditer(pattern, text):
            key, *value = map(int, match.groups())
            if key in result and result[key] != tuple(value):
                raise ValueError(f"Conflicting mapping for rank {key}: {log}")
            result[key] = tuple(value)
        return result
    groups = records(r"Rank (\d+)/\d+ will be in group (\d+) order (\d+)")
    if set(groups) != set(range(int(run["solver_ranks"]))):
        raise ValueError(f"Incomplete workgroup mapping: {log}")
    if run["layout"] == "default":
        # RoundRobinDistribution::createWorkgroups assigns my_gpu_device_ = node_rank.
        devices = records(r"Rank (\d+)/\d+ on its local machine is (\d+)/\d+ is the GPU controller!")
    else:
        devices = records(r"Rank (\d+) is GPU controller for GPU (\d+) on NUMA node \d+")
    controllers = {rank for rank, (_, order) in groups.items() if order == 0}
    if set(devices) != controllers or len(controllers) != int(run["gpu_count"]):
        raise ValueError(f"Missing controller/GPU selection: {log}")
    result = {}
    for rank in sorted(controllers):
        group = groups[rank][0]
        if group in result:
            raise ValueError(f"Multiple controllers in group {group}: {log}")
        result[group] = dict(controller=rank, gpu=devices[rank][0],
                             members=sorted(r for r, (g, _) in groups.items() if g == group))
    return result


def pair_intervals(rows, call, controller):
    """Pair unordered phase endpoints by full request identity; reject damaged data."""
    bases = ("ml_step", "range_inference", "torch_h2d", "torch_forward", "torch_d2h")
    endpoints = {}
    for row in rows:
        event = row["event"]
        for base in bases:
            if event in (base + "_start", base + "_end"):
                key = (call, controller, base, *(int(row[k]) for k in IDENTITY))
                side = event.rsplit("_", 1)[1]
                if side in endpoints.setdefault(key, {}):
                    raise ValueError(f"Duplicate endpoint: {key} {side}")
                endpoints[key][side] = number(row["time_s"])
    intervals = []
    for key, sides in endpoints.items():
        if set(sides) != {"start", "end"} or sides["end"] < sides["start"]:
            raise ValueError(f"Missing or negative interval: {key} {sides}")
        intervals.append(dict(event=key[2], **dict(zip(IDENTITY, key[3:])),
                              start_s=sides["start"], end_s=sides["end"],
                              duration_ms=(sides["end"]-sides["start"])*1000))
    unique([r for r in intervals if r["event"] == "ml_step"], "ml_step interval")
    requests = {tuple(r[k] for k in IDENTITY) for r in intervals if r["event"] == "range_inference"}
    if not requests:
        raise ValueError("No range inference intervals")
    for phase in PHASES:
        observed = {tuple(r[k] for k in IDENTITY) for r in intervals if r["event"] == "torch_"+phase}
        if observed != requests:
            raise ValueError(f"Missing/extra {phase} request intervals")
    return sorted(intervals, key=lambda r: (r["start_s"], r["event"]))


def load_run(run, inferences, call):
    directory = ROOT / run["run_dir"]
    log = directory / "solver.log"
    mapping = parse_mapping(log, run)
    inference = unique([r for r in inferences if r["cell"] == run["cell"]
                        and int(r["call"]) == call+1], "selected inference")
    if (inference["phase"] != "measured" or int(inference["step"]) != STEPS[call]
            or inference["jobid"] != run["jobid"]
            or run["exit_status"] != "0" or run["solver_exit_status"] != "0"):
        raise ValueError(f"Not a successful measured call: {run['cell']} call {call}")
    metrics, events, paths = [], [], [log]
    for group, mapping_row in sorted(mapping.items()):
        ctrl = mapping_row["controller"]
        p2p = run["comm"] != "coll"
        path = directory / "aix_gpu_diagnostics" / (f"aix_p2p_timeline_rank_{ctrl}.csv" if p2p
                                                    else f"aix_rank_{ctrl}.csv")
        paths.append(path)
        selected = [r for r in read_csv(path) if int(r["step" if p2p else "call"]) == call]
        if not selected or any(r["is_controller"] != "1" or int(r["world_rank"]) != ctrl
                               or int(r["workgroup_rank"]) != 0 for r in selected):
            raise ValueError(f"Missing/invalid controller diagnostics: {path}")
        metric = dict(model=run["model"], placement=run["placement"], layout=run["layout"],
                      rep=run["rep"], mode=run["comm"], workgroup=group, controller=ctrl,
                      gpu_ordinal=mapping_row["gpu"], members=";".join(map(str, mapping_row["members"])),
                      selected_job=run["jobid"], cell=run["cell"], cell_sha256=cell_hash(run),
                      run_dir=run["run_dir"], solver_log=str(log.relative_to(ROOT)),
                      diagnostic_path=str(path.relative_to(ROOT)), call=call,
                      inference_call=call+1, maia_step=STEPS[call],
                      overall_maia_mlstep_ms=number(inference["latency_ms"]))
        metric.update({k: "" for k in ("controller_start_s", "controller_end_s", "controller_mlstep_ms",
                                       "p2p_gpu_envelope_ms", "p2p_range_envelope_ms", "device_cpu_ms",
                                       "gather_arrival_wait_ms", "gather_collective_ms",
                                       "scatter_arrival_wait_ms", "scatter_collective_ms",
                                       "diagnostic_barriers", "device_batches", "nRanges")})
        if not p2p:
            row = unique(selected, "collective controller call")
            for phase in PHASES:
                metric[phase+"_ms"] = number(row[phase+"_gpu_ms"])
            for src in ("device_cpu", "gather_arrival_wait", "gather_collective",
                        "scatter_arrival_wait", "scatter_collective"):
                metric[src+"_ms"] = 1000*number(row[src+"_s"])
            for key in ("diagnostic_barriers", "device_batches"):
                metric[key] = int(row[key])
            metric["duration_metadata"] = "CUDA phase work; host device_cpu elapsed; no timestamps available"
        else:
            intervals = pair_intervals(selected, call, ctrl)
            ml = unique([r for r in intervals if r["event"] == "ml_step"], "ml_step")
            zero = ml["start_s"]
            metric.update(controller_start_s=zero, controller_end_s=ml["end_s"],
                          controller_mlstep_ms=ml["duration_ms"])
            for phase in PHASES:
                metric[phase+"_ms"] = sum(r["duration_ms"] for r in intervals if r["event"] == "torch_"+phase)
            for prefix, match in (("p2p_gpu", lambda r: r["event"].startswith("torch_")),
                                  ("p2p_range", lambda r: r["event"] == "range_inference")):
                spans = [r for r in intervals if match(r)]
                metric[prefix+"_envelope_ms"] = 1000*(max(r["end_s"] for r in spans)-min(r["start_s"] for r in spans))
            metric["nRanges"] = sum(r["event"] == "range_inference" for r in intervals)
            metric["duration_metadata"] = "CUDA-event-derived host-anchored phases; inter-request overlap approximate"
            common = dict(cell=run["cell"], jobid=run["jobid"], workgroup=group, controller=ctrl,
                          gpu_ordinal=mapping_row["gpu"], call=call, maia_step=STEPS[call],
                          source_path=str(path.relative_to(ROOT)))
            for row in intervals:
                events.append(dict(**common, kind="interval", **row,
                                   start_ms=(row["start_s"]-zero)*1000, end_ms=(row["end_s"]-zero)*1000))
            seen = set()
            for row in selected:
                key = (row["event"], *(row[k] for k in IDENTITY))
                if key in seen:
                    raise ValueError(f"Duplicate observation: {key}: {path}")
                seen.add(key)
                time = number(row["time_s"])
                events.append(dict(**common, kind="observation", event=row["event"],
                                   **{k: int(row[k]) for k in IDENTITY}, start_s=time, end_s=time,
                                   duration_ms=0, start_ms=(time-zero)*1000, end_ms=(time-zero)*1000))
        metric["gpu_sum_ms"] = sum(metric[p+"_ms"] for p in PHASES)
        metrics.append(metric)
    return metrics, events, paths


def collect(models, placements, layouts, modes, rep=1, call=6, data_dir=DATA):
    """Public data API: comparisons, metrics, events, provenance (without rendering)."""
    if call not in range(7):
        raise ValueError("Diagnostic call must be 0..6")
    runs = read_csv(data_dir / "runs.csv")
    inferences = read_csv(data_dir / "inferences.csv")
    comparisons, metrics, events, sources, cells = [], [], [], set(), {}
    cache = {}
    for model, placement, layout, mode in itertools.product(models, placements, layouts, modes):
        if mode not in ("p2p1", "p2pfull"):
            raise ValueError(f"Unsupported P2P mode: {mode}")
        pair = []
        name = f"{model}_{placement}_{layout}_{mode}_r{rep}_call{call}"
        for comm in ("coll", mode):
            run = unique([r for r in runs if (r["provider"], r["model"], r["placement"], r["layout"],
                                            r["comm"], r["rep"]) == ("aix", model, placement, layout, comm, str(rep))],
                         f"selected {model}/{placement}/{layout}/{comm}/r{rep} run")
            if run["cell"] not in cache:
                cache[run["cell"]] = load_run(run, inferences, call)
            rows, observations, paths = cache[run["cell"]]
            cells[run["cell"]] = dict(sha256=cell_hash(run), selected_row=run)
            sources.update(paths)
            selected_metrics = [dict(comparison=name, **r) for r in rows]
            metrics.extend(selected_metrics)
            events.extend(dict(comparison=name, **r) for r in observations)
            pair.append(selected_metrics)
        if {r["workgroup"]: (r["gpu_ordinal"], r["members"]) for r in pair[0]} != {
                r["workgroup"]: (r["gpu_ordinal"], r["members"]) for r in pair[1]}:
            raise ValueError(f"Workgroup/GPU membership mismatch: {name}")
        comparisons.append(dict(name=name, collective=pair[0], p2p=pair[1]))
    sources.update((data_dir / "runs.csv", data_dir / "inferences.csv", Path(__file__),
                    ROOT / "CPP-ML-Interface/extern/AIxeleratorService/src/distributionStrategy/roundRobinDistribution.cpp"))
    metadata = dict(cells=cells, sources={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                                        sha256(p) for p in sorted(sources)},
                    clock_caveats=CAVEAT, call=call, maia_step=STEPS[call],
                    comparisons=len(comparisons), workgroup_comparisons=len(metrics)//2,
                    metrics_rows=len(metrics), event_rows=len(events))
    return comparisons, metrics, events, metadata


def title(comparison):
    row = comparison["p2p"][0]
    return (f"{row['model']} | {row['placement']} | {row['layout']} | collective vs {row['mode']} | "
            f"rep {row['rep']} | call {row['call']} / MAIA step {row['maia_step']}")


def save(fig, out, name):
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(out / f"{name}.{suffix}", dpi=250, metadata=metadata)
    plt.close(fig)


def render_timeline(comparison, events, out):
    coll, p2p = comparison["collective"], comparison["p2p"]
    xmax = max(row["controller_mlstep_ms"] for row in p2p) * 1.015
    fig = plt.figure(figsize=(15.5, 2.1+2.05*len(p2p)))
    grid = fig.add_gridspec(len(p2p), 2, width_ratios=(3.2, 1.65), hspace=.58, wspace=.20)
    fig.suptitle(title(comparison), fontsize=12, y=.985)
    fig.text(.5, .947, f"Overall MAIA ML-step makespan: collective {coll[0]['overall_maia_mlstep_ms']:.2f} ms"
             f" | {p2p[0]['mode']} {p2p[0]['overall_maia_mlstep_ms']:.2f} ms", ha="center", fontsize=10)
    for i, (c, p) in enumerate(zip(coll, p2p)):
        ax = fig.add_subplot(grid[i, 0])
        selected = [r for r in events if r["cell"] == p["cell"] and r["workgroup"] == p["workgroup"]]
        for row in selected:
            if row["kind"] == "interval" and row["event"] == "range_inference":
                ax.broken_barh([(row["start_ms"], row["duration_ms"])], (3.7, .6),
                               facecolors="#c5c9cb", edgecolors="none", alpha=.65)
            if row["kind"] == "interval" and row["event"].startswith("torch_"):
                idx = PHASES.index(row["event"][6:])
                # True widths only. Hairline edges preserve tiny intervals without changing their data extent.
                ax.broken_barh([(row["start_ms"], row["duration_ms"])], (2.7-idx, .6),
                               facecolors=COLORS[idx], edgecolors=COLORS[idx], linewidth=.35)
            if row["kind"] == "observation":
                lane = {"input_ready": 5, "result_send_start": .12, "result_send_complete": -.12}.get(row["event"])
                if lane is not None:
                    ax.plot(row["start_ms"], lane, marker="|" if lane == 5 else ("^" if lane > 0 else "v"),
                            color="#27634d" if lane == 5 else "#333333", markersize=4, markeredgewidth=.6)
        ax.set_yticks([5, 4, 3, 2, 1, 0], ["Input ready", "Range (host)", "H2D", "Forward", "D2H", "Send post / done"])
        ax.set_ylim(-.5, 5.5)
        ax.axvline(p["controller_mlstep_ms"], color=".5", linestyle="--", linewidth=.6)
        ax.set_xlim(0, xmax)
        ax.set_title(f"WG {p['workgroup']} / GPU {p['gpu_ordinal']} | controller {p['controller']} | "
                     f"{p['nRanges']} ranges | own exchange {p['controller_mlstep_ms']:.2f} ms", loc="left", fontsize=9)
        ax.set_xlabel("ms since this controller's own ml_step_start", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        table_ax = fig.add_subplot(grid[i, 1])
        table_ax.axis("off")
        table = table_ax.table(cellText=[[label, f"{c[key]:.3f}", f"{p[key]:.3f}"] for label, key in
                                        (("H2D", "h2d_ms"), ("Forward", "forward_ms"),
                                         ("D2H", "d2h_ms"), ("GPU SUM", "gpu_sum_ms"))],
                              colLabels=["GPU work (ms)", "Collective", p["mode"]], loc="upper center",
                              colWidths=[.38, .31, .31], cellLoc="right")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.23)
        table_ax.text(0, -.03, f"Controllers: coll {c['controller']} / P2P {p['controller']}\n"
                      f"Coll host device elapsed: {c['device_cpu_ms']:.3f} ms\n"
                      f"P2P GPU envelope: {p['p2p_gpu_envelope_ms']:.3f} ms\n"
                      "Different scopes; not an elapsed-time comparison.", fontsize=8, va="top")
    fig.text(.5, .023, CAVEAT, ha="center", fontsize=8)
    fig.subplots_adjust(left=.105, right=.98, top=.88, bottom=.105)
    save(fig, out, comparison["name"]+"_timeline")


def render_gpu_work(comparison, out):
    coll, p2p = comparison["collective"], comparison["p2p"]
    fig, (ax, latency) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2.5, 1]})
    for i, (c, p) in enumerate(zip(coll, p2p)):
        for offset, row in ((-.19, c), (.19, p)):
            bottom = 0
            for phase, color in zip(PHASES, COLORS):
                value = row[phase+"_ms"]
                ax.bar(i+offset, value, .34, bottom=bottom, color=color,
                       hatch="//" if offset < 0 else None, edgecolor="white", linewidth=.3)
                bottom += value
            ax.annotate(f"{bottom:.2f}", (i+offset, bottom), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, rotation=30)
    ax.set_xticks(range(len(p2p)), [f"WG {r['workgroup']} / GPU {r['gpu_ordinal']}" for r in p2p])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Accumulated GPU phase work (ms), NOT elapsed")
    ax.set_title("GPU work SUM per workgroup", fontsize=11)
    ax.set_ylim(0, max(r["gpu_sum_ms"] for r in coll+p2p)*1.23)
    handles = [Patch(facecolor=c, label=p.upper() if p != "forward" else "Forward") for p, c in zip(PHASES, COLORS)]
    handles += [Patch(facecolor="white", edgecolor=".3", hatch="//", label="Collective (left)"),
                Patch(facecolor=".6", label=p2p[0]["mode"]+" (right)")]
    fig.legend(handles=handles, fontsize=8, ncol=5, loc="upper center",
               bbox_to_anchor=(.5, .925), frameon=False)
    values = [coll[0]["overall_maia_mlstep_ms"], p2p[0]["overall_maia_mlstep_ms"]]
    latency.bar([0, 1], values, color=["#687783", "#384e5a"], width=.6)
    for i, value in enumerate(values):
        latency.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    latency.set_xticks([0, 1], ["Collective", p2p[0]["mode"]])
    latency.set_ylim(0, max(values)*1.23)
    latency.set_title("Overall MAIA ML step", fontsize=11)
    latency.set_ylabel("Makespan (ms), separate metric")
    for axis in (ax, latency):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(title(comparison), fontsize=11)
    fig.text(.5, .025, "Left: accumulated CUDA phase durations, not a critical path. Right: separately measured MAIA makespan.\n"
             "One matched measured call, not a repetition-averaged performance result.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.18, wspace=.36)
    save(fig, out, comparison["name"]+"_gpu_work")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["transformer", "mlp"])
    parser.add_argument("--placements", nargs="+", default=["96c4g", "96p96c4g"])
    parser.add_argument("--layouts", nargs="+", choices=["default", "affine", "numa"], default=["default", "numa"])
    parser.add_argument("--modes", nargs="+", choices=["p2p1", "p2pfull"], default=["p2p1", "p2pfull"])
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--call", type=int, choices=range(7), default=6,
                        help="Zero-based diagnostic call; must be measured (0,1 are warmups and rejected)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/aix_pipeline_draft")
    args = parser.parse_args(argv)
    comparisons, metrics, events, metadata = collect(args.models, args.placements, args.layouts,
                                                    args.modes, args.rep, args.call)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for comparison in comparisons:
        selected = [r for r in events if r["comparison"] == comparison["name"]]
        render_timeline(comparison, selected, args.out_dir)
        render_gpu_work(comparison, args.out_dir)
    write_csv(args.out_dir / "workgroup_metrics.csv", metrics)
    write_csv(args.out_dir / "events.csv", events)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    lines = ["# AIx Pipeline Draft", "", CAVEAT.replace("\n", " "), "",
             "Only exact selected AIx cells in consolidated runs.csv and inferences.csv are used. "
             "Collective measurements are durations only; no collective timestamps are reconstructed. "
             "No worker/controller or cross-controller clock alignment is assumed. "
             "Send triangles mark controller post/completion observations, not worker receipt or network transfer duration.", "",
             "Calls are zero-based in diagnostics, one-based in inferences.csv; calls 0 and 1 are warmup. "
             "GPU phases use CUDA-event-derived durations anchored to each request's host clock; row order need not be chronological. "
              "Each timeline row resets to its own controller ml_step_start. "
              "Rows share the same time scale; dashed vertical lines mark each controller's exchange end. "
             "Range bars are host spans, GPU envelopes are first-phase-start to last-phase-end. "
             "Collective host device_cpu elapsed is a different scope, not a GPU envelope. "
             "Small intervals retain their real widths with hairline edges; none are inflated.", "",
             "[Workgroup metrics](workgroup_metrics.csv) include CPU gather/scatter audit durations, "
             "controller exchange duration, overall MAIA makespan and request counts. "
             "[Events](events.csv) include paired intervals and raw controller observations. "
             "[Metadata](metadata.json) records selected cells, their hashes, source CSV/diagnostic/log SHA256 hashes, and counts. "
             "Default GPU ordinals follow the logged node-local controller rank, as assigned by "
             "RoundRobinDistribution::createWorkgroups; affine/NUMA use explicit logged GPU selections. "
             "Missing/conflicting mappings are errors.", "", "## Gallery", ""]
    for comparison in comparisons:
        name = comparison["name"]
        lines += [f"### {title(comparison)}", ""]
        for kind in ("timeline", "gpu_work"):
            lines += [f"[{kind} PDF]({name}_{kind}.pdf)", f"![{kind}]({name}_{kind}.png)", ""]
    lines += ["## Selected Sources", ""]
    for cell, record in metadata["cells"].items():
        run = record["selected_row"]
        link = os.path.relpath(ROOT / run["run_dir"] / "solver.log", args.out_dir)
        lines.append(f"- `{cell}`, job `{run['jobid']}`: [solver.log]({link})")
    (args.out_dir / "README.md").write_text("\n".join(lines)+"\n")
    print(f"Rendered {len(comparisons)} comparisons, {len(metrics)//2} workgroup comparisons, "
          f"{len(metrics)} metrics rows to {args.out_dir}")


if __name__ == "__main__":
    main()
