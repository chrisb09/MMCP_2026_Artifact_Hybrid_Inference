#!/usr/bin/env python3
"""Four-GPU, all-member rank-local exchange-duration comparisons (no clock alignment)."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from matplotlib.transforms import Bbox

import plot_cmi_aix_timeline as base

plt = base.plt
STAGES = ("gather_arrival_wait_s", "gather_collective_s", "device_cpu_s",
          "scatter_arrival_wait_s", "scatter_collective_s")
COLORS = ("#356b85", "#b46738")
CAVEAT = ("Max WG duration is NOT a globally aligned makespan. Collective: stage-sum proxy (gaps omitted); "
          "P2P: continuous rank-local elapsed. No cross-rank clock alignment.")


def rank_duration(rows, mode):
    """Extract one scalar per rank/call; worker rank zero is a valid identity value."""
    raw = {key: "" for key in (*STAGES, "diagnostic_barriers", "start_s", "end_s", *base.IDENTITY)}
    if mode == "coll":
        row = base.unique(rows, "collective rank call")
        raw.update({key: base.number(row[key]) for key in STAGES})
        raw["diagnostic_barriers"] = int(row["diagnostic_barriers"])
        if raw["diagnostic_barriers"] != 0:
            raise ValueError("Enabled diagnostic barriers are not supported by this stage-sum proxy")
        if raw["gather_arrival_wait_s"] or raw["scatter_arrival_wait_s"]:
            raise ValueError("Arrival waits conflict with disabled diagnostic barriers")
        if row["is_controller"] == "0" and raw["device_cpu_s"] != 0:
            raise ValueError("Collective worker device time must be zero")
        elapsed = 1000 * sum(raw[key] for key in STAGES)
        scope = "collective rank-local stage-sum proxy; interstage gaps omitted"
        continuous = False
    else:
        endpoints = [r for r in rows if r["event"] in ("ml_step_start", "ml_step_end")]
        start = base.unique([r for r in endpoints if r["event"] == "ml_step_start"], "ml_step_start")
        end = base.unique([r for r in endpoints if r["event"] == "ml_step_end"], "ml_step_end")
        if any(int(start[k]) != int(end[k]) for k in base.IDENTITY):
            raise ValueError("Conflicting full ml_step endpoint identity")
        raw.update({k: int(start[k]) for k in base.IDENTITY})
        raw.update(start_s=base.number(start["time_s"]), end_s=base.number(end["time_s"]))
        elapsed = base.number(raw["end_s"] - raw["start_s"]) * 1000
        scope = "continuous rank-local ml_step elapsed"
        continuous = True
    return dict(**raw, duration_ms=elapsed, continuous=continuous, duration_scope=scope)


def summarize(groups):
    if len(groups) != 4:
        raise ValueError("Exactly four workgroups are required")
    durations = [r["duration_ms"] for r in groups]
    largest, smallest = max(durations), min(durations)
    if largest <= 0:
        raise ValueError("Largest WG duration must be positive")
    mean = sum(durations) / 4
    return dict(balance=mean / largest, max_ms=largest, mean_ms=mean, min_ms=smallest,
                range_ms=largest-smallest, n_workgroups=4)


def collect(models, placements, layouts, modes, rep=1, call=6, data_dir=base.DATA):
    comparisons, _, _, metadata = base.collect(models, placements, layouts, modes, rep, call, data_dir)
    ranks, groups, summaries, cache = [], [], [], {}
    for comparison in comparisons:
        for side in ("collective", "p2p"):
            selected = comparison[side]
            cell = selected[0]["cell"]
            run = metadata["cells"][cell]["selected_row"]
            if cell not in cache:
                directory = base.ROOT / run["run_dir"]
                mapping = base.parse_mapping(directory / "solver.log", run)
                members = [r for m in mapping.values() for r in m["members"]]
                if len(mapping) != 4 or sorted(members) != list(range(int(run["solver_ranks"]))):
                    raise ValueError("Incomplete or duplicate four-workgroup membership")
                local_ranks, local_groups = [], []
                for group, m in sorted(mapping.items()):
                    wg_rows = []
                    for rank in m["members"]:
                        p2p = run["comm"] != "coll"
                        filename = f"aix_p2p_timeline_rank_{rank}.csv" if p2p else f"aix_rank_{rank}.csv"
                        path = directory / "aix_gpu_diagnostics" / filename
                        rows = [r for r in base.read_csv(path) if int(r["step" if p2p else "call"]) == call]
                        if not rows:
                            raise ValueError(f"Missing rank timing: {path}, call {call}")
                        identity = {(int(r["world_rank"]), int(r["workgroup_rank"]), r["is_controller"]) for r in rows}
                        if len(identity) != 1:
                            raise ValueError(f"Conflicting rank metadata: {path}")
                        world, wg_rank, controller_flag = identity.pop()
                        if (world != rank or controller_flag != str(int(rank == m["controller"]))
                                or (wg_rank == 0) != (rank == m["controller"])):
                            raise ValueError(f"Invalid rank/controller metadata: {path}")
                        row = dict(cell=cell, jobid=run["jobid"], run_dir=run["run_dir"],
                                   model=run["model"], placement=run["placement"], layout=run["layout"],
                                   rep=rep, mode=run["comm"], call=call, inference_call=call+1,
                                   maia_step=base.STEPS[call], workgroup=group, world_rank=rank,
                                   workgroup_rank=wg_rank, is_controller=int(controller_flag),
                                   controller=m["controller"], gpu_ordinal=m["gpu"],
                                   members=";".join(map(str, m["members"])),
                                   source_path=str(path.relative_to(base.ROOT)), **rank_duration(rows, run["comm"]))
                        wg_rows.append(row)
                        metadata["sources"][str(path.relative_to(base.ROOT))] = base.sha256(path)
                    if sorted(r["workgroup_rank"] for r in wg_rows) != list(range(len(m["members"]))):
                        raise ValueError(f"Missing/duplicate CSV workgroup rank: {cell}, WG {group}")
                    worst = max(wg_rows, key=lambda r: r["duration_ms"])
                    controller = base.unique([r for r in wg_rows if r["is_controller"]], "controller")
                    common = {k: worst[k] for k in ("cell", "jobid", "run_dir", "model", "placement", "layout",
                              "rep", "mode", "call", "inference_call", "maia_step", "workgroup", "controller",
                              "gpu_ordinal", "members", "duration_scope", "continuous")}
                    local_groups.append(dict(**common, duration_ms=worst["duration_ms"],
                                             worst_rank=worst["world_rank"], n_members=len(wg_rows),
                                             controller_duration_ms=controller["duration_ms"]))
                    local_ranks.extend(wg_rows)
                cache[cell] = local_ranks, local_groups
            local_ranks, local_groups = cache[cell]
            selected_groups = [dict(comparison=comparison["name"], **r) for r in local_groups]
            ranks.extend(dict(comparison=comparison["name"], **r) for r in local_ranks)
            groups.extend(selected_groups)
            comparison[side] = selected_groups
            summaries.append(dict(comparison=comparison["name"], cell=cell, jobid=run["jobid"],
                                  model=run["model"], placement=run["placement"], layout=run["layout"],
                                  mode=run["comm"], rep=rep, call=call, maia_step=base.STEPS[call],
                                  maia_mlstep_ms=selected[0]["overall_maia_mlstep_ms"], **summarize(selected_groups)))
        # The base also checks this; retain the all-member contract at this API boundary.
        signatures = [{(r["members"], r["gpu_ordinal"]) for r in comparison[s]} for s in ("collective", "p2p")]
        if signatures[0] != signatures[1]:
            raise ValueError("Workgroup membership/GPU pairing mismatch")
    for path in (Path(__file__), Path(base.__file__), base.ROOT /
                 "CPP-ML-Interface/extern/AIxeleratorService/src/aixeleratorService/aixeleratorService.cpp"):
        metadata["sources"][str(path.relative_to(base.ROOT))] = base.sha256(path)
    metadata.pop("event_rows", None)
    metadata.pop("metrics_rows", None)
    metadata.update(clock_caveats=CAVEAT, metric_definition=CAVEAT, per_rank_rows=len(ranks), workgroup_rows=len(groups),
                    summary_rows=len(summaries), aggregation="individual selected call; no averaging or CI")
    return comparisons, ranks, groups, summaries, metadata


def render(comparison, summaries, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, .8, 1.3]})
    labels = ("Collective", comparison["p2p"][0]["mode"])
    for i, side in enumerate(("collective", "p2p")):
        rows = comparison[side]
        summary = base.unique([r for r in summaries if r["comparison"] == comparison["name"]
                               and r["mode"] == rows[0]["mode"]], "balance summary")
        bars = axes[0].bar([g + (i-.5)*.36 for g in range(4)], [r["duration_ms"] for r in rows],
                           .34, color=COLORS[i], label=labels[i])
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=6, rotation=45)
        bars = axes[1].bar(i, summary["balance"], .55, color=COLORS[i])
        axes[1].bar_label(bars, labels=[f"{summary['balance']:.3f}"], padding=-14, fontsize=8, color="white")
        bars = axes[2].bar([j + (i-.5)*.36 for j in range(2)],
                           [summary["max_ms"], summary["maia_mlstep_ms"]], .34, color=COLORS[i])
        axes[2].bar_label(bars, fmt="%.1f", padding=3, fontsize=7)
    axes[0].set_xticks(range(4), [f"WG {r['workgroup']}\nGPU {r['gpu_ordinal']}" for r in comparison["collective"]])
    axes[0].set_title("Max Rank-Local Exchange Duration", fontsize=10)
    axes[0].set_ylabel("Duration (ms)")
    axes[1].set_title("Exchange-Duration Balance", fontsize=10)
    axes[1].set_xticks(range(2), labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0, .25, .5, .75, 1])
    axes[1].set_ylabel(r"$B = \mathrm{mean}(D_g) / \max(D_g)$")
    axes[2].set_xticks(range(2), ["Largest WG\nduration (proxy)", "MAIA ML-step\nelapsed"])
    axes[2].set_title("Exchange Proxy vs Full ML Step", fontsize=10)
    axes[2].set_ylabel("Duration (ms)")
    ceiling = max(ax.get_ylim()[1] for ax in (axes[0], axes[2])) * 1.18
    for ax in (axes[0], axes[2]):
        ax.set_ylim(0, ceiling)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    fig.suptitle(base.title(comparison), fontsize=10, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.23, wspace=.42)
    # Crop the canvas, not the axes: preserve bar geometry and physical scale.
    fig.canvas.draw()
    bottom = max(0, fig.get_tightbbox(fig.canvas.get_renderer()).y0-.08)
    bounds = Bbox.from_extents(0, bottom, fig.get_figwidth(), fig.get_figheight())
    savefig = fig.savefig
    with patch.object(fig, "savefig", side_effect=lambda *a, **kw: savefig(*a, **kw, bbox_inches=bounds)):
        base.save(fig, out, comparison["name"]+"_exchange_balance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["transformer", "mlp"])
    parser.add_argument("--placements", nargs="+", default=["96c4g", "96p96c4g"])
    parser.add_argument("--layouts", nargs="+", choices=["default", "numa"], default=["default", "numa"])
    parser.add_argument("--modes", nargs="+", choices=["p2p1", "p2pfull"], default=["p2p1", "p2pfull"])
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--call", type=int, choices=range(7), default=6,
                        help="Zero-based diagnostic call, measured calls 2..6 only (0,1 rejected)")
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "figures/aix_exchange_balance_draft")
    args = parser.parse_args(argv)
    comparisons, ranks, groups, summaries, metadata = collect(
        args.models, args.placements, args.layouts, args.modes, args.rep, args.call)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    for comparison in comparisons:
        render(comparison, summaries, args.out_dir)
    for filename, rows in (("per_rank_durations.csv", ranks), ("workgroup_durations.csv", groups),
                           ("balance_summary.csv", summaries)):
        base.write_csv(args.out_dir / filename, rows)
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    lines = ["# AIx Exchange-Duration Balance", "", CAVEAT, "",
              "Narrative notes and statistics are README/caption-only in every graphic version. "
              "Numeric bar labels are retained. Saved canvases crop unused bottom space without stretching axes.", "",
              f"Output: {len(comparisons)} comparisons, each PNG/PDF; "
              f"{2*len(comparisons)} figure files; no PGF. "
              f"{len(ranks)} rank rows, {len(groups)} WG rows, {len(summaries)} summaries.", "",
             r"For collective rank $r$, $e_r=1000(\mathrm{gather}_r+\mathrm{device}_r+\mathrm{scatter}_r)$ ms. "
             "Diagnostic barriers must be disabled (enabled barriers are rejected); arrival waits must be zero. "
             "Workers have zero device time; scatter already includes waiting, so controller device time is never added again. "
             "This stage SUM omits interstage gaps and is not a continuous observed interval.", "",
             r"For P2P, $e_r=1000(t_{r,\mathrm{end}}-t_{r,\mathrm{start}})$ uses only that rank's clock. "
             "Endpoints are unique scalars paired by full request identity, including valid zero-valued worker fields.", "",
             r"For each full-membership WG, $D_g=\max_{r\in g}e_r$. Exactly four WGs are required. "
             r"Exchange-duration balance is $B=\frac{\sum_g D_g}{4\max_g D_g}$; 1 means even durations, "
             "not equal work, computational load balance, or completion-time skew. "
             "The right panel compares max(D_g), a proxy NOT a global makespan, with separately measured full MAIA ML-step elapsed.", "",
             "Scope: AIx provider only; each P2P cell is bound to the collective baseline with the same model, placement, "
             "layout, repetition and selected call. Pairing requires exact full WG membership and GPU mapping, never rank intersection. "
             "Membership/controller/GPU selection comes from base.parse_mapping(solver.log); CSV communicator ranks are validated "
             "as a complete permutation with controller rank zero, not assumed equal to logged distribution order. "
             "Default GPU mapping uses logged node-local controller ranks; NUMA uses explicit GPU selections.", "",
             f"Selected repetition {args.rep}, diagnostic call {args.call} (inference call {args.call+1}, "
             f"MAIA step {base.STEPS[args.call]}). Calls 0,1 are warmups; 2..6 are measured. "
             "These are individual representative calls, NOT averages over five measured calls or three repetitions; no CI. "
             "The shared collective baseline appears in both P2P comparisons and must not be counted as independent samples.", "",
             "[Per-rank raw stages and scopes](per_rank_durations.csv), [WG maxima, worst rank and controller duration]"
             "(workgroup_durations.csv), [balance and MAIA summaries](balance_summary.csv), "
             "[selected run rows and SHA256 input/source hashes](metadata.json). All member-rank diagnostic files are hashed.", "",
             "## Gallery", ""]
    for comparison in comparisons:
        name = comparison["name"]+"_exchange_balance"
        lines += [f"### {base.title(comparison)}", "", f"[Vector PDF]({name}.pdf)", f"![Comparison]({name}.png)", ""]
    lines += ["## Selected Sources", ""]
    for cell, record in metadata["cells"].items():
        run = record["selected_row"]
        link = os.path.relpath(base.ROOT / run["run_dir"] / "solver.log", args.out_dir)
        lines.append(f"- `{cell}`, job `{run['jobid']}`: [solver.log]({link})")
    (args.out_dir / "README.md").write_text("\n".join(lines)+"\n")
    print(f"Rendered {len(comparisons)} comparisons; {len(ranks)} rank rows, {len(groups)} WG rows, "
          f"{len(summaries)} summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
