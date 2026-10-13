#!/usr/bin/env python3
"""Draw AIx rank/workgroup placement diagrams for the layout experiments.

Each diagram shows the physical node topology: two sockets, each made of four
NUMA domains arranged in a 2x2 grid (adjacent domains linked by EMIB, the two
sockets linked by UPI). Every core is a small square labelled with the MPI rank
bound to it, filled with the colour of the AIx workgroup the rank belongs to.
GPU-controller ranks get a red border and each GPU is drawn with the workgroup
colour of the controller that drives it.

Modes:
  * single layout  -> one PNG/PDF/PGF
  * --combined     -> the three layouts side by side with one shared legend

Group assignments and the rank->NUMA binding are parsed from the actual job
logs (``will be in group``/``order``, ``is bound to NUMA node``). Multi-cell
logs require an explicit run ID; incomplete or conflicting records are errors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PGF = "--pgf" in sys.argv
if PGF:
    sys.path.insert(0, str(Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"))
    import thesis_style as ts

    ts.apply()
else:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

BINDING_RE = re.compile(r"Rank (\d+) is bound to NUMA node (\d+)")
GROUP_RE = re.compile(r"Rank (\d+)/\d+ will be in group (\d+) order (\d+)")
HOST_RE = re.compile(r"Rank (\d+)/\d+ NUMA-local assignment: rank NUMA (\d+) host (\d+)/(\d+)")

GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  # no red
CONTROLLER_EDGE = "#d62728"
GPU_NEUTRAL = "#d9d9d9"

CORES_PER_NUMA = 12
NUMA_GRID = (4, 3)  # cores inside one NUMA domain: 4 columns x 3 rows
SOCKET_NUMA_GRID = (2, 2)  # NUMA domains per socket: 2x2

CORE_SIZE = 0.34
CORE_GAP = 0.07
NUMA_PAD = 0.12
NUMA_GAP = 0.55
SOCKET_GAP = 1.10
NODE_GAP = 1.2
BUS_DROP = 0.9

NUMA_W = NUMA_GRID[0] * CORE_SIZE + (NUMA_GRID[0] - 1) * CORE_GAP + 2 * NUMA_PAD
NUMA_H = NUMA_GRID[1] * CORE_SIZE + (NUMA_GRID[1] - 1) * CORE_GAP + 2 * NUMA_PAD
SOCKET_W = SOCKET_NUMA_GRID[0] * NUMA_W + (SOCKET_NUMA_GRID[0] - 1) * NUMA_GAP
SOCKET_H = SOCKET_NUMA_GRID[1] * NUMA_H + (SOCKET_NUMA_GRID[1] - 1) * NUMA_GAP + 0.75
NODE_H = 2 * SOCKET_H + SOCKET_GAP

GPU_NUMAS_ALL = (0, 2, 4, 6)   # GPU k sits on NUMA 2k on c23g
GPU_NUMAS = GPU_NUMAS_ALL       # sliced to the config's GPU count in main()
NUM_GPUS = 4                    # set from --num-gpus in main()
LAYOUTS = ("default", "affine", "numa")


def log_lines(path: Path, cell: str | None = None) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    starts = [(i, re.search(r"cell run_id=(\S+)", line).group(1))
              for i, line in enumerate(lines) if "cell run_id=" in line]
    if not starts:
        if cell:
            raise ValueError(f"{path}: no cell markers for {cell}")
        return lines
    selected = [(i, name) for i, name in starts if name == cell]
    if len(selected) != 1:
        raise ValueError(f"{path}: select exactly one cell run ID (got {cell!r})")
    start, name = selected[0]
    end = next((i for i, _ in starts if i > start), len(lines))
    section = lines[start:end]
    if f"CELL_RESULT run_id={name} rc=0" not in section:
        raise ValueError(f"{path}: cell {name} did not finish successfully")
    print(f"Source: {path.name}:{start + 1}-{end}, cell={name}")
    return section


def parse_records(lines, regex, value):
    records = {}
    for line in lines:
        for match in regex.finditer(line):
            rank = int(match.group(1))
            record = value(match)
            if rank in records and records[rank] != record:
                raise ValueError(f"Conflicting records for rank {rank}")
            records[rank] = record
    if not records:
        raise ValueError(f"No records matching {regex.pattern}")
    return records


def parse_binding(path: Path, cell: str | None = None) -> dict[int, int]:
    return parse_records(log_lines(path, cell), BINDING_RE, lambda m: int(m.group(2)))


def parse_groups(path: Path, cell: str | None = None) -> dict[int, tuple[int, int]]:
    return parse_records(log_lines(path, cell), GROUP_RE,
                         lambda m: (int(m.group(2)), int(m.group(3))))


def parse_hosts(path: Path, cell: str | None = None) -> tuple[dict[int, int], int]:
    records = parse_records(log_lines(path, cell), HOST_RE,
                            lambda m: (int(m.group(3)), int(m.group(4))))
    counts = {count for _, count in records.values()}
    if len(counts) != 1:
        raise ValueError("Inconsistent host counts")
    return {rank: host for rank, (host, _) in records.items()}, counts.pop()


def compute_controllers(layout: str, binding: dict[int, int], hosts: dict[int, int],
                        gpu_host: int) -> list[int]:
    if layout == "default":
        gpu_ranks = sorted(r for r in binding if hosts.get(r) == gpu_host)
        return gpu_ranks[:NUM_GPUS]
    controllers: list[int] = []
    for numa in GPU_NUMAS:
        candidates = sorted(r for r in binding
                            if hosts.get(r) == gpu_host and binding[r] == numa)
        if candidates:
            controllers.append(candidates[0])
    return controllers


def compute_gpu_groups(layout: str, binding: dict[int, int], hosts: dict[int, int],
                       gpu_host: int, group_of: dict[int, tuple[int, int]]) -> dict[int, int]:
    controllers = compute_controllers(layout, binding, hosts, gpu_host)
    gpu_group: dict[int, int] = {}
    if layout == "default":
        gpu_ranks = sorted(r for r in binding if hosts.get(r) == gpu_host)
        for ctrl in controllers:
            if ctrl in gpu_ranks:
                gpu_group[gpu_ranks.index(ctrl)] = group_of[ctrl][0]
    else:
        for ctrl in controllers:
            numa = binding.get(ctrl)
            if numa in GPU_NUMAS:
                gpu_group[GPU_NUMAS.index(numa)] = group_of[ctrl][0]
    return gpu_group


def draw_numa(ax, x0, y0, ranks, group_of, numa_index, gpu_index, gpu_group, gpu_side,
              label_below):
    ax.add_patch(Rectangle((x0, y0), NUMA_W, NUMA_H, facecolor="white",
                           edgecolor="#888888", linewidth=1.0, zorder=1))
    for idx, rank in enumerate(sorted(ranks)):
        cx, cy = idx % NUMA_GRID[0], idx // NUMA_GRID[0]
        sx = x0 + NUMA_PAD + cx * (CORE_SIZE + CORE_GAP)
        sy = y0 + NUMA_PAD + (NUMA_GRID[1] - 1 - cy) * (CORE_SIZE + CORE_GAP)
        group, order = group_of[rank]
        edge = CONTROLLER_EDGE if order == 0 else "#333333"
        lw = 2.2 if order == 0 else 0.6
        ax.add_patch(Rectangle((sx, sy), CORE_SIZE, CORE_SIZE,
                               facecolor=GROUP_COLORS[group % len(GROUP_COLORS)],
                               edgecolor=edge, linewidth=lw, zorder=3))
        ax.text(sx + CORE_SIZE / 2, sy + CORE_SIZE / 2, str(rank),
                ha="center", va="center", fontsize=10, color="black", zorder=4)

    if label_below:
        ax.text(x0 + 0.05, y0 - 0.06, f"NUMA {numa_index}",
                ha="left", va="top", fontsize=11, zorder=5)
    else:
        ax.text(x0 + 0.05, y0 + NUMA_H + 0.06, f"NUMA {numa_index}",
                ha="left", va="bottom", fontsize=11, zorder=5)

    if gpu_index is not None:
        bar_w, bar_h = 0.90, 0.32
        bx = x0 + (NUMA_W - bar_w) / 2
        by = y0 + NUMA_H + 0.02 if gpu_side == "above" else y0 - bar_h - 0.02
        color = GROUP_COLORS[gpu_group[gpu_index]]
        ax.add_patch(FancyBboxPatch((bx, by), bar_w, bar_h,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=color, edgecolor="black",
                                    linewidth=0.9, zorder=6))
        ax.text(bx + bar_w / 2, by + bar_h / 2, f"GPU {gpu_index}",
                ha="center", va="center", fontsize=9, color="white", zorder=7, weight="bold")


def draw_node(ax, x0, y0, node_label, gpu, ranks_by_numa, group_of, gpu_group):
    for socket in range(2):
        sy = y0 + (1 - socket) * (SOCKET_H + SOCKET_GAP)
        ax.add_patch(FancyBboxPatch((x0 - 0.16, sy - 0.16), SOCKET_W + 0.32, SOCKET_H + 0.32,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor="#f4f4f4", edgecolor="#555555",
                                    linewidth=1.4, zorder=0))
        ax.text(x0 - 0.29, sy + SOCKET_H / 2, f"socket {socket}",
                ha="center", va="center", rotation=90, fontsize=12, weight="bold",
                color="#333333", zorder=5)

        centers: dict[int, tuple[float, float]] = {}
        for slot in range(4):
            numa = socket * 4 + slot
            col, row = slot % 2, slot // 2
            nx = x0 + col * (NUMA_W + NUMA_GAP)
            ny = sy + 0.42 + (1 - row) * (NUMA_H + NUMA_GAP)
            gpu_index = None
            gpu_side = "above"
            if gpu and numa in GPU_NUMAS:
                gpu_index = GPU_NUMAS.index(numa)
                gpu_side = "above" if row == 0 else "below"
            draw_numa(ax, nx, ny, sorted(ranks_by_numa.get(numa, [])),
                      group_of, numa, gpu_index, gpu_group, gpu_side,
                      label_below=(row == 0))
            centers[slot] = (nx + NUMA_W / 2, ny + NUMA_H / 2)

        for a, b, orient in [(0, 1, "h"), (2, 3, "h"), (0, 2, "v"), (1, 3, "v")]:
            (x1, y1), (x2, y2) = centers[a], centers[b]
            if orient == "h":
                p1 = (x1 + NUMA_W / 2, y1)
                p2 = (x2 - NUMA_W / 2, y2)
            else:
                # Keep the vertical EMIB link clear of the enlarged NUMA labels.
                link_x = x1 + NUMA_W / 2 - 0.20
                p1 = (link_x, y1 - NUMA_H / 2)
                p2 = (link_x, y2 + NUMA_H / 2)
            ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<->", color="#aaaaaa",
                                         linewidth=0.9, zorder=2))

    ax.add_patch(FancyBboxPatch((x0 - 0.42, y0 - 0.42), SOCKET_W + 0.84, NODE_H + 0.84,
                                boxstyle="round,pad=0.02,rounding_size=0.16",
                                facecolor="none", edgecolor="#111111",
                                linewidth=2.0, zorder=-1))
    x_upi = x0 + SOCKET_W / 2
    y_lo = y0 + SOCKET_H + 0.12
    y_hi = y0 + SOCKET_H + SOCKET_GAP - 0.12
    ax.add_patch(FancyArrowPatch((x_upi, y_lo), (x_upi, y_hi), arrowstyle="<->",
                                 color="#444444", linewidth=1.5, zorder=2))
    ax.text(x_upi + 0.16, (y_lo + y_hi) / 2, "UPI", ha="left", va="center",
            fontsize=11, color="#333333", zorder=4)
    ax.text(x0 + SOCKET_W / 2, y0 + NODE_H + 0.72, node_label, ha="center", va="bottom",
            fontsize=17, weight="bold", zorder=5)
    return SOCKET_W


def build_legend(host_count: int):
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=8,
                      markerfacecolor="white", markeredgecolor="#333333",
                      label="core (rank)")]
    handles += [Patch(facecolor=GROUP_COLORS[g], edgecolor="none",
                      label=f"workgroup {g}") for g in range(NUM_GPUS)]
    handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                          markerfacecolor="white", markeredgecolor=CONTROLLER_EDGE,
                          markeredgewidth=2.0, label="GPU controller"))
    handles.append(Patch(facecolor=GPU_NEUTRAL, edgecolor="black", label="GPU"))
    handles.append(Line2D([0], [0], color="#aaaaaa", label="EMIB"))
    handles.append(Line2D([0], [0], color="#444444", linewidth=1.5, label="UPI"))
    if host_count > 1:
        handles.append(Line2D([0], [0], color="#7b3294", linewidth=2.5, label="InfiniBand"))
    return handles


def draw_layout(ax, config, layout, binding, hosts, host_count, gpu_host, group_of,
                title):
    if host_count == 1:
        node_labels = [f"c23g\n96 cores, {NUM_GPUS} active H100s"]
        node_gpus = [True]
    else:
        node_labels = [f"c23g CPU-side\n96 cores, GPUs hidden" if h != gpu_host else
                       f"c23g GPU-side\n96 cores, {NUM_GPUS} active H100s"
                       for h in range(host_count)]
        node_gpus = [h == gpu_host for h in range(host_count)]

    gpu_group = compute_gpu_groups(layout, binding, hosts, gpu_host, group_of)
    ranks_by_host_numa: dict[int, dict[int, list[int]]] = {}
    for rank, numa in binding.items():
        ranks_by_host_numa.setdefault(hosts.get(rank, 0), {}).setdefault(numa, []).append(rank)

    if host_count == 1:
        node_w = draw_node(ax, 0.0, 0.0, node_labels[0], node_gpus[0],
                           ranks_by_host_numa.get(0, {}), group_of, gpu_group)
        ax.set_xlim(-0.6, node_w + 0.6)
        ax.set_ylim(-0.6, NODE_H + 1.8)
    else:
        # Lay the nodes out side by side (host 0 left ... host N-1 right) and
        # connect each to a single vertical InfiniBand bus that runs in the gap
        # between the nodes.
        node_gap_x = 1.65
        xs = [h * (SOCKET_W + node_gap_x) for h in range(host_count)]
        for h in range(host_count):
            draw_node(ax, xs[h], 0.0, node_labels[h], node_gpus[h],
                      ranks_by_host_numa.get(h, {}), group_of, gpu_group)
        node_w = xs[-1] + SOCKET_W
        bus_x = xs[0] + SOCKET_W + node_gap_x / 2.0
        y_mid = NODE_H / 2.0
        ax.plot([bus_x, bus_x], [0.10, NODE_H - 0.10], color="#7b3294", linewidth=3.0,
                solid_capstyle="round", zorder=2)
        for h in range(host_count):
            x_edge = (xs[h] + SOCKET_W + 0.42) if xs[h] < bus_x else (xs[h] - 0.42)
            ax.plot([x_edge, bus_x], [y_mid, y_mid], color="#7b3294", linewidth=2.0,
                    zorder=2)
        ax.text(bus_x + 0.12, y_mid + 0.25, "IB", ha="left", va="bottom",
                  fontsize=10, color="#5a1d6e", zorder=4)
        ax.set_xlim(-0.6, node_w + 0.6)
        ax.set_ylim(-0.6, NODE_H + 1.8)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=20, weight="bold", pad=3)


def save_figure(fig, out: Path, metadata: dict):
    out.parent.mkdir(parents=True, exist_ok=True)
    if PGF:
        ts.save(fig, out.with_suffix(""))
        print(f"Wrote {out.with_suffix('.pgf')} (+ .pdf, .png)")
    else:
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")
    metadata_path = out.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Wrote {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIx rank/workgroup layout diagram")
    parser.add_argument("--config", required=True,
                        help="Config label (e.g. 96c4g, 48c2g, 24p24c1g)")
    parser.add_argument("--nodes", type=int, default=1, choices=[1, 2],
                         help="1 = single c23g node, 2 = two-c23g surrogate")
    parser.add_argument("--num-gpus", type=int, default=4, choices=range(1, 5),
                        help="Number of GPUs per c23g node (1-4)")
    parser.add_argument("--layout", choices=LAYOUTS, default=None)
    parser.add_argument("--combined", action="store_true",
                        help="Draw all three layouts side by side with one legend")
    parser.add_argument("--binding-log", type=Path, required=True,
                        help="Log with 'is bound to NUMA node' lines")
    parser.add_argument("--group-log", type=Path, default=None,
                        help="Single-layout log with 'will be in group ... order' lines")
    parser.add_argument("--default-group-log", type=Path, default=None)
    parser.add_argument("--affine-group-log", type=Path, default=None)
    parser.add_argument("--numa-group-log", type=Path, default=None)
    parser.add_argument("--host-log", type=Path, default=None)
    parser.add_argument("--binding-cell", help="Exact run ID for the binding section")
    parser.add_argument("--host-cell", help="Exact run ID for the host section")
    parser.add_argument("--group-cell", help="Exact run ID for a single-layout section")
    for layout in LAYOUTS:
        parser.add_argument(f"--{layout}-cell", help=f"Exact run ID for {layout} groups")
    parser.add_argument("--binding-note", default="",
                        help="Disclosure saved in companion metadata, not displayed on the figure")
    parser.add_argument("--gpu-host", type=int, default=None)
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--pgf", action="store_true",
                        help="Write .pgf/.pdf via thesis_style instead of .png (handled pre-import)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    binding = parse_binding(args.binding_log, args.binding_cell)
    global NUM_GPUS, GPU_NUMAS
    NUM_GPUS = args.num_gpus
    GPU_NUMAS = GPU_NUMAS_ALL[:NUM_GPUS]
    hosts: dict[int, int] = {}
    host_count = args.nodes
    if args.host_log and args.host_log.exists():
        hosts, host_count = parse_hosts(args.host_log, args.host_cell)
    if host_count != args.nodes:
        parser.error("Host log count disagrees with --nodes")
    if not hosts:
        if host_count > 1:
            parser.error("Multi-node diagrams require an observed --host-log")
        hosts = {rank: 0 for rank in binding}
    gpu_host = args.gpu_host if args.gpu_host is not None else host_count - 1
    if set(hosts) != set(binding) or set(hosts.values()) != set(range(host_count)):
        parser.error("Incomplete or invalid host assignments")
    if gpu_host not in range(host_count) or any(n not in range(8) for n in binding.values()):
        parser.error("Invalid GPU host or NUMA domain")
    for host in range(host_count):
        for numa in range(8):
            if sum(hosts[r] == host and n == numa for r, n in binding.items()) > CORES_PER_NUMA:
                parser.error("More ranks than physical cores in a NUMA domain")

    layout_logs = {
        "default": args.default_group_log,
        "affine": args.affine_group_log,
        "numa": args.numa_group_log,
    }

    def groups_for(layout: str):
        log = layout_logs.get(layout)
        cell = getattr(args, f"{layout}_cell")
        if layout == args.layout and args.group_log:
            log = args.group_log
            cell = args.group_cell
        if log is None:
            parser.error(f"Observed group log required for {layout}")
        if cell and f"_{layout}_" not in cell:
            parser.error(f"{layout}: selected run ID names a different layout")
        lines = log_lines(log, cell)
        local = parse_records(
            lines, re.compile(r"Rank (\d+)/\d+ on its local machine is (\d+)/(\d+)"),
            lambda m: (int(m.group(1)) - int(m.group(2)), int(m.group(3))))
        starts = sorted({start for start, _ in local.values()})
        if set(local) != set(binding) or len(starts) != host_count:
            parser.error(f"{layout}: incomplete local-machine records")
        if any(hosts[r] != starts.index(start) for r, (start, _) in local.items()):
            parser.error(f"{layout}: local-machine records disagree with host reference")
        if any(BINDING_RE.search(line) for line in lines):
            observed = parse_records(lines, BINDING_RE, lambda m: int(m.group(2)))
            if observed != binding:
                parser.error(f"{layout}: section binding differs from reference binding")
        elif not args.binding_note:
            parser.error(f"{layout}: missing binding records; disclose reference via --binding-note")
        if layout != "default":
            attachments = parse_records(
                lines, re.compile(r"GPU (\d+) is attached to NUMA node (\d+)"),
                lambda m: int(m.group(2)))
            if attachments != dict(enumerate(GPU_NUMAS)):
                parser.error(f"{layout}: GPU attachments disagree with drawn topology")
        groups = parse_groups(log, cell)
        world_sizes = {int(m.group(1)) for line in lines
                       for m in re.finditer(r"Rank \d+/(\d+) will be in group", line)}
        if world_sizes != {len(groups)} or set(groups) != set(range(len(groups))):
            parser.error(f"{layout}: incomplete MPI world assignments")
        if set(groups) != set(binding):
            parser.error(f"{layout}: binding and group rank sets differ")
        if any(g not in range(NUM_GPUS) or order < 0 for g, order in groups.values()):
            parser.error(f"{layout}: invalid group or order")
        controllers = {r for r, (_, order) in groups.items() if order == 0}
        if controllers != set(compute_controllers(layout, binding, hosts, gpu_host)):
            parser.error(f"{layout}: observed controllers disagree with topology")
        if {groups[r][0] for r in controllers} != set(range(NUM_GPUS)):
            parser.error(f"{layout}: expected one controller per workgroup")
        print(f"{layout}: {len(groups)} observed ranks; controllers "
              f"{[(r, groups[r][0], binding[r]) for r in sorted(controllers)]}")
        return groups

    handles = build_legend(host_count)

    if args.combined:
        fig_w = 12.0 if host_count == 1 else 24.0
        # Match the axes' physical aspect ratio to avoid equal-aspect padding.
        panel_w = SOCKET_W + 1.2 if host_count == 1 else 2 * SOCKET_W + 2.85
        fig_h = (fig_w / 3) * (NODE_H + 2.4) / panel_w + 0.9
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
        for ax, layout in zip(axes, LAYOUTS):
            draw_layout(ax, args.config, layout, binding, hosts, host_count, gpu_host,
                        groups_for(layout), title=None if args.no_title else f"layout '{layout}'")
        fig.legend(handles=handles, loc="lower center",
                    ncol=5 if host_count == 1 else len(handles),
                    frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=15)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.95,
                             bottom=0.11, wspace=0.025)
        save_figure(fig, args.out, vars(args))
        return

    if args.layout is None:
        parser.error("--layout is required unless --combined is used")
    fig, ax = plt.subplots(figsize=(5.6 if host_count == 1 else 14.0,
                                    8.6 if host_count == 1 else 6.2))
    single_title = None if args.no_title else f"{args.config} layout: {args.layout}"
    draw_layout(ax, args.config, args.layout, binding, hosts, host_count, gpu_host,
                groups_for(args.layout), title=single_title)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005),
              ncol=3, frameon=False, fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.20)
    save_figure(fig, args.out, vars(args))


if __name__ == "__main__":
    main()
