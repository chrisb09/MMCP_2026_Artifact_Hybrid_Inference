#!/usr/bin/env python3
"""
Point-trace line graph for one cell across all inference cycles.
Shows reference sent history, reference received, and new CMI received
for each cycle at the cell with maximum divergence.
"""
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REF = Path("/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5")
DUMPS = ROOT / "debug_dumps"
OUT_DIR = ROOT / "analysis"
FIELD = "U"
FIELD_IDX = 0
GRID = (67, 69, 126)


def attr_index(h5):
    idx = {}
    for name in h5.keys():
        grp = h5[name]
        ts = grp.attrs.get("globalTimeStep")
        tp = grp.attrs.get("type")
        if isinstance(tp, bytes):
            tp = tp.decode()
        if ts is not None and tp is not None:
            idx[(int(ts), tp)] = name
    return idx


def load_reference_cycles(h5file, idx):
    """Return list of cycles: each is list of (label, value) for 5 sent + 1 received."""
    cycles = []
    # reference cycles: (send_steps, received_step)
    ref_cycles = [
        ([11, 12, 13, 14, 15], 15),
        ([47, 48, 49, 50, 51], 51),
    ]
    with h5py.File(h5file, "r") as h5:
        for sends, recv in ref_cycles:
            pairs = []
            for step in sends:
                gname = idx.get((step, "sent"))
                if gname is None:
                    break
                v = np.array(h5[gname][FIELD]).flatten().astype(np.float32)
                pairs.append((f"ref sent {step}", v))
            gname = idx.get((recv, "received"))
            if gname:
                v = np.array(h5[gname][FIELD]).flatten().astype(np.float32)
                pairs.append((f"ref recv {recv}", v))
            cycles.append(pairs)
    return cycles


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Determine flat_idx from max diff in cycle 1
    with h5py.File(REF, "r") as h5:
        idx = attr_index(h5)
        ref_recv_1 = np.array(h5[idx[(15, "received")]][FIELD]).flatten().astype(np.float32)
    cpp_recon_1 = np.fromfile(DUMPS / "step0004_rank0_reconstructed_output_field0.bin", dtype=np.float32)
    flat_idx = int(np.argmax(np.abs(cpp_recon_1 - ref_recv_1)))
    coords = np.unravel_index(flat_idx, GRID)

    # Load reference data
    ref_cycles = load_reference_cycles(REF, idx)

    # Load C++ received data for each inference cycle
    cpp_cycles = [
        ("new CMI recv ~15",
         np.fromfile(DUMPS / "step0004_rank0_reconstructed_output_field0.bin", dtype=np.float32)[flat_idx]),
        ("new CMI recv ~39",
         np.fromfile(DUMPS / "step0009_rank0_reconstructed_output_field0.bin", dtype=np.float32)[flat_idx]),
    ]

    # Build all labels/values for plot
    labels = []
    values = []
    colors = []
    cycle_starts = []

    for ci, cycle in enumerate(ref_cycles):
        cycle_starts.append(len(labels))
        for lbl, arr in cycle:
            val = float(arr[flat_idx])
            labels.append(lbl)
            values.append(val)
            colors.append("#1f77b4")
        # Add C++ received for this cycle
        cpp_lbl, cpp_val = cpp_cycles[ci]
        labels.append(cpp_lbl)
        values.append(cpp_val)
        colors.append("#2ca02c")

    # Write CSV
    csv_path = OUT_DIR / "point_trace_all_cycles_U.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "value"])
        writer.writerows(zip(labels, values))

    # Summary
    summary_path = OUT_DIR / "point_trace_all_cycles_U_summary.txt"
    lines = [
        f"field: {FIELD}",
        f"flat_idx: {flat_idx}",
        f"coords_zyx: {coords}",
        "",
        "Cycle 1 (global ~11-15):",
        f"  ref recv: {values[cycle_starts[0]+5]:.9e}",
        f"  new CMI recv: {cpp_cycles[0][1]:.9e}",
        "",
        "Cycle 2 (global ~47-51):",
        f"  ref recv: {values[cycle_starts[1]+5]:.9e}" if len(cycle_starts) > 1 else "  ref recv: N/A",
        f"  new CMI recv: {cpp_cycles[1][1]:.9e}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    # Plot
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(labels))

    # Draw all points
    ax.scatter(x, values, c=colors, s=30, zorder=3)

    # Draw reference sent lines within each cycle
    for ci in range(len(ref_cycles)):
        cs = cycle_starts[ci]
        n_ref_points = len(ref_cycles[ci])
        ax.plot(x[cs:cs + n_ref_points], values[cs:cs + n_ref_points],
                "o-", color="#1f77b4", linewidth=1.5, markersize=5)
        # Dashed line from last ref point to C++ received
        cpp_x = cs + n_ref_points
        ax.plot([x[cpp_x - 1], x[cpp_x]], [values[cpp_x - 1], values[cpp_x]],
                "--", color="#2ca02c", linewidth=1.5, alpha=0.7)

    # Vertical divider between cycles
    if len(cycle_starts) > 1:
        ax.axvline(x=cycle_starts[1] - 0.5, color="#999999", linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"{FIELD} value at flat_idx={flat_idx}")
    ax.set_title(f"Point trace across inference cycles for max {FIELD} divergence\n"
                 f"coords z/y/x={coords}  |  blue=reference  |  green=new CMI")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#1f77b4", marker="o", linewidth=1.5, label="reference"),
        Line2D([0], [0], color="#2ca02c", marker="s", linestyle="--", label="new CMI recv"),
    ]
    ax.legend(handles=handles, loc="best")

    fig.tight_layout()
    png_path = OUT_DIR / "point_trace_all_cycles_U.png"
    fig.savefig(png_path, dpi=160)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
