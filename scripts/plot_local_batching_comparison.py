#!/usr/bin/env python3
"""Plot validated local batching comparisons; never submit jobs or rewrite inputs."""
import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/local_batching_comparison"
STEPS = (15, 44, 73, 102, 131, 160, 189)
PLACEMENTS = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
T95 = 4.302652729911275


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def values(text):
    result = [float(x) for x in text.split(";")]
    require(all(math.isfinite(x) and x > 0 for x in result), "invalid timing")
    return result


def check_log(row, expected):
    path = ROOT / row["run_dir"] / "solver.log"
    text = path.read_text(errors="strict")
    calls = re.findall(r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)", text)
    require(text.count("ML_STEP_TIMING step=") == len(calls) == 7, f"timing count: {path}")
    require(tuple(int(step) for step, _ in calls) == STEPS, f"timing steps: {path}")
    logged = [float(value) for _, value in calls]
    require(all(math.isfinite(x) and x > 0 for x in logged), f"bad timing: {path}")
    require(logged == expected, f"analysis/log mismatch: {path}")


def close(a, b, tolerance=1e-6):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=tolerance)


def load():
    phy_dir = ROOT / "analysis/phydll_cap_comparison_local"
    smart_dir = ROOT / "analysis/smartsim_aggregation_comparison_local"
    groups = defaultdict(list)
    phy = rows(phy_dir / "observations.csv")
    checks = rows(phy_dir / "validation.csv")
    require(len(checks) == 24 and all(r["status"] == "ok" for r in checks), "PhyDLL validation failures")
    require(len(phy) == 48, "PhyDLL run count")
    for r in phy:
        measured = values(r["measured_ms"])
        require(len(measured) == 5, f"PhyDLL measured count: {r['run_dir']}")
        path = ROOT / r["run_dir"] / "solver.log"
        raw = re.findall(r"ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)", path.read_text(errors="strict"))
        require(tuple(int(step) for step, _ in raw) == STEPS, f"PhyDLL steps: {path}")
        check_log(r, [float(value) for _, value in raw[:2]] + measured)
        mean = statistics.mean(measured)
        require(close(mean, r["mean_ms"]), f"PhyDLL mean: {path}")
        groups[("phydll", r["wait"], r["placement"], r["cap"])].append((r["rep"], r["jobid"], mean))

    smart = rows(smart_dir / "runs.csv")
    require(len(smart) == 48, "SmartSim run count")
    for r in smart:
        calls = values(r["all_calls_ms"])
        require(len(calls) == 7, f"SmartSim call count: {r['run_dir']}")
        check_log(r, calls)
        mean = statistics.mean(calls[2:])
        require(close(mean, r["measured_mean_ms"]), f"SmartSim mean: {r['run_dir']}")
        groups[("smartsim", r["model"], r["placement"], r["variant"])].append((r["rep"], r["jobid"], mean))

    expected = {(kind, panel, p, v) for kind, panels, variants in (
        ("phydll", ("default", "readiness"), ("2048", "50000")),
        ("smartsim", ("transformer", "mlp"), ("baseline", "aggregation")),
    ) for panel in panels for p in PLACEMENTS for v in variants}
    require(set(groups) == expected, "missing or unexpected configurations")
    stats = {}
    for key, group in groups.items():
        require(len(group) == 3 and {r[0] for r in group} == {"1", "2", "3"} and len({r[1] for r in group}) == 3, f"repetitions: {key}")
        means = [r[2] for r in group]
        mean = statistics.mean(means)
        half = T95 * statistics.stdev(means) / math.sqrt(3)
        stats[key] = (mean, half)

    phy_summary = [r for r in rows(phy_dir / "summary.csv") if r["metric"] == "mean_ms"]
    require(len(phy_summary) == 16, "PhyDLL summary count")
    for r in phy_summary:
        mean, half = stats[("phydll", r["wait"], r["placement"], r["cap"])]
        require(r["n"] == "3" and all(close(a, b) for a, b in ((mean, r["mean"]), (mean - half, r["ci95_low"]), (mean + half, r["ci95_high"]))), "PhyDLL summary mismatch")
    smart_summary = rows(smart_dir / "summary.csv")
    require(len(smart_summary) == 8, "SmartSim summary count")
    for r in smart_summary:
        require(r["n"] == "3", "SmartSim summary n")
        for variant in ("baseline", "aggregation"):
            mean, half = stats[("smartsim", r["model"], r["placement"], variant)]
            require(close(mean, r[f"{variant}_mean_ms"], 0.00051) and close(half, r[f"{variant}_ci95_half_ms"], 0.00051), "SmartSim summary mismatch")
    return stats


def plot(stats, kind, panels, variants, labels, filename, pgf):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "pdf.fonttype": 42,
                         "pgf.texsystem": "xelatex", "pgf.rcfonts": False})
    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 3.3), sharey=True, layout="constrained")
    colors = ("#21618c", "#bf562c")
    all_values = [stats[(kind, panel, placement, variant)]
                  for panel, _ in panels for placement in PLACEMENTS for variant in variants]
    lower = min(0, min((mean - half) / 1000 for mean, half in all_values))
    upper = max((mean + half) / 1000 for mean, half in all_values)
    for ax, (panel, title) in zip(axes, panels):
        for j, (variant, label, color) in enumerate(zip(variants, labels, colors)):
            vals = [stats[(kind, panel, placement, variant)] for placement in PLACEMENTS]
            positions = [i + (-0.19 if j == 0 else 0.19) for i in range(4)]
            ax.bar(positions, [m / 1000 for m, _ in vals], width=0.36, color=color, label=label,
                   yerr=[h / 1000 for _, h in vals],
                   error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2, "capthick": 0.8})
            for x, (mean, half) in zip(positions, vals):
                ax.annotate(f"{mean / 1000:.2f}", (x, (mean + half) / 1000),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(4), PLACEMENTS, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_ylim(lower - (upper - lower) * 0.03 if lower < 0 else 0, upper + (upper - lower) * 0.14)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", color="#d9dfe3", linewidth=0.65)
        ax.tick_params(axis="x", length=0, pad=4)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=2, frameon=False, fontsize=8)
    axes[0].set_ylabel("ML coupling time per call (s)")
    for ext in (["png", "pdf", "pgf"] if pgf else ["png", "pdf"]):
        fig.savefig(OUT / f"{filename}.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgf", action="store_true", help="also export PGF (requires XeLaTeX)")
    args = parser.parse_args()
    stats = load()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "plotted_statistics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("comparison", "panel", "placement", "variant", "n", "mean_ms", "ci95_low_ms", "ci95_high_ms"))
        for key, (mean, half) in sorted(stats.items()):
            writer.writerow((*key, 3, f"{mean:.9f}", f"{mean-half:.9f}", f"{mean+half:.9f}"))
    plot(stats, "phydll", (("default", "Transformer / default"), ("readiness", "Transformer / readiness")),
         ("2048", "50000"), ("Cap 2,048", "Cap 50,000"), "phydll_cap", args.pgf)
    plot(stats, "smartsim", (("transformer", "Transformer"), ("mlp", "MLP")),
         ("baseline", "aggregation"), ("Shared, nonaggregated", "Aggregation"), "smartsim_aggregation", args.pgf)
    print("Verified 96 solver logs, 480 measured calls, 32 groups of 3 independent run means.")
    print(f"Wrote figures and statistics to {OUT}")


if __name__ == "__main__":
    main()
