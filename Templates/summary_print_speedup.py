#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from plotting import plot_parabel  # reuse your existing plot functions
import re
import math
import matplotlib


#matplotlib.rc('pdf', fonttype=42)
#matplotlib.rcParams['font.family'] = 'sans-serif'
#matplotlib.rcParams['font.sans-serif'] = 'DeJaVu Sans'
#matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{sansmath} \sansmath'
#matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
#matplotlib.rcParams['mathtext.rm'] = 'DeJaVu Sans'
#matplotlib.rcParams['mathtext.it'] = 'DeJaVu Sans:italic'
#matplotlib.rcParams['mathtext.bf'] = 'DeJaVu Sans:bold'
def get_case_sort_key(case):
    """Sort by GPUs first, then CPU count."""
    match = re.match(r"(\d+).*?cpu.*?(\d+).*?gpu", case.lower())
    if match:
        cpu = int(match.group(1))
        gpu = int(match.group(2))
    else:
        cpu, gpu = (0, 0)
    return (gpu, cpu)

def get_min_runtime_info(experiments_path):
    """
    Find the worksplit (folder name) that gives the minimal runtime
    from the pre-generated dumps in plotting.py output.
    Returns (optimal_worksplit, min_runtime).
    """
    runtimes = {}
    for subdir, dirs, files in os.walk(experiments_path):
        base = os.path.basename(subdir)
        if base.startswith("0,") or base.startswith("1,"):
            # Heuristic: look for ml_step dump or output
            ml_step_path = os.path.join(subdir, "mlStep_incl.dump")
            if os.path.exists(ml_step_path):
                # Get runtime directly from dump if possible
                with open(ml_step_path, "r") as f:
                    lines = f.readlines()
                for line in lines[7:]:
                    if "MLCouplingMaia::ml_step" in line:
                        try:
                            vals = [float(x) for x in line.split()[1:] if float(x) > 1e-9]
                            runtimes[base] = max(vals)  
                        except Exception:
                            pass
                        break
    if not runtimes:
        return None, None
    optimal = min(runtimes, key=runtimes.get)
    return optimal, runtimes[optimal], runtimes 

def plotA(real_speeds, order):
    # --- Plot 1: Real speedup vs node count ---
    plt.figure(figsize=(6, 4))
    ordered_cases = [c for c in order if c in real_speeds]
    plt.bar(ordered_cases, [real_speeds[c] for c in ordered_cases], label="Real Speedup")
    plt.xlabel("Case (Nodes - GPUs)")
    plt.ylabel("Speedup")
    plt.title("Speedup of best Worksplit per Case")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("summary_speedup_vs_case.pdf")
    plt.close()
    print("Saved plotA: summary_speedup_vs_case.pdf")

def plotB(real_speeds, expected_speeds, expected_opt_splits, real_opt_splits, order):
    # --- Plot 2: Difference between expected and real speedup + worksplit ---
    common = set(real_speeds) & set(expected_speeds)
    diff_speedup = {c: expected_speeds[c] - real_speeds[c] for c in common}
    diff_worksplit = {}

    for c in common:
        try:
            diff = (float(str(expected_opt_splits[c]).replace(",", ".")) -
                       float(str(real_opt_splits[c]).replace(",", "."))) #abs
        except Exception:
            diff = np.nan
        diff_worksplit[c] = diff

    ordered_cases = [c for c in order if c in common]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Bars for Δ speedup
    bar1 = ax1.bar(ordered_cases, [diff_speedup[c] for c in ordered_cases], color="steelblue", alpha=0.7, label="Δ Speedup")
    # Line for Δ worksplit
    line1, = ax2.plot(ordered_cases, [diff_worksplit[c] for c in ordered_cases], color="crimson", marker="o", label="Δ Optimal CPU Fraction")

    ax1.set_xlabel("Case (Nodes - GPUs)")
    ax1.set_ylabel("Δ Speedup", color="steelblue")
    ax2.set_ylabel("Δ CPU Fraction", color="crimson")
    plt.title("Expected vs Real Differences per Case", y=1.10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle="--", alpha=0.5)
    # Combine legends
    handles = [bar1, line1]
    labels = [h.get_label() if hasattr(h, "get_label") else h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)

    plt.tight_layout()
    plt.savefig("summary_expected_vs_real_diff.pdf")
    plt.close()
    print("Saved plotB: summary_expected_vs_real_diff.pdf")

# -------------------------------------
#  Plot C – Speedup (Real + Expected)
# -------------------------------------
def plotC(real_speeds, expected_speeds, order):
    plt.figure(figsize=(6, 4))
    ordered_cases = [c for c in order if c in real_speeds or c in expected_speeds]

    width = 0.35
    x = np.arange(len(ordered_cases))
    expected_speedup = [expected_speeds.get(c, np.nan) for c in ordered_cases]
    real_speedup = [real_speeds.get(c, np.nan) for c in ordered_cases]
    abs_err = [e - r for (e,r) in list(zip(expected_speedup, real_speedup))]
    rel_err = [(e - r)/r for (e,r) in list(zip(expected_speedup, real_speedup))]

    print(f"Expected speedups = {expected_speedup}")
    print(f"Real speedups = {real_speedup}")
    print(f"Abs error = {abs_err}")
    print(f"Rel error = {rel_err}")

    plt.bar(x-width/2, expected_speedup, width=width, label="Theoretical Speedup")
    plt.bar(x+width/2, real_speedup, width=width, label="Real Speedup")
    #plt.xticks(x, ordered_cases, rotation=45, ha="right")

    plt.xlabel("Case (Nodes - GPUs)")
    plt.ylabel("Speedup")
    plt.title("Real vs Expected Speedup per Case")
    x_labels = []
    for name in ordered_cases:
        # Find where the "Nodes" and "GPU" parts are based on capitalization
        # Assume format is like "4CPUNodes2GPU"
        i_node = name.find("CPU")
        i_gpu = name.find("GPU")
        
        # Extract numbers before "CPU" and before "GPU"
        n_nodes = name[:i_node]
        n_gpus = name[i_node + 8:i_gpu]  # skip "CPUNodes"
        
        if n_nodes == "1":
            label = f"{n_nodes} Node \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        else:
            label = f"{n_nodes} Nodes \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        x_labels.append(label)
    plt.xticks(x, x_labels)# plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("summary_real_vs_expected_speedup.pdf")
    plt.close()
    print("Saved plotC: summary_real_vs_expected_speedup.pdf")


# -------------------------------------
#  Plot D – Speedup vs Worksplit per Case
# -------------------------------------
def plotD(all_runtimes):
    # Separate cases into groups by GPUs keyword
    gpu_groups = {"1GPU": {}, "4GPU": {}}
    for case, data in all_runtimes.items():
        if "4GPU" in case or "4Gpu" in case or "4gpu" in case:
            gpu_groups["4GPU"][case] = data
        else:
            gpu_groups["1GPU"][case] = data

    fig, axes = plt.subplots(1, 2, figsize=(8.27, 3), sharey=False)
    #fig.suptitle("Speedup vs CPU Fraction per Case", fontsize=12)

    for ax_idx, (gpu_label, cases) in enumerate(gpu_groups.items()):
        ax = axes[ax_idx]
        all_x_vals  = set()

        for case, runtimes in cases.items():
            parsed = []
            baseline_time = None

            # Parse worksplit strings to floats and find baseline
            for k, v in runtimes.items():
                try:
                    val = float(k.replace(",", "."))
                    parsed.append((val, v))
                    all_x_vals.add(val)
                    if k == "0,00":  # baseline for this case
                        baseline_time = v
                except ValueError:
                    continue

            if not parsed or baseline_time is None:
                continue

            parsed.sort(key=lambda x: x[0])
            worksplits = [p[0] for p in parsed]
            speeds = [baseline_time / p[1] for p in parsed]  # speedup relative to 0,00 worksplit

            ax.plot(worksplits, speeds, marker="o", label=case, markersize=4, zorder=1+1/int(case[:1]))
    
        if all_x_vals:
            min_x, max_x = min(all_x_vals), max(all_x_vals)
            # Create ticks only at clean 0.1 steps
            major_ticks = np.arange(math.floor(min_x * 10) / 10,
                                    math.ceil(max_x * 10) / 10 + 0.1,
                                    0.1)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])

        ax.set_xlabel("CPU Fraction")
        if ax_idx == 0:
            ax.set_ylabel("Speedup")
        ax.set_title(f"{gpu_label[:1]} {gpu_label[1:4]} Cases")
        ax.grid(True, linestyle="--", alpha=0.6)
        #ax.legend(fontsize=8)
        #for tick in ax.get_xticklabels():
            #tick.set_rotation(45)
            #tick.set_ha("right")
        ax.set_yticks(np.arange(0, math.ceil(max(speeds) * 2) / 2 + 0.5, 0.5))
        ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()])

    cpus = [name[:1] for name in gpu_groups["1GPU"].keys()]
    plt.ylim(bottom=0)
    fig.legend(cpus, loc="lower center", ncol=4,bbox_to_anchor=(0.525, 0.0), fontsize=8, title="Nodes")
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(bottom=0.25)  
    plt.savefig("summary_speedup_vs_worksplit.pdf")
    plt.close()
    print("Saved plotD")

def plotD_2(all_runtimes):
    # Separate cases into groups by GPUs keyword
    gpu_groups = {"1GPU": {}, "4GPU": {}}
    for case, data in all_runtimes.items():
        if "4GPU" in case or "4Gpu" in case or "4gpu" in case:
            gpu_groups["4GPU"][case] = data
        else:
            gpu_groups["1GPU"][case] = data

    for gpu_label, cases in gpu_groups.items():
        fig, ax = plt.subplots(figsize=(5, 3))
        all_x_vals = set()

        for case, runtimes in cases.items():
            parsed = []
            baseline_time = None

            # Parse worksplit strings to floats and find baseline
            for k, v in runtimes.items():
                try:
                    val = float(k.replace(",", "."))
                    parsed.append((val, v))
                    all_x_vals.add(val)
                    if k == "0,00":  # baseline for this case
                        baseline_time = v
                except ValueError:
                    continue

            if not parsed or baseline_time is None:
                continue

            parsed.sort(key=lambda x: x[0])
            worksplits = [p[0] for p in parsed]
            speeds = [baseline_time / p[1] for p in parsed]  # speedup relative to 0,00 worksplit

            ax.plot(worksplits, speeds, marker="o", label=case, markersize=4, zorder=1+1/int(case[:1]))
            print(f"Speedups for GPU-label {gpu_label}, case {case}, speeds = {speeds}")

        if all_x_vals:
            min_x, max_x = min(all_x_vals), max(all_x_vals)
            major_ticks = np.arange(math.floor(min_x * 10) / 10,
                                    math.ceil(max_x * 10) / 10 + 0.1,
                                    0.1)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])

        ax.set_xlabel("CPU Fraction")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{gpu_label[:1]} {gpu_label[1:4]} Cases")
        ax.grid(True, linestyle="--", alpha=0.6)

        if speeds:
            ax.set_yticks(np.arange(0, math.ceil(max(speeds) * 2) / 2 + 0.5, 0.5))
            ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()])

        plt.ylim(bottom=0)

        cpus = [name[:1] for name in cases.keys()]
        #fig.legend(cpus, loc="lower center", ncol=4, bbox_to_anchor=(0.525, 0.0), fontsize=8, title="Nodes")
        plt.legend(cpus, loc="best", title="Nodes")
        plt.tight_layout(pad=0.5)
        #plt.subplots_adjust(bottom=0.25)
        plt.savefig(f"summary_speedup_vs_worksplit_{gpu_label}.pdf")
        plt.close()
        print(f"Saved plot for {gpu_label}")


def plotE(real_speeds, expected_speeds, expected_opt_splits, real_opt_splits, order):
    # --- Plot 2: Difference between expected and real speedup + worksplit ---   
    common = set(real_speeds) & set(expected_speeds)
    ordered_cases = [c for c in order if c in common]

    # Convert German-style strings to floats
    real_vals = []
    expected_vals = []
    abs_err_vals = []
    for c in ordered_cases:
        try:
            real_vals.append(float(str(real_opt_splits[c]).replace(",", ".")))
        except ValueError:
            real_vals.append(np.nan)
        try:
            expected_vals.append(float(str(expected_opt_splits[c]).replace(",", ".")))
        except ValueError:
            expected_vals.append(np.nan)

    for idx, val in enumerate(real_vals):
        abs_err_vals.append(expected_vals[idx] - val)

    width = 0.35
    x = np.arange(len(ordered_cases))

    plt.subplots(figsize=(6, 4))
    plt.bar(x - width/2, expected_vals, width=width, label="Theoretical Fraction")
    plt.bar(x + width/2, real_vals, width=width, label="Real Fraction")

    print(f"Optimal fractions: theo = {expected_vals}")
    print(f"Optimal fractions: real = {real_vals}")
    print(f"Differences = {abs_err_vals}")

    plt.xlabel("Case (Nodes - GPUs)")
    plt.ylabel("CPU Fraction")
    x_labels = []
    for name in ordered_cases:
        # Find where the "Nodes" and "GPU" parts are based on capitalization
        # Assume format is like "4CPUNodes2GPU"
        i_node = name.find("CPU")
        i_gpu = name.find("GPU")
        
        # Extract numbers before "CPU" and before "GPU"
        n_nodes = name[:i_node]
        n_gpus = name[i_node + 8:i_gpu]  # skip "CPUNodes"
                
        if n_nodes == "1":
            label = f"{n_nodes} Node \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        else:
            label = f"{n_nodes} Nodes \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        x_labels.append(label)
    plt.xticks(x, x_labels)
    #plt.xticks(x, ordered_cases, rotation=45, ha="right")
    plt.title("Theoretical vs Real Optimal CPU Fraction per Case")
    #plt.tick_params(axis='x', rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0.5)
    plt.savefig("summary_expected_vs_real_worksplit.pdf")
    plt.close()
    print("Saved plotE: summary_expected_vs_real_worksplit.pdf")

def plotF(expected_speeds, expected_opt_splits, real_opt_splits, all_case_runtimes, order):
    """
    Barplot showing:
    - Theoretical maximum speedup
    - Speedup at theoretical optimum (real runtime at that split)
    - Speedup at actual (real) optimum
    """
    plt.figure(figsize=(7, 4))
    ordered_cases = [c for c in order if c in expected_speeds and c in all_case_runtimes]

    theo_max = []
    theo_opt_real = []
    real_opt = []

    for c in ordered_cases:
        baseline = all_case_runtimes[c].get("0,00", np.nan) 
        theo_split = str(expected_opt_splits[c])[0:4].replace(".",",")
        real_split = real_opt_splits[c]
        theo_speedup = expected_speeds.get(c, np.nan)

        # Real runtime at theoretical optimum
        theo_opt_time = all_case_runtimes[c].get(theo_split, np.nan)
        theo_opt_speedup = baseline / theo_opt_time if theo_opt_time and theo_opt_time > 0 else np.nan

        # Real speedup at real optimum
        real_opt_time = all_case_runtimes[c].get(real_split, np.nan)
        real_opt_speedup = baseline / real_opt_time if real_opt_time and real_opt_time > 0 else np.nan

        theo_max.append(theo_speedup)
        theo_opt_real.append(theo_opt_speedup)
        real_opt.append(real_opt_speedup)

    x = np.arange(len(ordered_cases))
    width = 0.25

    plt.bar(x - width, theo_max, width, label="Theoretical (Speedup @ Best Worksplit)")
    plt.bar(x, theo_opt_real, width, label="Real Speedup @ Theoretical Best Worksplit")
    plt.bar(x + width, real_opt, width, label="Real( Speedup @ Best Worksplit)")

    plt.xlabel("Case (Nodes - GPUs)")
    plt.ylabel("Speedup")
    plt.title("Comparison of Theoretical and Real Speedups per Case")
    plt.xticks(x, ordered_cases, rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("summary_theoretical_vs_real_speedup.pdf")
    plt.close()
    print("Saved plotF: summary_theoretical_vs_real_speedup.pdf")


def plotG(example_case, all_case_runtimes, expected_opt_splits=None, real_opt_splits=None):
    """
    Plot G Theoretical runtime model for one example case.
    Shows linear CPU and GPU runtime assumptions vs worksplit fraction.
    
    - X: CPU fraction (0 → all GPU, 1 → all CPU)
    - Y: Runtime (arbitrary units)
    """
    plt.figure(figsize=(6, 4))
    if example_case not in all_case_runtimes:
        print(f"Case '{example_case}' not found in runtime data!")
        return

    runtimes = all_case_runtimes[example_case]
    worksplits = sorted([float(k.replace(",", ".")) for k in runtimes.keys() if re.match(r"^\d+,\d+$", k)])
    if not worksplits:
        print(f"No numeric worksplits found for case {example_case}")
        return

    exp_dir = os.path.join(example_case, "Experiments")
    if not os.path.exists(exp_dir):
        print(f"No experiment directory found for {example_case}")
        return

    # --- Gather real CPU & GPU inference data ---
    real_cpu_infer = {}
    real_gpu_infer = {}

    for sub in os.listdir(exp_dir):
        worksplit_dir = os.path.join(exp_dir, sub)
        dump_path = os.path.join(worksplit_dir, "mlStep_incl.dump")
        if not os.path.exists(dump_path):
            continue

        try:
            worksplit_val = float(sub.replace(",", "."))
        except ValueError:
            continue

        with open(dump_path, "r") as f:
            lines = f.readlines()

        cpu_vals = []
        gpu_vals = []

        for line in lines:
            if "inferenceHost" in line:
                parts = [float(x) for x in line.split()[1:] if re.match(r"^-?\d+(\.\d+)?$", x)]
                cpu_vals.extend(parts)
            elif "inferenceDevice" in line:
                parts = [float(x) for x in line.split()[1:] if re.match(r"^-?\d+(\.\d+)?$", x)]
                gpu_vals.extend(parts)

        if cpu_vals:
            real_cpu_infer[worksplit_val] = np.mean(cpu_vals)
        if gpu_vals:
            real_gpu_infer[worksplit_val] = np.mean(gpu_vals)


    # Assume baseline pure CPU and pure GPU runtimes
    print(runtimes.get("1,00", max(runtimes.values())))
    i_node = example_case.find("CPU")
    i_gpu = example_case.find("GPU")
    
    # Extract numbers before "CPU" and before "GPU"
    n_nodes = int(example_case[:i_node])
    n_gpus = int(example_case[i_node + 8:i_gpu])  # skip "CPUNodes"        
    cpu_time = runtimes.get("1,00", max(runtimes.values()))*(n_nodes * 96/ (n_nodes * 96- n_gpus))
    print(cpu_time)
    gpu_time = runtimes.get("0,00", min(runtimes.values()))

    # Linear theoretical runtimes (simplified model)
    cpu_line = [cpu_time * w for w in worksplits]
    gpu_line = [gpu_time * (1 - w) for w in worksplits]

    exp_dir = os.path.join(example_case, "Experiments")
    if not os.path.exists(exp_dir):
        print(f"No experiment directory found for {example_case}")
        return

    real_cpu_inf = {}
    real_gpu_inf = {}

    # Reuse same logic as get_min_runtime_info
    for subdir, dirs, files in os.walk(exp_dir):
        base = os.path.basename(subdir)
        dump_path = os.path.join(subdir, "mlStep_incl.dump")

        if not os.path.exists(dump_path):
            continue
        with open(dump_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if "inferenceHost" in line:
                try:
                    vals = [float(x) for x in line.split()[1:] if float(x) > 1e-9]
                    real_cpu_inf[base] = max(vals)
                except Exception:
                    pass
            elif "inferenceDevice" in line:
                try:
                    vals = [float(x) for x in line.split()[1:] if float(x) > 1e-9]
                    real_gpu_inf[base] = max(vals)
                except Exception:
                    pass
                
    if real_cpu_inf:
        xs = sorted([k for k in real_cpu_inf])
        y = [real_cpu_inf[x.replace(".", ",")] for x in xs]
        xs = sorted([float(k.replace(",", ".")) for k in real_cpu_inf])
        plt.plot(xs, y, "o--", label="Real CPU Inference", color="tab:orange")

    if real_gpu_inf:
        xs = sorted([k for k in real_gpu_inf])
        y = [real_gpu_inf[x.replace(".", ",")] for x in xs]
        xs = sorted([float(k.replace(",", ".")) for k in real_gpu_inf])
        plt.plot(xs, y, "o--", label="Real GPU Inference", color="tab:green")

    plt.plot(worksplits, cpu_line, label="Theoretical CPU Runtime")
    plt.plot(worksplits, gpu_line, label="Theoretical GPU Runtime")
    
    #a = 5
    #cpu_bulge_line = cpu_time * np.log1p(a * np.array(worksplits)) / np.log1p(a)
    #print(cpu_bulge_line)
    #plt.plot(worksplits, cpu_bulge_line, label="Bulge")

    # Add theoretical and real optimal vertical lines
    theo_x = None
    real_x = None
    if expected_opt_splits and example_case in expected_opt_splits:
        try:
            theo_x = float(str(expected_opt_splits[example_case]).replace(",", "."))
            plt.axvline(
                theo_x, color="tab:green", linestyle="--", linewidth=1.5,
                #label=f"Theoretical Opt ({theo_x:.2f})"
            )
        except ValueError:
            pass

    if real_opt_splits and example_case in real_opt_splits:
        try:
            real_x = float(str(real_opt_splits[example_case]).replace(",", "."))
            plt.axvline(
                real_x, color="crimson", linestyle=":", linewidth=1.5,
                #label=f"Real Opt ({real_x:.2f})"
            )
        except ValueError:
            pass

    plt.xlabel("CPU Fraction")
    plt.ylabel("Runtime (s)")
    plt.title(f"Theoretical Runtime Model - {example_case}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"summary_runtime_model_{example_case}.pdf")
    plt.close()
    print(f"Saved plotG: summary_runtime_model_{example_case}.pdf")


def plotG_distance_summary(all_case_runtimes, expected_opt_splits=None, real_opt_splits=None):
    """
    Calculate squared distance between real and theoretical CPU & GPU runtimes separately for all worksplits,
    then plot per case as grouped bars:
    - Sum of squared errors for CPU
    - Sum of squared errors for GPU
    - Distance between real and theoretical optimal split
    """
    case_names = sorted(all_case_runtimes.keys())
    
    cpu_errors = []
    gpu_errors = []
    opt_split_distances = []
    r2_cpu = []
    r2_gpu = []

    for case in case_names:
        runtimes = all_case_runtimes[case]
        worksplits = sorted([float(k.replace(",", ".")) for k in runtimes.keys() if "," in k or "." in k])
        if not worksplits:
            cpu_errors.append(0)
            gpu_errors.append(0)
            opt_split_distances.append(0)
            continue

        # Theoretical CPU & GPU runtimes
        cpu_time = runtimes.get("1,00", max(runtimes.values()))
        gpu_time = runtimes.get("0,00", min(runtimes.values()))
        theoretical_cpu = [cpu_time * w for w in worksplits]
        theoretical_gpu = [gpu_time * (1 - w) for w in worksplits]

        # Gather real CPU & GPU runtimes
        real_cpu = []
        real_gpu = []
        exp_dir = os.path.join(case, "Experiments")
        if not os.path.exists(exp_dir):
            cpu_errors.append(0)
            gpu_errors.append(0)
            opt_split_distances.append(0)
            continue

        for w in worksplits:
            w_str = str(w).replace(".", ",")
            dump_path = os.path.join(exp_dir, w_str, "mlStep_incl.dump")
            cpu_val = None
            gpu_val = None
            if os.path.exists(dump_path):
                with open(dump_path, "r") as f:
                    lines = f.readlines()
                cpu_vals = []
                gpu_vals = []
                for line in lines:
                    parts = line.split()[1:]  # skip first token
                    try:
                        numbers = [float(x.replace(",", ".")) for x in parts]
                    except Exception:
                        numbers = []
                    if line.startswith("inferenceHost"):
                        cpu_vals.extend(numbers)
                    elif line.startswith("inferenceDevice"):
                        gpu_vals.extend(numbers)
                if cpu_vals:
                    cpu_val = max(cpu_vals)
                if gpu_vals:
                    gpu_val = max(gpu_vals)

            # Fallback to theoretical if missing
            real_cpu.append(cpu_val if cpu_val is not None else theoretical_cpu[worksplits.index(w)])
            real_gpu.append(gpu_val if gpu_val is not None else theoretical_gpu[worksplits.index(w)])

        # Compute squared errors
        cpu_errors.append(sum((r - t) ** 2 for r, t in zip(real_cpu, theoretical_cpu)))
        gpu_errors.append(sum((r - t) ** 2 for r, t in zip(real_gpu, theoretical_gpu)))

        ssres_gpu = sum((r - t) ** 2 for r, t in zip(real_gpu, theoretical_gpu))
        sstot_gpu = sum([(r - np.mean(theoretical_gpu)) ** 2 for r in real_gpu])
        r2_gpu.append(1- ((ssres_gpu)/(sstot_gpu)))

        ssres_cpu = sum((r - t) ** 2 for r, t in zip(real_cpu, theoretical_cpu))
        sstot_cpu = sum([(r - np.mean(theoretical_cpu)) ** 2 for r in real_cpu])
        r2_cpu.append(1- ((ssres_cpu)/(sstot_cpu)))

        # Optimal split distance
        try:
            theo_x = float(str(expected_opt_splits.get(case, 0)).replace(",", "."))
            real_x = float(str(real_opt_splits.get(case, 0)).replace(",", "."))
            deviation = 1 - abs(real_x - theo_x)  
            opt_split_distances.append(deviation)
        except Exception:
            opt_split_distances.append(0)

    # Plot grouped bar chart
    x = np.arange(len(case_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, r2_cpu, width, label="CPU", color="tab:orange")
    plt.bar(x , r2_gpu, width, label="GPU", color="tab:green")
    plt.ylim(bottom=0.85, top=1.01)
    plt.bar(x + width, opt_split_distances, width, label="Opt Split Closeness", color="tab:blue")
    
    plt.xticks(x, case_names, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("R2 Error of CPU and GPU Inference times per Case")
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("summary_cpu_gpu_error_vs_opt_distance.pdf")
    plt.close()
    print("Saved plot: summary_cpu_gpu_error_vs_opt_distance.pdf")



def plotH(real_opt_splits, order):
    """
    Plot H: Energy consumption per case (kWh)
    - GPU-only (0,00)
    - CPU-only (1,00)
    - Real optimal worksplit
    """
    summary_file = os.path.join("..", "summary.csv")
    slurm_map_file = os.path.join("..", "slurm_ids_with_folders.csv")

    if not os.path.exists(summary_file) or not os.path.exists(slurm_map_file):
        print("summary.csv or slurm_ids_with_folders.csv not found in parent directory!")
        return

    # Load data
    df_energy = pd.read_csv(summary_file)
    df_map = pd.read_csv(slurm_map_file)

    het_jobs = df_map[~df_map["folder"].str.contains("1,00")].copy()
    
    # Duplicate and increment slurmid for the second subjob
    het_jobs["slurmid"] = het_jobs["slurmid"] + 1

    # Append to original map
    df_map_expanded = pd.concat([df_map, het_jobs], ignore_index=True)

    # Merge both to map job_id -> folder path
    df = df_energy.merge(df_map_expanded, left_on="job_id", right_on="slurmid", how="inner")

    # Compute total energy in kWh
    df = df.fillna(0)
    df["total_kwh"] = (df["cpu_energy_j"] + df["gpu_energy_j"]) / 3_600#only wH _000  # J → kWh
    df["cpu_kwh"] = (df["cpu_energy_j"]) / 3_600#only wH _000  # J → kWh
    df["gpu_kwh"] = (df["gpu_energy_j"]) / 3_600#only wH _000  # J → kWh


    energy_gpu_only = {}
    energy_cpu_only = {}
    energy_optimal = {}

    for case in order:
        exp_dir = os.path.join(case, "Experiments")
        if not os.path.exists(exp_dir):
            continue
        
        # Find folders for the three targets
        cpu_folder = os.path.join(exp_dir, "1,00")
        gpu_folder = os.path.join(exp_dir, "0,00")
        opt_split = real_opt_splits.get(case, None)
        opt_folder = os.path.join(exp_dir, opt_split) if opt_split else None

        def find_energy(folder):
            folder = "/"+("/".join((os.path.abspath(folder)).split("/")[4:]))
            if folder is None:
                return np.nan
            
            matching_rows = df.loc[df["folder"].str.endswith(folder, na=False)]

            # Warning for heterogeneous jobs with missing parts
            if "1,00" not in folder and ((matching_rows["cpu_kwh"] == 0) & (matching_rows["gpu_kwh"] == 0)).any():
                print(f"WARNING: {folder}")
                print(matching_rows)

            if "1,00" not in folder and len(matching_rows) < 2:
                return np.nan

            # Merge all matching rows by summing total_kwh
            total_energy = matching_rows["total_kwh"].sum()

            return total_energy

            # match by folder suffix
            if "1,00" not in folder and (df.loc[df["folder"].str.contains(folder, na=False), "cpu_kwh"].iloc[0] == 0 or df.loc[df["folder"].str.contains(folder, na=False), "gpu_kwh"].iloc[0] == 0):
                print(f"WARNING: {folder}")
                print(f"{df.loc[df["folder"].str.contains(folder, na=False)]}")
                
            return df.loc[df["folder"].str.contains(folder, na=False), "total_kwh"].iloc[0]

        energy_cpu_only[case] = find_energy(cpu_folder)
        energy_gpu_only[case] = find_energy(gpu_folder)
        energy_optimal[case] = find_energy(opt_folder)

        print(f"Energy: Case = {case}, CPU = {energy_cpu_only[case]}, GPU = {energy_gpu_only[case]}, opt = {energy_optimal[case]}")
        
    # Prepare plot
    ordered_cases = [c for c in order if c in energy_optimal]
    new_cases = [c for c in order if c in energy_optimal]
    for c in ordered_cases:
        if math.isnan(energy_gpu_only[c]) or math.isnan(energy_cpu_only[c]) or math.isnan(energy_optimal[c]):
            energy_cpu_only.pop(c)
            energy_gpu_only.pop(c)
            energy_optimal.pop(c)
            new_cases.remove(c)
            
    x = np.arange(len(energy_optimal))
    width = 0.25
    # plt.figure(figsize=(8.27, 3))
    plt.figure(figsize=(6, 3))
    plt.bar(x - width, [energy_gpu_only[c] for c in new_cases], width=width, label="GPU-only")
    plt.bar(x, [energy_cpu_only[c] for c in new_cases], width=width, label="CPU-only")
    plt.bar(x + width, [energy_optimal[c] for c in new_cases], width=width, label="Optimal Worksplit")

    plt.xlabel("Case (Nodes - GPUs)")
    plt.ylabel("Total Energy (Wh)")
    plt.title("Energy Consumption per Case and Worksplit")
    x_labels = []
    for name in new_cases:
        # Find where the "Nodes" and "GPU" parts are based on capitalization
        # Assume format is like "4CPUNodes2GPU"
        i_node = name.find("CPU")
        i_gpu = name.find("GPU")
        
        # Extract numbers before "CPU" and before "GPU"
        n_nodes = name[:i_node]
        n_gpus = name[i_node + 8:i_gpu]  # skip "CPUNodes"        
        if n_nodes == "1":
            label = f"{n_nodes} Node \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        else:
            label = f"{n_nodes} Nodes \n{n_gpus} GPU{'s' if n_gpus != '1' else ''}"
        x_labels.append(label)

    plt.xticks(x, x_labels)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("summary_energy_consumption.pdf")
    plt.close()
    print("Saved plotH: summary_energy_consumption.pdf")

    print("\nEnergy savings per case:")
    for c in new_cases:
        total = energy_optimal[c]
        gpu = energy_gpu_only[c]
        cpu = energy_cpu_only[c]

        saved_vs_gpu = gpu - total
        saved_vs_cpu = cpu - total

        pct_vs_gpu = (saved_vs_gpu / gpu * 100) if gpu != 0 else np.nan
        pct_vs_cpu = (saved_vs_cpu / cpu * 100) if cpu != 0 else np.nan

        print(f"Case: {c}")
        print(f"  Energy saved vs GPU-only: {saved_vs_gpu:.2f} Wh ({pct_vs_gpu:.1f}%)")
        print(f"  Energy saved vs CPU-only: {saved_vs_cpu:.2f} Wh ({pct_vs_cpu:.1f}%)")


def main():
    base_dir = "."
    cases = sorted([d for d in os.listdir(base_dir) if os.path.isdir(d) and "nodes" in d.lower()], key=get_case_sort_key)
    worksplits_csv = os.path.join(base_dir, "worksplits.csv")
    if not os.path.exists(worksplits_csv):
        print("No worksplits.csv found at top level!")
        return
    df_expected = pd.read_csv(worksplits_csv)
    expected_times = dict(zip(df_expected["Case"], df_expected["ExpectedTime"]))
    expected_opt = dict(zip(df_expected["Case"], df_expected["OptimalWorksplit"]))
    expected_speedup = dict(zip(df_expected["Case"], df_expected["SpeedupGPU"]))

    real_opt_splits = {}
    real_speeds = {}

    all_case_runtimes = {}
    baseline_case_time = None

    for case in cases:
        exp_dir = os.path.join(case, "Experiments")

        # --- Get real min runtime from experiments
        opt_split, min_time, runtimes = get_min_runtime_info(exp_dir)
        if min_time is None:
            print(f"No valid data found for {case}")
            continue

        try:
            baseline_case_time = runtimes["0,00"]
        except KeyError:
            baseline_case_time = min_time  # fallback if 0,00 not present
            print(f"Warning: '0,00' worksplit not found for {case}, using min_time as baseline.")

        speedup = baseline_case_time / min_time
        real_opt_splits[case] = opt_split
        all_case_runtimes[case] = runtimes
        real_speeds[case] = speedup

    order = sorted(real_speeds.keys(), key=get_case_sort_key)

    # plotA(real_speeds, order)
    # plotB(real_speeds, expected_speedup, expected_opt, real_opt_splits, order)
    plotC(real_speeds, expected_speedup, order)
    # plotD(all_case_runtimes)
    plotE(real_speeds, expected_speedup, expected_opt, real_opt_splits, order)

    # plotF(expected_speedup, expected_opt, real_opt_splits, all_case_runtimes, order)#bars: Theoretical Speedup, Speeup at theoretical optimum, Speedup at actual optimum
    # plotG("8CPUNodes4GPU", all_case_runtimes, expected_opt, real_opt_splits)#Two linear lines of theoretical assumed runtime, one for GPU , one for CPU. basis behind calculation
    # plotG_distance_summary(all_case_runtimes, expected_opt, real_opt_splits)
    plotH(real_opt_splits, order)
    
    plotD_2(all_case_runtimes)
if __name__ == "__main__":
    main()
