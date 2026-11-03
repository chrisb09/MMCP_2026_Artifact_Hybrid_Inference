import numpy as np
import argparse
import os, subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
import h5py
import pandas as pd

def extract_from_mlog(dumpFile, keyword):
    """
    Extracts the keyword time value from lines in the dump file that match a specific format.
    Returns a list of extracted times in seconds.
    If no matching lines are found, returns an empty list.
    """
    times = []
    
    try:
        with open(dumpFile, "r") as f:
            for line in f:
                # Check if the line contains the expected format
                if keyword in line and '[sec]' in line:
                    # Split the line into parts
                    parts = line.split()
                    
                    # Find and extract the time value just before '[sec]'
                    for part in parts:
                        if '[sec]' in part:
                            # The previous part should be the time value
                            try:
                                time_value = float(parts[parts.index(part) - 1])
                                times.append(time_value)
                            except (ValueError, IndexError):
                                print(f"Could not convert to float: {part}")
                            break  # No need to check further once we find our time
    
    except Exception as e:
        print(f"Error reading {dumpFile}: {e}")
    if times:
        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0.0
    else:
        mean_time = 0.0
        std_time = 0.0
    return mean_time, std_time


def runCubeDump(path_to_cube_dump, callpath, metric, info_mode, path_to_output, path_to_cube_profile):
    """
    Run cube_dump with the given arguments.
    """
    command = f"{path_to_cube_dump} -c {callpath} -m {metric} -z {info_mode} -o {path_to_output} {path_to_cube_profile}"
    #print(f"Executing: {command}")
    subprocess.call(command, shell=True)

def getAllTimeForFunction(dumpFile, function_name):
    """
    Searches through the dump file for a line starting with function_name.
    Extracts the timing values (skipping those below 1e-9) and computes both
    the mean and the standard deviation.
    Returns (mean_time, std_time); if not found, returns (0.0, 0.0).
    """
    times = []
    try:
        with open(dumpFile, "r") as f:
            lines = f.readlines()[7:]  # skip header lines
            for line in lines:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0].startswith(function_name):
                    try:
                        vals = [float(x) for x in tokens[1:]]
                        filtered = [v for v in vals if v > 1e-9]
                        times.extend(filtered)
                    except Exception as e:
                        print(f"Error processing timings in line: {line}\n{e}")
                    #break
    except Exception as e:
        print(f"Error reading {dumpFile}: {e}")
    return times

def getAverageTimeForFunction(dumpFile, function_name):
    """
    Searches through the dump file for a line starting with function_name.
    Extracts the timing values (skipping those below 1e-9) and computes both
    the mean and the standard deviation.
    Returns (mean_time, std_time); if not found, returns (0.0, 0.0).
    """
    times = []
    try:
        with open(dumpFile, "r") as f:
            lines = f.readlines()[7:]  # skip header lines
            for line in lines:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0].startswith(function_name):
                    try:
                        vals = [float(x) for x in tokens[1:]]
                        filtered = [v for v in vals if v > 1e-9]
                        times.extend(filtered)
                    except Exception as e:
                        print(f"Error processing timings in line: {line}\n{e}")
                    #break
    except Exception as e:
        print(f"Error reading {dumpFile}: {e}")
    if times:
        mean_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        std_time = np.std(times) if len(times) > 1 else 0.0
    else:
        mean_time = 0.0
        std_time = 0.0
        max_time = 0.0
        min_time = 0.0
    return mean_time, std_time , max_time, min_time

def getCPUGPUTime(dumpFile, function_name):
    """
    Searches through the dump file for a line starting with function_name.
    Extracts the timing values (skipping those below 1e-9) and computes both
    the mean and the standard deviation.
    Returns (mean_time, std_time); if not found, returns (0.0, 0.0).
    """
    times = []
    try:
        with open(dumpFile, "r") as f:
            lines = f.readlines()[7:]  # skip header lines
            for line in lines:
                tokens = line.split()
                if not tokens:
                    continue
                if tokens[0].startswith(function_name):
                    try:
                        vals = [float(x) for x in tokens[1:]]
                        filtered = [v for v in vals if v > 1e-9]
                        times.extend(filtered)
                        #print(tokens)
                    except Exception as e:
                        print(f"Error processing timings in line: {line}\n{e}")
                    #break
    except Exception as e:
        print(f"Error reading {dumpFile}: {e}")
    if times:
        mean_time_gpu = np.mean(times[0:3])
        max_time_gpu = np.max(times[0:3])
        mean_time_cpu = np.mean(times[4:]) if len(times[4:]) > 0 else math.nan
        max_time_cpu = np.max(times[4:]) if len(times[4:]) > 0 else math.nan
    else:
        mean_time_gpu = 0.0
        mean_time_cpu = 0.0
        max_time_gpu = 0.0
        max_time_cpu = 0.0
    return mean_time_gpu, mean_time_cpu, max_time_gpu, max_time_cpu


def findMLStepCallpath(full_profile_dump):
    """
    Scan the dump file (generated with callpath 'all') to locate
    the first occurrence of ml_step and then postprocess_output.
    Return a callpath string as "startID-endID".
    """
    callPathStart = -1
    callPathEnd = -1

    try:
        with open(full_profile_dump, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Cannot open file {full_profile_dump}: {e}")
        return ""
    # Skip first few header lines.
    for idx, line in enumerate(lines):
        if idx < 7:  # assume header lines > 7
            continue
        tokens = line.split()
        if not tokens:
            continue
        # Expect token of the form "FunctionName(id=XXX)"
        parts = tokens[0].split("(id=")
        func_name = parts[0]
        if callPathStart < 0 and func_name.startswith("MLCouplingMaia::ml_step"):
            try:
                callPathStart = int(parts[1].rstrip(")"))
            except Exception as e:
                print(f"Error parsing ml_step callpath from line: {line}\n{e}")
        if callPathStart > 0 and "::postprocess_output" in func_name:
            try:
                callPathEnd = int(parts[1].rstrip(")"))
                break
            except Exception as e:
                print(f"Error parsing postprocess_output callpath from line: {line}\n{e}")
    if callPathStart < 0 or callPathEnd < 0:
        print("Warning: Unable to determine ml_step callpath from full dump.")
        return ""
    callpath = f"{callPathStart}-{callPathEnd}"
    return callpath


def cubes_needed(length, c, o):
    """
    Return the number of cubes needed to cover an interval of a given length
    with cubes of side length c when allowing a deliberate overlap o.
    """
    assert 0 <= o < c, "overlap must be smaller than cube size"
    return math.ceil( max(0, length-c) / (c - o))+1 #+1 because code adds cubes starting from 0. So 


def calc_cubes(rank_array):
    #ranks = rank_blocks_in_memory("/hpcwork/rwth1859/OverlapInvestigation/S/1/exp_0/out/3010.hdf5")
    cube_size = 8
         # read into memory (dtype typically int)

    unique_ranks = np.unique(rank_array)
    results = []
    for rank in unique_ranks:
        # mask == True where the current rank is present
        mask = (rank_array == rank)

        # axis-aligned bounding box
        z_idx, y_idx, x_idx = np.where(mask)
        z0, z1 = z_idx.min(), z_idx.max() + 1   # +1 because python slices are half-open
        y0, y1 = y_idx.min(), y_idx.max() + 1
        x0, x1 = x_idx.min(), x_idx.max() + 1

        Δx, Δy, Δz = x1 - x0, y1 - y0, z1 - z0

        concatX = math.ceil((Δx-cube_size) / cube_size)#+1
        concatY = math.ceil((Δy-cube_size) / cube_size)#+1
        concatZ = math.ceil((Δz-cube_size) / cube_size)#+1
        nx = len(np.linspace(0, Δx-cube_size, concatX))+1
        ny = len(np.linspace(0, Δy-cube_size, concatY))+1
        nz = len(np.linspace(0, Δz-cube_size, concatZ))+1


        total_cubes = nx * ny * nz

        results.append([
            int(rank),
            (Δx, Δy, Δz),
            (nx, ny, nz),
            int(total_cubes)
        ])

    # nicely formatted table
    df = pd.DataFrame(
            results,
            columns=["Rank",
                    "BBox dims (dx,dy,dz)",
                    "Cubes per axis (nx,ny,nz)",
                    "Total cubes"])
    #print(df)
    return df["Total cubes"].sum()

def plot_parabel(data, outdir, name, ylabel = "Runtime (s)", log = True):
    plt.figure(figsize=(6,4))
    plt.plot(data.keys(), data.values(), marker='o')
    plt.xlabel(r"Worksplit")
    plt.ylabel(ylabel)
    plt.title(rf"Hybrid Inference Worksplit to ML Runtime")
    plt.xticks(ticks = range(len(data.keys())), labels = data.keys(), rotation=45, ha="right")
    
    if log:
        print("Log")
        orig_yticks = plt.yticks()[0]
        plt.yscale("log")
        plt.yticks(orig_yticks)
        ax = plt.gca() 
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.ylim(bottom=0)
    plt.minorticks_on()

    plt.grid(which='major', axis='both', linestyle='--', alpha=0.6)
    plt.grid(which='minor', axis='both', linestyle=':',  alpha=0.6)

    fname = os.path.join(outdir, f"hybrid_parabel_{name}.pdf")
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"Saved parabel plot to {fname}")


def plot_bar(data, outdir, name, ylabel = "Runtime (s)", log = True):
    
    means = [np.mean(values) for values in data.values()]
    stds = [np.std(values) for values in data.values()]
    plt.figure(figsize=(6,4))
    x_pos = np.arange(len(data))

    # Create bar plot with error bars
    plt.bar(x_pos, means, yerr=stds, capsize=5, color='skyblue', alpha=0.8)
    plt.xticks(x_pos, data.keys())
    plt.xlabel(r"Worksplit")
    plt.ylabel(ylabel)
    plt.title(rf"Hybrid Inference Worksplit to ML Runtime")
    plt.xticks(ticks = range(len(data.keys())), labels = data.keys(), rotation=45, ha="right")
    
    if log:
        print("Log")
        orig_yticks = plt.yticks()[0]
        plt.yscale("log")
        plt.yticks(orig_yticks)
        ax = plt.gca() 
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        plt.ylim(bottom=0)
    plt.minorticks_on()

    plt.grid(which='major', axis='both', linestyle='--', alpha=0.6)
    plt.grid(which='minor', axis='both', linestyle=':',  alpha=0.6)

    fname = os.path.join(outdir, f"hybrid_parabel_{name}.pdf")
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"Saved parabel plot to {fname}")


def plot_parabel2(datas, outdir, name, names, y0 = True):
    plt.figure(figsize=(6,4))
    for data in datas:
        plt.plot(data.keys(), data.values(), marker="o", markersize=3)
    plt.xlabel(r"Worksplit")
    plt.ylabel(r"Runtime (s)")
    plt.title(rf"Hybrid Inference Worksplit to ML Runtime")
    plt.xticks(ticks = range(len(datas[0].keys())), labels = datas[0].keys(), rotation=45, ha="right")
    
    orig_yticks = plt.yticks()[0]
    #plt.yscale("log", base=2)
    plt.yticks(orig_yticks)
    ax = plt.gca() 
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    if y0 == True:
        plt.ylim(bottom=0)
    #plt.minorticks_on()

    plt.grid(which='major', axis='both', linestyle='--', alpha=0.6)
    #plt.grid(which='minor', axis='both', linestyle=':',  alpha=0.6)
    plt.legend(names, loc="best")
    fname = os.path.join(outdir, f"hybrid_parabel_{name}.pdf")
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"Saved parabel plot to {fname}")


def plot_parabel2_scaled(datas, outdir, name, names, y0 = True):
    plt.figure(figsize=(6,4))

    # Convert string keys "number,numbers" to floats for proper spacing
    x_vals = [float(k.replace(",", ".")) for k in datas[0].keys()]

    # Plot each dataset at numerical x positions
    for data in datas:
        y_vals = list(data.values())
        plt.plot(x_vals, y_vals, marker="o", markersize=3)

    plt.xlabel(r"CPU Fraction")
    plt.ylabel(r"Runtime (s)")
    plt.title(r"Hybrid Inference Worksplit vs ML Runtime")

    from matplotlib.ticker import MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    #plt.xticks(rotation=45, ha="right")

    # Y-axis formatting
    orig_yticks = plt.yticks()[0]
    plt.yticks(orig_yticks)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)


    if y0:
        plt.ylim(bottom=0)

    plt.grid(which='major', axis='both', linestyle='--', alpha=0.6)
    plt.legend(names, loc="best")
    fname = os.path.join(outdir, f"hybrid_parabel_{name}_scaled.pdf")
    plt.tight_layout(pad=0.5)
    plt.savefig(fname, format='pdf')
    plt.close()
    print(f"Saved parabel plot to {fname}")




def main():
    parser = argparse.ArgumentParser(
        description="Traverse benchmark folder tree and extract performance metrics per experiment."
    )
    parser.add_argument("--benchmarkdir", type=str, default="./Experiments",
                        help="Path to the top-level benchmark directory.")
    parser.add_argument("--output", type=str, default="plots", help="Directory to save plots")
    
    parser.add_argument("--cubedump", type=str, default="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CubeLib/4.9-GCCcore-13.3.0/bin/cube_dump",
                        help='Path to cube_dump executable (e.g. --cubedump "$(which cube_dump)")')
    #/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CubeLib/4.9-GCCcore-13.3.0/bin/cube_dump
    args = parser.parse_args()    

    pathToCubeDump = args.cubedump
    os.makedirs(args.output, exist_ok=True)

    benchmark_dir = os.path.abspath(args.benchmarkdir)

    data = {}
    data_inference = {}
    data_overhead = {}
    data_pre = {}
    data_post = {}

    data_cpu_inf = {}
    data_gpu_inf = {}
    data_overhead_rem = {}

    
    data_max = {}
    data_inference_max = {}
    data_overhead_max = {}
    data_pre_max = {}
    data_post_max = {}

    data_cpu_inf_max = {}
    data_gpu_inf_max = {}
    data_overhead_rem_max = {}
    
    data_list = {}
    """for folder in os.listdir(benchmark_dir):
        folder_path = os.path.join(benchmark_dir, folder)
        if not os.path.isdir(folder_path) or not folder.startswith("0,"):
            continue
        mlog_file = os.path.join(folder_path, "m_log")
        ml_step_mean, ml_step_std = extract_from_mlog(mlog_file, "MLCoupling")
        data[folder] = ml_step_mean
    """
    #with h5py.File(f"/hpcwork/rwth1859/HybridInference/Experiments/0,07/out/3010.hdf5", "r") as f:
    #    rank_array = f["block0"]["blockId"][()] 


    for root, dirs, files in os.walk(benchmark_dir):
        base = os.path.basename(root)
        print(f"---- {base} {root}")
        scorep_dirs = [d for d in dirs if d.startswith("scorep")]
        if not scorep_dirs:
            #print(f"Skipping {root}: no Score-P folder found.")
            continue
        scorep_folder = scorep_dirs[0]
        profile_path = os.path.join(root, scorep_folder, "profile.cubex")
        if not os.path.exists(profile_path):
            continue
        full_dump_file = os.path.join(root, "full.dump")
        runCubeDump(pathToCubeDump, "all", "time", "excl", full_dump_file, profile_path)
        # Find the ml_step callpath.
        ml_step_callpath = findMLStepCallpath(full_dump_file) + ",72-73"
        if ml_step_callpath == "":
            #print(f"Skipping {root}: ml_step callpath not determined.")
            continue
        # Create a dump for the ml_step region specifically.
        step_dump_file = os.path.join(root, "mlStep_incl.dump")
        runCubeDump(pathToCubeDump, ml_step_callpath, "time", "incl", step_dump_file, profile_path)
        # Extract raw timing metrics (mean and std) for key functions.
        ml_step_mean, ml_step_std, ml_step_max, _ = getAverageTimeForFunction(step_dump_file, "MLCouplingMaia::ml_step")
        preproc_mean, preproc_std, preproc_max, _ = getAverageTimeForFunction(step_dump_file, f"MLCouplingMaiaAix::preprocess_input")
        inference_mean, inference_std, inference_max, _ = getAverageTimeForFunction(step_dump_file, "torchInference::inference")
        gather_mean, gather_std, _, gather_min = getAverageTimeForFunction(step_dump_file, "gatherInputData")
        scatter_mean, scatter_std, _, scatter_min = getAverageTimeForFunction(step_dump_file, "scatterOutputData")
        #inference_mean, _ = getAverageTimeForFunction(step_dump_file, "")
        print(scatter_min)
        print(gather_min)
        inference_cpu, _, inference_cpu_max, _= getAverageTimeForFunction(step_dump_file, "inferenceHost")
        inference_gpu, _, inference_gpu_max, _ = getAverageTimeForFunction(step_dump_file, "inferenceDevice")
        print(inference_cpu_max)
        print(inference_gpu_max)
        print(inference_max)
        postproc_mean, postproc_std, postproc_max, _ = getAverageTimeForFunction(step_dump_file, f"MLCouplingMaiaAix::postprocess_output")
        # Compute computed metric: overhead = ml_step - (preproc + inference + postproc)
        overhead_mean = ml_step_mean - (preproc_mean + np.max([inference_gpu, (inference_cpu if not math.isnan(inference_cpu) else 0)]) + postproc_mean)
        overhead_max = ml_step_max - (preproc_max + np.max([inference_gpu_max, (inference_cpu_max if not math.isnan(inference_cpu_max) else 0)]) + postproc_max)
        overhead_std  = np.sqrt(ml_step_std**2 + preproc_std**2 + inference_std**2 + postproc_std**2)

        inference_list = []
        if base != "1,00" and base != "0,00":
            # Both lists exist, sum element-wise
            device_times = getAllTimeForFunction(step_dump_file, "inferenceDevice")
            host_times = getAllTimeForFunction(step_dump_file, "inferenceHost")
            inference_list = [(d + h ) for d, h in zip(device_times, host_times)]
        elif base != "1,00":
            # Only device, filter > 0.1
            device_times = getAllTimeForFunction(step_dump_file, "inferenceDevice")
            inference_list = [t for t in device_times if t > 0.1]
        elif base != "0,00":
            # Only host, filter > 0.1
            host_times = getAllTimeForFunction(step_dump_file, "inferenceHost")
            inference_list = [t for t in host_times if t > 0.1]
        if base not in data.keys():
            data_cpu_inf[base] = [] 
            data_gpu_inf[base] = [] 
            data[base] = [] 
            data_inference[base] = [] 
            data_pre[base] = [] 
            data_post[base] = [] 
            data_overhead[base] = [] 
            data_cpu_inf[base] = [] 
            data_gpu_inf[base] = [] 
            
            data_cpu_inf_max[base] = [] 
            data_gpu_inf_max[base] = [] 
            data_max[base] = [] 
            data_inference_max[base] = [] 
            data_pre_max[base] = [] 
            data_post_max[base] = [] 
            data_overhead_max[base] = [] 

            data_list[base] = []

        data[base].append(ml_step_mean)
        data_inference[base].append(inference_mean)
        data_pre[base].append(preproc_mean)
        data_post[base].append(postproc_mean)
        data_overhead[base].append(overhead_mean)

        data_cpu_inf[base].append(inference_cpu)
        data_gpu_inf[base].append(inference_gpu)

        data_max[base].append(ml_step_max)
        data_inference_max[base].append(inference_max)
        data_pre_max[base].append(preproc_max)
        data_post_max[base].append(postproc_max)
        data_overhead_max[base].append(overhead_max)

        data_cpu_inf_max[base].append(inference_cpu_max)
        data_gpu_inf_max[base].append(inference_gpu_max)
        data_list[base].append(inference_list)
        #data_overhead_rem[base] = overhead_mean - 

        #cubes_total = calc_cubes(rank_array)
        #cubes_cpu = round(cubes_total * 3 * float(base.replace(",", ".")))
        #print(f"Cubes CPU: {cubes_cpu}")
        #print(f"Cubes GPU: {cubes_total - cubes_cpu}")

    data = {k: sum(v) / len(v) for k, v in data.items()}
    data_inference = {k: sum(v) / len(v) for k, v in data_inference.items()}
    data_pre = {k: sum(v) / len(v) for k, v in data_pre.items()}
    data_post = {k: sum(v) / len(v) for k, v in data_post.items()}
    data_overhead = {k: sum(v) / len(v) for k, v in data_overhead.items()}
    data_cpu_inf = {k: sum(v) / len(v) for k, v in data_cpu_inf.items()}
    data_gpu_inf = {k: sum(v) / len(v) for k, v in data_gpu_inf.items()}
    
    data_list = {k: [sum(x)/len(x) for x in zip(*v)] for k, v in data_list.items()}
    
    data_max = {k: sum(v) / len(v) for k, v in data_max.items()}
    data_inference_max = {k: sum(v) / len(v) for k, v in data_inference_max.items()}
    data_pre_max = {k: sum(v) / len(v) for k, v in data_pre_max.items()}
    data_post_max = {k: sum(v) / len(v) for k, v in data_post_max.items()}
    data_overhead_max = {k: sum(v) / len(v) for k, v in data_overhead_max.items()}
    data_cpu_inf_max = {k: sum(v) / len(v) for k, v in data_cpu_inf_max.items()}
    data_gpu_inf_max = {k: sum(v) / len(v) for k, v in data_gpu_inf_max.items()}

    sorted_data = dict(
        sorted(
            data.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_inference = dict(
        sorted(
            data_inference.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_pre = dict(
        sorted(
            data_pre.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_post = dict(
        sorted(
            data_post.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_overhead = dict(
        sorted(
            data_overhead.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_cpu = dict(
        sorted(
            data_cpu_inf.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_gpu = dict(
        sorted(
            data_gpu_inf.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_ovrem = dict(
        sorted(
            data_overhead_rem.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_list = dict(
        sorted(
            data_list.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )

    sorted_data_max = dict(
        sorted(
            data_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_inference_max = dict(
        sorted(
            data_inference_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_pre_max = dict(
        sorted(
            data_pre_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_post_max = dict(
        sorted(
            data_post_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_overhead_max = dict(
        sorted(
            data_overhead_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_cpu_max = dict(
        sorted(
            data_cpu_inf_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_gpu_max = dict(
        sorted(
            data_gpu_inf_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    sorted_data_ovrem_max = dict(
        sorted(
            data_overhead_rem_max.items(),
            key=lambda kv: float(kv[0].replace(",", "."))  
        )
    )
    plot_parabel(sorted_data_max, args.output, "ml_step")
    plot_parabel(sorted_data_inference_max, args.output, "inference")
    plot_parabel(sorted_data_post_max, args.output, "post")
    plot_parabel(sorted_data_pre_max, args.output, "pre")
    plot_parabel(sorted_data_overhead_max, args.output, "overhead")
    
    #plot_parabel2([sorted_data_cpu, sorted_data_gpu, sorted_data_overhead, sorted_data], args.output, "both", ["CPU Inference", "GPU Inference", "Overhead", "ML_Step"])
    plot_parabel2([sorted_data_cpu_max, sorted_data_gpu_max, sorted_data_overhead_max, sorted_data_max], args.output, "both", ["CPU Inference", "GPU Inference", "Overhead", "ML_Step"])

    plot_parabel2_scaled([sorted_data_cpu_max, sorted_data_gpu_max, sorted_data_overhead_max, sorted_data_max], args.output, "both", ["CPU Inference", "GPU Inference", "Overhead", "ML_Step"])


    plot_parabel2([sorted_data_cpu_max, sorted_data_gpu_max], args.output, "cpu_gpu", ["CPU Inference", "GPU Inference"])

    normalized_data_max = {
        k: sorted_data_max["0,00"] / v for k, v in sorted_data_max.items()
    }
    #normalized_data_max.pop("0,00", None)
    plot_parabel(normalized_data_max, args.output, "speedup", ylabel= "Speedup", log=False)
    
    normalized_data = {
        k: sorted_data["0,00"] / v for k, v in sorted_data.items()
    }
    #plot_parabel(normalized_data, args.output, "speedup", ylabel= "Speedup", log=False)
    plot_parabel2([filter_close_keys(sorted_data_cpu_max), filter_close_keys(sorted_data_gpu_max), filter_close_keys(sorted_data_overhead_max), filter_close_keys(sorted_data_max)], args.output, "both_filtered", ["CPU Inference", "GPU Inference", "Overhead", "ML_Step"])
    plot_parabel(filter_close_keys(sorted_data_max), args.output, "ml_step_filtered", "ML_Step")

    plot_bar(sorted_data_list, args.output, "inference_bar", "Inference", log=False)


def filter_close_keys(d, min_distance=0.09):
    def to_float(k):
        # convert '0,00' → 0.00 safely
        try:
            return float(str(k).replace(',', '.'))
        except ValueError:
            return None  # skip if conversion fails
   
    keys = sorted(d.keys())
    result = {}

    for i, key in enumerate(keys):
        # Get neighbors
        prev_key = keys[i - 1] if i > 0 else None
        next_key = keys[i + 1] if i < len(keys) - 1 else None

        # Check distances
        too_close_to_prev = prev_key is not None and (to_float(key) - to_float(prev_key)) < min_distance
        too_close_to_next = next_key is not None and (to_float(next_key) - to_float(key)) < min_distance

        # Keep key only if it is not too close to neighbors
        if (too_close_to_prev or too_close_to_next):
            result[key] = d[key]

    return result
    
if __name__ == "__main__":
    main()