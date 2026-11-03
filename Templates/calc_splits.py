import os
import csv
import numpy as np
import math
import subprocess
import argparse

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
    return mean_time, std_time , max_time, min_time

def find_experiment_dir(case_dir, preferred_dirs):
    """
    Return the first existing directory from the list of preferred_dirs inside case_dir/Experiments.
    """
    for d in preferred_dirs:
        path = os.path.join(case_dir, "Experiments", d)
        print(path)
        if os.path.isdir(path):
            return path
    return None

def runCubeDump(path_to_cube_dump, callpath, metric, info_mode, path_to_output, path_to_cube_profile):
    """
    Run cube_dump with the given arguments.
    """
    command = f"{path_to_cube_dump} -c {callpath} -m {metric} -z {info_mode} -o {path_to_output} {path_to_cube_profile}"
    #print(f"Executing: {command}")
    subprocess.call(command, shell=True)


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

def extract_case_times(case_dir, pathToCubeDump):
    """
    Extracts the relevant max times for GPU and CPU inference from subdirectories.
    """
    exp_0_dir = find_experiment_dir(case_dir, ["0,00", "Round1", "Round2"])
    # CPU directory
    exp_1_dir = find_experiment_dir(case_dir, ["1,00", "Round1", "Round2"])
    
    max_gpu_inference = 0.0
    max_gpu_overhead = 0.0
    max_cpu_inference = 0.0

    # Process 0,00: max GPU inference and overhead
    if exp_0_dir is not None:
        for root, dirs, files in os.walk(exp_0_dir):
            createDump(root, dirs, pathToCubeDump)

            for f in files:
                if f.endswith(".dump"):
                    dump_file = os.path.join(root, f)                   
                    _, _, gpu_max, _ = getAverageTimeForFunction(dump_file, "inferenceDevice")
                    _, _, ml_step_max, _ = getAverageTimeForFunction(dump_file, "MLCouplingMaia::ml_step")
                    _, _, pre_max, _ = getAverageTimeForFunction(dump_file, "MLCouplingMaiaAix::preprocess_input")
                    #_, _, inf_max = getAverageTimeForFunction(dump_file, "torchInference::inference")
                    _, _, post_max, _ = getAverageTimeForFunction(dump_file, "MLCouplingMaiaAix::postprocess_output")
                    
                    overhead_max = ml_step_max - (pre_max + gpu_max + post_max)

                    if gpu_max > max_gpu_inference:
                        max_gpu_inference = ml_step_max
                    if overhead_max > max_gpu_overhead:
                        max_gpu_overhead = overhead_max

    # Process 1,00: max CPU inference
    if exp_1_dir is not None:
        for root, dirs, files in os.walk(exp_1_dir):
            createDump(root, dirs, pathToCubeDump)
            for f in files:
                if f.endswith(".dump"):
                    dump_file = os.path.join(root, f)
                    _, _, cpu_max, _ = getAverageTimeForFunction(dump_file, "MLCouplingMaia::ml_step")
                    if not math.isnan(cpu_max) and cpu_max > max_cpu_inference:
                        max_cpu_inference = cpu_max

    return max_gpu_inference, 0, max_cpu_inference

def createDump(root, dirs, pathToCubeDump):
    base = os.path.basename(root)
    print(f"---- {base} {root}")
    scorep_dirs = [d for d in dirs if d.startswith("scorep")]
    if not scorep_dirs:
        #print(f"Skipping {root}: no Score-P folder found.")
        return #continue
    scorep_folder = scorep_dirs[0]
    profile_path = os.path.join(root, scorep_folder, "profile.cubex")
    if not os.path.exists(profile_path):
        return
    full_dump_file = os.path.join(root, "full.dump")
    runCubeDump(pathToCubeDump, "all", "time", "excl", full_dump_file, profile_path)
    # Find the ml_step callpath.
    ml_step_callpath = findMLStepCallpath(full_dump_file) + ",72-73"
    if ml_step_callpath == "":
        return
    # Create a dump for the ml_step region specifically.
    step_dump_file = os.path.join(root, "mlStep_incl.dump")
    runCubeDump(pathToCubeDump, ml_step_callpath, "time", "incl", step_dump_file, profile_path)

def compute_metrics(max_gpu_inf, max_gpu_ovr, max_cpu_inf):
    total = max_gpu_inf + max_gpu_ovr + max_cpu_inf
    print(total)
    print(max_gpu_inf)
    print(max_gpu_ovr)
    print(max_cpu_inf)
    if total == 0:
        optimal_split = 0.0
        expected_time = 0.0
        speedup_gpu = 0.0
        speedup_cpu = 0.0
    else:
        optimal_split = (max_gpu_inf + max_gpu_ovr) / total
        print(optimal_split)
        expected_time = max_cpu_inf * optimal_split
        speedup_gpu = (max_gpu_inf + max_gpu_ovr) / expected_time if expected_time != 0 else 0.0
        speedup_cpu = max_cpu_inf / expected_time if expected_time != 0 else 0.0
    return total, optimal_split, expected_time, speedup_cpu, speedup_gpu

def main(root_dir, output_csv="worksplits.csv"):  
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubedump", type=str, default="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CubeLib/4.9-GCCcore-13.3.0/bin/cube_dump",
                        help='Path to cube_dump executable (e.g. --cubedump "$(which cube_dump)")')
    #/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CubeLib/4.9-GCCcore-13.3.0/bin/cube_dump
    args = parser.parse_args()    

    pathToCubeDump = args.cubedump
    case_results = []

    for case_name in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_name)
        if not os.path.isdir(case_path):
            continue
        
        import re
        gpu_match = re.search(r"(\d+)\s*gpu", case_name.lower())
        node_match = re.search(r"(\d+)\s*cpu", case_name.lower())
        
        gpu_count = int(gpu_match.group(1)) if gpu_match else 0
        node_count = int(node_match.group(1)) if node_match else 1  # default to 1 if not found

        max_gpu_inf, max_gpu_ovr, max_cpu_inf = extract_case_times(case_path, pathToCubeDump)
        
        # Adjust max_cpu_inf according to formula
        if node_count * 96 - gpu_count > 0:
            max_cpu_inf = max_cpu_inf * (node_count * 96) / (node_count * 96 - gpu_count)
        else:
            # Avoid division by zero
            max_cpu_inf = max_cpu_inf

        total, optimal_split, expected_time, speedup_cpu, speedup_gpu = compute_metrics(
            max_gpu_inf, max_gpu_ovr, max_cpu_inf
        )

        case_results.append([
            case_name, max_gpu_inf, max_gpu_ovr, max_cpu_inf, total,
            optimal_split, speedup_cpu, speedup_gpu, expected_time
        ])

    # Write CSV
    headers = [
        "Case", "MaxGPUInference", "MaxGPUOverhead", "MaxCPUInference",
        "Total", "OptimalWorksplit", "SpeedupCPU", "SpeedupGPU", "ExpectedTime"
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(case_results)

    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    root_directory = "./"  # point this to your directory containing cases
    main(root_directory)
