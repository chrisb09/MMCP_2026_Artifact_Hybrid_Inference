import os
import math
import subprocess
import argparse
import numpy as np

def get_max_time(dumpFile, function_name):
    """
    Reads a dump file and returns the max exclusive time for a function.
    If not found, return 0.0.
    """
    times = []
    try:
        with open(dumpFile, "r") as f:
            lines = f.readlines()[7:]  # skip Cube header
            for line in lines:
                tokens = line.split()
                if not tokens: 
                    continue
                if tokens[0].startswith(function_name):
                    try:
                        vals = [float(x) for x in tokens[1:]]
                        filtered = [v for v in vals if v > 1e-12]
                        times.extend(filtered)
                    except:
                        pass
    except:
        return 0.0
    return np.max(times) if times else 0.0


def run_cube_dump(path_to_cube_dump, callpath, metric, info_mode, out_file, profile):
    cmd = f"{path_to_cube_dump} -c {callpath} -m {metric} -z {info_mode} -o {out_file} {profile}"
    subprocess.call(cmd, shell=True)


def find_callpath(full_dump):
    """
    Finds the first ml_step and next postprocess_output in a full dump.
    """
    try:
        with open(full_dump, "r") as f:
            lines = f.readlines()
    except:
        return ""

    start = -1
    end = -1

    for idx, line in enumerate(lines):
        if idx < 7:
            continue
        tokens = line.split()
        if not tokens:
            continue
        parts = tokens[0].split("(id=")
        func_name = parts[0]

        if start < 0 and func_name.startswith("MLCouplingMaia::ml_step"):
            try:
                start = int(parts[1].rstrip(")"))
            except:
                pass

        if start > 0 and "::postprocess_output" in func_name:
            try:
                end = int(parts[1].rstrip(")"))
                break
            except:
                pass

    if start < 0 or end < 0:
        return ""
    return f"{start}-{end}"


def process_subfolder(root, pathToCubeDump):
    """
    Processes one measurement subfolder and computes optimal worksplit from that single point.
    """
    scorep_folders = [d for d in os.listdir(root) if d.startswith("scorep")]
    if not scorep_folders:
        return None, None

    scorep = os.path.join(root, scorep_folders[0])
    profile = os.path.join(scorep, "profile.cubex")
    if not os.path.exists(profile):
        return None, None

    full_dump = os.path.join(root, "full.dump")
    run_cube_dump(pathToCubeDump, "all", "time", "excl", full_dump, profile)

    cp = find_callpath(full_dump)
    if cp == "":
        return None, None

    ml_dump = os.path.join(root, "ml.dump")
    run_cube_dump(pathToCubeDump, cp, "time", "incl", ml_dump, profile)

    # Extract busy times
    R_G = get_max_time(ml_dump, "inferenceDevice")
    R_C = get_max_time(ml_dump, "inferenceHost")
    print(R_C)
    print(R_G)
    return R_G, R_C


def compute_f_opt(f_test, R_G, R_C, O=0.0):
    """
    Computes optimal worksplit using only a single measurement.
    f_test must be known from directory naming (passed externally).
    """
    if f_test <= 0 or f_test >= 1:
        return None  # cannot infer if 0 or 1 oder muss die 96/92 dinge beachten!

    # infer per-unit speeds
    T_g = (R_G - O) / f_test
    T_c = R_C / (1 - f_test)

    if T_g <= 0 or T_c <= 0:
        return None

    # formula: f_opt = (T_c - O) / (T_g + T_c)
    #f_opt = (T_c - O) / (T_g + T_c)
    #f_opt = ((1-f_test) * R_C)/(R_G*f_test+R_C*(1-f_test))

    # Fit linear functions
    # We know for GPU its y=0 at x=1. So with one test point we can deduce the rest:
    # f(x) = mx+b. f(1) = m*1+b = 0 <=> 
    # b = -m => f(x) = mx-m = m(x-1)
    # f(xtest) = m(xtest-1) = ytest <=> #And want to find m
    # m = ytest/(xtest-1) => f(x) = (ytest/(xtest-1)) * (x-1) <=>
    # f(x) = ytest * (x-1)/(xtest-1)
    # Similarly, for CPU it would result in f(x) = ytest * (x-0)/(xtest-0) = ytest * x/xtest
    # For min we have to set them equal
    # ytest0 * (x-1)/(xtest0-1) = ytest1 * x/xtest1 <=>
    # ytest0*(x-1)*xtest1 = ytest1 * x*(xtest0-1) <=>
    # ytest0*x*xtest1 - ytest0*xtest1 = ytest1*xtest0*x - ytest1*x <=>
    # x = (ytest0 * xtest1)/(-xtest0*ytest1 + ytest1 + ytest0 * xtest1)
    # gpu y0 = 182, cpu y1 = 90, x = (182 * 0.1)/(-0.1*90 + 90 + 182*0.1)

    # x = y1x0/(x0(y1-y2)+y2) = 182*0.1/(0.1*(182-90)+90)
    f_opt = (R_G * f_test)/(f_test*(R_G-R_C)+R_C)

    # clamp
    if math.isnan(f_opt):
        return None
    f_opt = max(0.0, min(1.0, f_opt))
    return f_opt


def extract_fraction_from_name(name):
    """
    Extract numeric worksplit fraction from folder name e.g. ws30 => 0.30
    """
    m = float(name.replace(",", "."))
    if not m:
        return None
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", help="Path to case folder containing worksplit runs")
    parser.add_argument("--cubedump", type=str, default="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CubeLib/4.9-GCCcore-13.3.0/bin/cube_dump",
                        help='Path to cube_dump executable')
    args = parser.parse_args()

    case_dir = args.case_dir
    pathToCubeDump = args.cubedump

    results = []

    for sub in os.listdir(case_dir):
        subpath = os.path.join(case_dir, sub)
        if not os.path.isdir(subpath):
            continue

        f_test = extract_fraction_from_name(sub)
        if f_test is None:
            continue

        print(f"Processing worksplit {sub} (f_test={f_test:.2f})...")

        R_G, R_C = process_subfolder(subpath, pathToCubeDump)
        if R_G is None or R_C is None:
            print(f"  -> Missing Score-P or dumps.")
            continue

        f_opt = compute_f_opt(f_test, R_G, R_C, O=0.0)
        if f_opt is None:
            print("  -> Could not compute optimal worksplit.")
            continue

        print(f"  R_G={R_G:.4f}s  R_C={R_C:.4f}s  --> f_opt={f_opt:.4f}")
        results.append((sub, f_test, R_G, R_C, f_opt))

    print("\n=== Independent Optimal Splits (one-point inference for each) ===")
    from operator import itemgetter      
    for r in sorted(results, key=itemgetter(1)):
        sub, f_test, R_G, R_C, f_opt = r
        print(f"{sub:20s}  test={f_test:.2f}  f_opt={f_opt:.4f}")


if __name__ == "__main__":
    main()
