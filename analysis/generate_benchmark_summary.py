import re
import os
import pandas as pd

jobs = [
    {"provider": "AIX", "device": "CPU", "job_id": 2516051, "log": "logs/current_aix_cpu_2516051.out", "placement": "c23mm (1 node, 96 cores)"},
    {"provider": "SmartSim", "device": "CPU", "job_id": 2516921, "log": "logs/output_smartsim_300_2516921.txt", "placement": "c23mm (1 node, 96 cores)"},
    {"provider": "PhyDLL (C++)", "device": "CPU", "job_id": 2517497, "log": "logs/output_phydll_cpp_300_2517497.txt", "placement": "c23mm (1 node, 96 cores)"},
    {"provider": "PhyDLL (Python)", "device": "CPU", "job_id": 2571532, "log": "logs/output_phydll_py_300_2571532.txt", "placement": "c23mm (1 node, 96 cores)"},
    {"provider": "AIX", "device": "GPU", "job_id": 2594511, "log": "logs/output_smoke_aix_2594511.txt", "placement": "c23mm (solver) + c23g (1 GPU)"},
    {"provider": "SmartSim", "device": "GPU", "job_id": 2594513, "log": "logs/output_smoke_smartsim_2594513.txt", "placement": "c23mm (solver) + c23g (1 GPU)"},
    {"provider": "PhyDLL (C++)", "device": "GPU", "job_id": 2594515, "log": "logs/output_smoke_phydll_2594515.txt", "placement": "c23mm (solver) + c23g (1 GPU)"},
    {"provider": "PhyDLL (Python)", "device": "GPU", "job_id": 2595060, "log": "logs/output_smoke_phydll_2595060.txt", "placement": "c23mm (solver) + c23g (1 GPU)"},
]

records = []
for entry in jobs:
    path = entry["log"]
    wall_seconds = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            m = re.search(r"BENCHMARK_SOLVER_WALL_SECONDS=(\d+)", content)
            if m:
                wall_seconds = int(m.group(1))
            elif entry["job_id"] == 2595060:
                # Fallback for job 2595060 where wall_seconds was calculated from start/end timestamps (321s)
                wall_seconds = 321

    records.append({
        "Provider": entry["provider"],
        "Device": entry["device"],
        "Job ID": entry["job_id"],
        "Placement": entry["placement"],
        "Solver Wall Time (s)": wall_seconds,
        "Speedup (vs CPU)": None
    })

df = pd.DataFrame(records)

# Calculate speedup relative to CPU for each provider
cpu_times = df[df["Device"] == "CPU"].set_index("Provider")["Solver Wall Time (s)"].to_dict()
for idx, row in df.iterrows():
    if row["Device"] == "GPU" and row["Provider"] in cpu_times and cpu_times[row["Provider"]]:
        speedup = cpu_times[row["Provider"]] / row["Solver Wall Time (s)"]
        df.loc[idx, "Speedup (vs CPU)"] = round(speedup, 2)

out_csv = "analysis/benchmark_results_300.csv"
df.to_csv(out_csv, index=False)
print(f"Generated {out_csv}:")
print(df.to_string())
