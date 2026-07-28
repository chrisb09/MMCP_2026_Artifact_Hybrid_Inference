import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

files = {
    'AIX (Current)': 'scratch/current_aix_cpu_2210454/debug/snapshots/snapshots_2210454_rank_0.h5',
    'SmartSim': 'debug_dumps/smartsim/snapshots_2182876_rank_0.h5',
    'PhyDLL-C++': 'debug_dumps/phydll_cpp/snapshots_2183300_rank_0.h5',
    'PhyDLL-Py': 'debug_dumps/phydll_py/snapshots_2182319_rank_0.h5',
}
ref_path = '/rwthfs/rz/cluster/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5'

def get_index(filepath):
    idx = {}
    with h5py.File(filepath, 'r') as f:
        for gname in f.keys():
            grp = f[gname]
            ts = grp.attrs.get('globalTimeStep', None)
            tp = grp.attrs.get('type', None)
            if ts is not None and tp is not None:
                if isinstance(tp, bytes): tp = tp.decode()
                idx[(int(ts), tp)] = gname
    return idx

ref_idx = get_index(ref_path)
aix_idx = get_index(files['AIX (Current)'])

provider_indices = {p: get_index(f) for p, f in files.items()}

# Find matching steps present in AIX
common_keys = sorted(aix_idx.keys(), key=lambda x: (x[0], 0 if x[1]=='sent' else 1))

x_steps = []
aix_vs_ref = []
providers_vs_aix = {p: [] for p in ['SmartSim', 'PhyDLL-C++', 'PhyDLL-Py']}

with h5py.File(ref_path, 'r') as f_ref, h5py.File(files['AIX (Current)'], 'r') as f_aix:
    f_provs = {p: h5py.File(files[p], 'r') for p in ['SmartSim', 'PhyDLL-C++', 'PhyDLL-Py']}
    
    for (ts, tp) in common_keys:
        if ts > 105: continue
        
        a_g = aix_idx[(ts, tp)]
        a_u = np.array(f_aix[a_g]['U']).flatten()
        
        # AIX vs Ref
        r_g = ref_idx.get((ts, tp))
        if r_g:
            r_u = np.array(f_ref[r_g]['U']).flatten()
            diff_ref = np.max(np.abs(a_u - r_u))
        else:
            diff_ref = np.nan
            
        x_steps.append(f"{ts}_{tp[:1]}")
        aix_vs_ref.append(diff_ref)
        
        for p in ['SmartSim', 'PhyDLL-C++', 'PhyDLL-Py']:
            p_g = provider_indices[p].get((ts, tp))
            if p_g:
                p_u = np.array(f_provs[p][p_g]['U']).flatten()
                diff_aix = np.max(np.abs(p_u - a_u))
            else:
                diff_aix = np.nan
            providers_vs_aix[p].append(diff_aix)

    for f in f_provs.values(): f.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top plot: Providers vs Current AIX Baseline
x_indices = np.arange(len(x_steps))
for p, diffs in providers_vs_aix.items():
    ax1.plot(x_indices, diffs, label=f"{p} vs AIX", marker='o', linewidth=1.5, alpha=0.8)

ax1.set_yscale('log')
ax1.set_ylabel('Max Abs Diff in U vs Current AIX')
ax1.set_title('Inter-Provider Consistency (Providers vs Current AIX Baseline)')
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

# Bottom plot: Current AIX vs Legacy Golden Reference
ax2.plot(x_indices, aix_vs_ref, label="Current AIX vs Legacy Golden Reference", color='black', marker='s', linewidth=2)
ax2.set_yscale('log')
ax2.set_ylabel('Max Abs Diff in U vs Legacy Ref')
ax2.set_xlabel('Time Step & Snapshot Type (e.g. 15_r = 15 received)')
ax2.set_title('Trajectory Shift due to Initial LibTorch/CFD Evolution (Current AIX vs Legacy Ref)')
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend(loc="upper left")

plt.xticks(x_indices, x_steps, rotation=45, ha='right', fontsize=9)
plt.tight_layout()
os.makedirs('analysis', exist_ok=True)
out_png = 'analysis/provider_breakdown_aix_vs_ref.png'
plt.savefig(out_png, dpi=150)
print(f"Saved breakdown plot to {out_png}")
