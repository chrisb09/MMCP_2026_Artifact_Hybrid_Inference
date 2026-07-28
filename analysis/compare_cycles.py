import h5py
import numpy as np
import os

files = {
    'Ref': '/rwthfs/rz/cluster/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5',
    'AIX': 'scratch/current_aix_cpu_2314057/debug/snapshots/snapshots_2314057_rank_0.h5',
    'SmartSim': 'debug_dumps/smartsim/snapshots_2182876_rank_0.h5',
    'PhyDLL-C++': 'debug_dumps/phydll_cpp/snapshots_2183300_rank_0.h5',
    'PhyDLL-Py': 'debug_dumps/phydll_py/snapshots_2182319_rank_0.h5',
}

indices = {}
for name, path in files.items():
    idx = {}
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        continue
    with h5py.File(path, 'r') as f:
        for gname in f.keys():
            grp = f[gname]
            ts = grp.attrs.get('globalTimeStep', None)
            tp = grp.attrs.get('type', None)
            if ts is not None and tp is not None:
                if isinstance(tp, bytes): tp = tp.decode()
                idx[(int(ts), tp)] = gname
    indices[name] = idx

all_steps = sorted(list(set.union(*[set(idx.keys()) for idx in indices.values()])), key=lambda x: (x[0], 0 if x[1]=='sent' else 1))

print(f"{'Step':<6} | {'Type':<8} | {'VS':<18} | {'Max Abs U':<12} | {'Rel L2 U':<12} | {'Max Abs V':<12} | {'Max Abs W':<12}")
print('-' * 96)

target_steps = [11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 97, 98, 99, 100, 101]

for (ts, tp) in all_steps:
    if ts not in target_steps: continue
    
    aix_g = indices['AIX'].get((ts, tp))
    ref_g = indices['Ref'].get((ts, tp))
    
    if aix_g and ref_g:
        with h5py.File(files['AIX'], 'r') as f_aix, h5py.File(files['Ref'], 'r') as f_ref:
            a_u = np.array(f_aix[aix_g]['U']).flatten()
            a_v = np.array(f_aix[aix_g]['V']).flatten()
            a_w = np.array(f_aix[aix_g]['W']).flatten()
            
            r_u = np.array(f_ref[ref_g]['U']).flatten()
            r_v = np.array(f_ref[ref_g]['V']).flatten()
            r_w = np.array(f_ref[ref_g]['W']).flatten()

            max_u = np.max(np.abs(a_u - r_u))
            rel_u = np.linalg.norm(a_u - r_u) / (np.linalg.norm(r_u) + 1e-12)
            max_v = np.max(np.abs(a_v - r_v))
            max_w = np.max(np.abs(a_w - r_w))
            print(f"{ts:<6d} | {tp:<8} | {'AIX vs Ref':<18} | {max_u:12.4e} | {rel_u:12.4e} | {max_v:12.4e} | {max_w:12.4e}")

        for pname in ['PhyDLL-C++', 'SmartSim', 'PhyDLL-Py']:
            pg = indices[pname].get((ts, tp))
            if not pg: continue
            with h5py.File(files[pname], 'r') as f_p, h5py.File(files['AIX'], 'r') as f_aix:
                p_u = np.array(f_p[pg]['U']).flatten()
                p_v = np.array(f_p[pg]['V']).flatten()
                p_w = np.array(f_p[pg]['W']).flatten()
                
                max_u = np.max(np.abs(p_u - a_u))
                rel_u = np.linalg.norm(p_u - a_u) / (np.linalg.norm(a_u) + 1e-12)
                max_v = np.max(np.abs(p_v - a_v))
                max_w = np.max(np.abs(p_w - a_w))
                tag = f"{pname} vs AIX"
                print(f"{'':<6} | {'':<8} | {tag:<18} | {max_u:12.4e} | {rel_u:12.4e} | {max_v:12.4e} | {max_w:12.4e}")
        print('-' * 96)
