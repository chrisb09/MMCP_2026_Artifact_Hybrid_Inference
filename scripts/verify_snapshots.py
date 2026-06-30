#!/usr/bin/env python3
"""
verify_snapshots.py — Compare new MAIA snapshots against the golden reference.

Usage:
    python verify_snapshots.py <new_snapshots.h5> <reference_snapshots.h5> [--tol FLOAT]

Compares per-step U/V/W fields (both "sent" and "received" groups) between the
new run output and the golden reference. Reports maximum absolute difference
and L2 norm per field per step. Passes if all fields are within tolerance
(default 1e-4 for float-based comparison).
"""

import argparse
import sys
import numpy as np
import h5py

def compare_snapshots(new_path, reference_path, tol=1e-4):
    with h5py.File(new_path, 'r') as new_f, h5py.File(reference_path, 'r') as ref_f:
        new_groups = sorted([g for g in new_f.keys() if g.startswith('step_')])
        ref_groups = sorted([g for g in ref_f.keys() if g.startswith('step_')])

        common = set(new_groups) & set(ref_groups)
        if not common:
            print("No common step groups found between files.")
            return False

        print(f"Comparing {len(common)} common step groups (out of {len(new_groups)} new, {len(ref_groups)} ref)")
        all_pass = True

        for group_name in sorted(common):
            new_group = new_f[group_name]
            ref_group = ref_f[group_name]

            new_type = new_group.attrs.get('type', '')
            ref_type = ref_group.attrs.get('type', '')
            if new_type != ref_type:
                print(f"  {group_name}: type mismatch ({new_type} vs {ref_type})")
                all_pass = False
                continue

            for field in ['U', 'V', 'W']:
                if field not in new_group or field not in ref_group:
                    print(f"  {group_name}/{field}: missing in one file")
                    all_pass = False
                    continue

                new_data = np.array(new_group[field])
                ref_data = np.array(ref_group[field])

                if new_data.shape != ref_data.shape:
                    print(f"  {group_name}/{field}: shape mismatch {new_data.shape} vs {ref_data.shape}")
                    all_pass = False
                    continue

                diff = np.abs(new_data - ref_data)
                max_abs = np.max(diff)
                l2 = np.sqrt(np.sum(diff ** 2)) / max(1, np.sqrt(new_data.size))

                if max_abs > tol:
                    print(f"  {group_name}/{field} [{new_type}]: max_abs={max_abs:.6e}  L2_rel={l2:.6e}  **FAIL**")
                    all_pass = False
                else:
                    print(f"  {group_name}/{field} [{new_type}]: max_abs={max_abs:.6e}  L2_rel={l2:.6e}  OK")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description='Verify MAIA snapshots against reference')
    parser.add_argument('new_snapshots', help='HDF5 file with new simulation output')
    parser.add_argument('reference_snapshots', help='HDF5 file with golden reference')
    parser.add_argument('--tol', type=float, default=1e-4, help='Float tolerance (default 1e-4)')
    args = parser.parse_args()

    ok = compare_snapshots(args.new_snapshots, args.reference_snapshots, args.tol)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
