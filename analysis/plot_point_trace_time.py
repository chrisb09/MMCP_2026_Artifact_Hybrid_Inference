import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

def load_hdf5_index(filepath):
    index = {}
    with h5py.File(filepath, 'r') as f:
        for gname in f.keys():
            grp = f[gname]
            ts = grp.attrs.get('globalTimeStep', None)
            tp = grp.attrs.get('type', None)
            if ts is not None and tp is not None:
                if isinstance(tp, bytes):
                    tp = tp.decode()
                index[(int(ts), tp)] = gname
    return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--field', default='U')
    parser.add_argument('--out-plot', default='analysis/comparison_plot_time.png')
    parser.add_argument('--target-idx', type=int, default=-1, help='Index to trace (-1 to auto-detect max diff at last step)')
    args = parser.parse_args()

    ref_index = load_hdf5_index(args.ref)
    new_index = load_hdf5_index(args.new)

    common_keys = sorted(set(ref_index.keys()).intersection(set(new_index.keys())))
    
    if not common_keys:
        print("No common keys found.")
        return

    # Find target index if not provided
    target_idx = args.target_idx
    if target_idx == -1:
        # Find the last 'received' step
        last_step = [k for k in common_keys if k[1] == 'received'][-1]
        with h5py.File(args.ref, 'r') as f_ref, h5py.File(args.new, 'r') as f_new:
            ref_val = np.array(f_ref[ref_index[last_step]][args.field]).flatten()
            new_val = np.array(f_new[new_index[last_step]][args.field]).flatten()
            diff = np.abs(ref_val - new_val)
            target_idx = np.argmax(diff)
            print(f"Auto-detected index {target_idx} (Max diff {diff[target_idx]:.4e} at step {last_step[0]} {last_step[1]})")

    time_steps = []
    ref_trace = []
    new_trace = []
    diff_trace = []

    with h5py.File(args.ref, 'r') as f_ref, h5py.File(args.new, 'r') as f_new:
        for (ts, tp) in common_keys:
            # We can plot both sent and received, but to make a clean line plot over time, 
            # we might want to just plot 'sent' steps + 'received' steps, adding a slight offset for received.
            # To make it strictly chronological: 'sent' happens before 'received'.
            # We will use ts as the x-axis. If there is a 'sent' and 'received' for the same ts,
            # we'll plot 'received' at ts + 0.1 so they don't overlap perfectly on x.
            
            x_val = ts if tp == 'sent' else ts + 0.1
            
            ref_val = np.array(f_ref[ref_index[(ts, tp)]][args.field]).flatten()[target_idx]
            new_val = np.array(f_new[new_index[(ts, tp)]][args.field]).flatten()[target_idx]
            
            time_steps.append(x_val)
            ref_trace.append(ref_val)
            new_trace.append(new_val)
            diff_trace.append(np.abs(ref_val - new_val))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Time Trace for Field {args.field}, Index {target_idx}", fontsize=14)

    axes[0].plot(time_steps, ref_trace, marker='.', label="Reference", color="blue")
    axes[0].set_ylabel(f"Ref {args.field}")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_steps, new_trace, marker='.', label="New CMI", color="orange")
    axes[1].set_ylabel(f"New {args.field}")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time_steps, diff_trace, marker='x', label="Abs Diff", color="red")
    axes[2].set_ylabel("Absolute Difference")
    axes[2].set_xlabel("Global Time Step (received offset by +0.1)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"Plot saved to {args.out_plot}")

if __name__ == "__main__":
    main()
