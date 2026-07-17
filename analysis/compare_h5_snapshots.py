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


def sort_key(step_type):
    ts, tp = step_type
    return (ts, 0 if tp == 'sent' else 1)


def aligned_field_values(ref_grp, new_grp, field):
    """Return flattened field values over the common global cell extent."""
    ref_val = np.array(ref_grp[field])
    new_val = np.array(new_grp[field])

    if ref_val.ndim != new_val.ndim:
        raise ValueError(f"Field rank mismatch: {ref_val.shape} vs {new_val.shape}")

    ref_offset = np.array(ref_grp["nOffsetCells"] if "nOffsetCells" in ref_grp else np.zeros(ref_val.ndim), dtype=int)
    new_offset = np.array(new_grp["nOffsetCells"] if "nOffsetCells" in new_grp else np.zeros(new_val.ndim), dtype=int)
    ref_shape = np.array(ref_val.shape, dtype=int)
    new_shape = np.array(new_val.shape, dtype=int)
    start = np.maximum(ref_offset, new_offset)
    stop = np.minimum(ref_offset + ref_shape, new_offset + new_shape)

    if np.any(stop <= start):
        raise ValueError(
            f"No overlapping cells for {field}: ref offset/shape {ref_offset}/{ref_shape}, "
            f"new offset/shape {new_offset}/{new_shape}"
        )

    ref_slices = tuple(slice(start[i] - ref_offset[i], stop[i] - ref_offset[i]) for i in range(ref_val.ndim))
    new_slices = tuple(slice(start[i] - new_offset[i], stop[i] - new_offset[i]) for i in range(new_val.ndim))
    return ref_val[ref_slices].flatten(), new_val[new_slices].flatten(), tuple(stop - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='Path to reference HDF5')
    parser.add_argument('--new', required=True, help='Path to new HDF5')
    parser.add_argument('--field', default='U', help='Field to compare (U, V, W)')
    parser.add_argument('--out-txt', default='analysis/summary_table.txt', help='Output text file')
    parser.add_argument('--out-plot', default='analysis/comparison_plot.png', help='Output plot file')
    parser.add_argument('--plot-step', type=int, default=15, help='Global step to plot')
    parser.add_argument('--plot-type', default='received', help='Type to plot (sent or received)')
    parser.add_argument('--plot-index', type=int, default=-1, help='Flattened cell index to trace across time steps')
    parser.add_argument('--steps', default='', help='Comma-separated globalTimeStep values to compare (empty = all)')
    parser.add_argument('--max-abs-diff', type=float, default=None, help='Fail if any absolute difference exceeds this threshold')
    args = parser.parse_args()

    print(f"Loading indices...")
    ref_index = load_hdf5_index(args.ref)
    new_index = load_hdf5_index(args.new)

    common_keys = sorted(set(ref_index.keys()).intersection(set(new_index.keys())), key=sort_key)
    print(f"Found {len(common_keys)} matching steps/types.")

    selected_steps = None
    if args.steps.strip():
        selected_steps = {int(step.strip()) for step in args.steps.split(',') if step.strip()}
        common_keys = [item for item in common_keys if item[0] in selected_steps]
        print(f"Restricted to steps: {sorted(selected_steps)} -> {len(common_keys)} matching steps/types.")

    out_dir = os.path.dirname(args.out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_txt, 'w') as f:
        header = f"{'Step':>6} | {'Type':>8} | {'Min Diff':>12} | {'Max Diff':>12} | {'Avg Diff':>12} | {'Med Diff':>12}"
        f.write(f"Summary of absolute differences for field {args.field}\n")
        f.write("-" * len(header) + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        plot_data_ref = None
        plot_data_new = None
        plot_step_type = None
        threshold_exceeded = False
        threshold_message = None

        with h5py.File(args.ref, 'r') as f_ref, h5py.File(args.new, 'r') as f_new:
            for (ts, tp) in common_keys:
                ref_grp = f_ref[ref_index[(ts, tp)]]
                new_grp = f_new[new_index[(ts, tp)]]

                ref_val, new_val, overlap_shape = aligned_field_values(ref_grp, new_grp, args.field)
                if tuple(ref_grp[args.field].shape) != tuple(new_grp[args.field].shape):
                    print(f"Step {ts} {tp}: comparing common global extent {overlap_shape} "
                          f"from ref {tuple(ref_grp[args.field].shape)} and new {tuple(new_grp[args.field].shape)}")

                diff = np.abs(ref_val - new_val)
                min_d = np.min(diff)
                max_d = np.max(diff)
                avg_d = np.mean(diff)
                med_d = np.median(diff)

                line = f"{ts:6d} | {tp:>8} | {min_d:12.4e} | {max_d:12.4e} | {avg_d:12.4e} | {med_d:12.4e}"
                f.write(line + "\n")
                print(line)

                if args.max_abs_diff is not None and max_d > args.max_abs_diff:
                    threshold_exceeded = True
                    threshold_message = (
                        f"Absolute diff {max_d:.4e} exceeds threshold {args.max_abs_diff:.4e} at step {ts} {tp}"
                    )

                if ts == args.plot_step and tp == args.plot_type:
                    plot_data_ref = ref_val
                    plot_data_new = new_val
                    plot_step_type = (ts, tp)
                    
    print(f"Summary table written to {args.out_txt}")

    if plot_data_ref is not None and plot_data_new is not None:
        if args.plot_index >= 0:
            target_idx = args.plot_index
            print(f"Using requested trace index {target_idx}")
        else:
            diff = np.abs(plot_data_ref - plot_data_new)
            target_idx = int(np.argmax(diff))
            print(f"Auto-selected trace index {target_idx} from step {plot_step_type[0]} {plot_step_type[1]} with max diff {diff[target_idx]:.4e}")

        x_vals = []
        ref_trace = []
        new_trace = []
        diff_trace = []

        with h5py.File(args.ref, 'r') as f_ref, h5py.File(args.new, 'r') as f_new:
            for (ts, tp) in common_keys:
                x_vals.append(ts + (0.1 if tp == 'received' else 0.0))
                ref_val, new_val, _ = aligned_field_values(
                    f_ref[ref_index[(ts, tp)]], f_new[new_index[(ts, tp)]], args.field)
                ref_val = ref_val[target_idx]
                new_val = new_val[target_idx]
                ref_trace.append(ref_val)
                new_trace.append(new_val)
                diff_trace.append(np.abs(ref_val - new_val))

        diff = np.asarray(diff_trace)
        max_idx = int(np.argmax(diff))
        max_val = float(diff[max_idx])

        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Field {args.field} trace at flat index {target_idx}", fontsize=16)

        axes[0].plot(x_vals, ref_trace, label="Reference", color="blue", marker='.', linewidth=0.8, alpha=0.9)
        axes[0].set_ylabel("Reference Value")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend(loc="upper right")

        axes[1].plot(x_vals, new_trace, label="New CMI Output", color="orange", marker='.', linewidth=0.8, alpha=0.9)
        axes[1].set_ylabel("New Output Value")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(loc="upper right")

        axes[2].plot(x_vals, diff, label="Absolute Difference", color="red", marker='x', linewidth=0.8, alpha=0.9)
        axes[2].set_ylabel("Abs Diff")
        axes[2].set_xlabel("Global Time Step")
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        # Mark max diff
        axes[2].plot(x_vals[max_idx], max_val, marker='o', markersize=8, color='black', markerfacecolor='none', markeredgewidth=2)
        axes[2].annotate(f"Max Diff:\n{max_val:.4e}", 
                         xy=(x_vals[max_idx], max_val), 
                         xytext=(x_vals[max_idx], max_val + (np.max(diff)*0.2 if np.max(diff)>0 else 0.1)),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                         fontsize=12, ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(args.out_plot, dpi=150)
        print(f"Plot written to {args.out_plot}")
    else:
        print(f"Warning: Step {args.plot_step} and type {args.plot_type} not found. Plot not generated.")

    if args.max_abs_diff is not None:
        print(f"Max abs diff threshold was {args.max_abs_diff:.4e}")
        if threshold_exceeded:
            raise SystemExit(threshold_message)

if __name__ == "__main__":
    main()
