#!/usr/bin/env python3
"""
verify_inference.py — Standalone Python reimplementation of the FlowExtrapolator
pipeline for step-15 divergence analysis.

Steps performed:
  1. Load reference snapshots (steps 11-15) from HDF5.
  2. Read cube layout from debug_dumps/cube_layout.txt.
  3. Reimplement cube extraction and input tensor assembly exactly as the C++ does.
  4. Load the TorchScript model and run inference in Python.
  5. Reimplement output reconstruction (scatter + weight normalization).
  6. Compare at each stage against C++ debug dumps.
  7. Print a structured comparison table and decision-tree conclusion.

Usage:
    python verify_inference.py [--ref REF_H5] [--dumps DUMP_DIR] [--model MODEL_PT]
                               [--inference-step STEP] [--tol TOL]

Defaults:
    --ref            /hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5
    --dumps          debug_dumps
    --model          input/transformer_inference_scripted_fw2.pt
    --inference-step 15  (the first inference step)
    --tol            1e-5
"""

import argparse
import os
import sys
import struct
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports — warn gracefully if missing
# ---------------------------------------------------------------------------
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Cube layout reader
# ---------------------------------------------------------------------------

def parse_int_list(s: str):
    """Parse '[a, b, c]' -> [a, b, c]."""
    s = s.strip().strip('[]')
    return [int(x) for x in s.split(',') if x.strip()]


def load_cube_layout(path: str) -> dict:
    """Parse cube_layout.txt produced by FlowExtrapolator's #ifdef block."""
    layout = {}
    with open(path) as f:
        lines = f.readlines()

    # First pass: simple key: value lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if ': ' in line and not line.startswith('cube_volume_indices:'):
            key, _, val = line.partition(': ')
            key = key.strip()
            val = val.strip()
            if val.startswith('[') and not val.endswith(']'):
                # Multi-token list on same line — should be closed
                layout[key] = parse_int_list(val)
            elif val.startswith('['):
                layout[key] = parse_int_list(val)
            else:
                try:
                    layout[key] = int(val)
                except ValueError:
                    layout[key] = val
            i += 1
        elif line.startswith('cube_volume_indices:'):
            # Parse multi-line block
            cubes = []
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l == ']':
                    break
                if l.startswith('cube_') and ': ' in l:
                    _, _, vals = l.partition(': ')
                    cubes.append(parse_int_list(vals))
                i += 1
            layout['cube_volume_indices'] = cubes
            i += 1
        else:
            i += 1

    return layout


# ---------------------------------------------------------------------------
# C++ dump reader
# ---------------------------------------------------------------------------

def load_bin(path: str, dtype=np.float32) -> np.ndarray:
    """Load a raw binary dump written by write_bin<float>."""
    return np.fromfile(path, dtype=dtype)


def load_manifest(dump_dir: str) -> list:
    """Parse manifest.txt -> list of dicts."""
    path = os.path.join(dump_dir, 'manifest.txt')
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            records.append({
                'filename':          parts[0],
                'global_step':       int(parts[1]),
                'step_type':         parts[2],
                'phase':             parts[3],
                'field_index':       int(parts[4]),
                'n_elements':        int(parts[5]),
                'element_size_bytes': int(parts[6]),
                'shape':             parts[7],
            })
    return records


def find_dump(dump_dir: str, step: int, phase: str, field: Optional[int] = None, rank: int = 0) -> Optional[str]:
    """Construct the expected filename and check existence."""
    # First try with rank in name
    if field is not None:
        fname_rank = f"step{step:04d}_rank{rank}_{phase}_field{field}.bin"
    else:
        fname_rank = f"step{step:04d}_rank{rank}_{phase}.bin"
    p_rank = os.path.join(dump_dir, fname_rank)
    if os.path.exists(p_rank):
        return p_rank

    # Fallback to no rank in name (for older dumps)
    if field is not None:
        fname_norank = f"step{step:04d}_{phase}_field{field}.bin"
    else:
        fname_norank = f"step{step:04d}_{phase}.bin"
    p_norank = os.path.join(dump_dir, fname_norank)
    if os.path.exists(p_norank):
        return p_norank

    return None


# ---------------------------------------------------------------------------
# Python reimplementation of FlowExtrapolator preprocessing
# ---------------------------------------------------------------------------

def linspace_int(start: int, end: int, count: int) -> list:
    """Mirrors the C++ linspace helper."""
    if count <= 0:
        return []
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [int(round(start + i * step)) for i in range(count)]


def get_full_indices(length: int, cube_dim: int, step: int) -> list:
    """Mirrors the C++ get_full_indices helper."""
    indices = []
    i = 0
    while i <= length - cube_dim:
        indices.append(i)
        i += step
    if not indices or indices[-1] != length - cube_dim:
        indices.append(length - cube_dim)
    return indices


def build_cube_starts(active_cells: list, cube_dim: int, cube_overlap: int):
    zs = (linspace_int(0, active_cells[0] - cube_dim,
                       int(np.ceil(active_cells[0] / cube_dim)))
          if cube_overlap == 0 else
          get_full_indices(active_cells[0], cube_dim, cube_dim - cube_overlap))
    ys = (linspace_int(0, active_cells[1] - cube_dim,
                       int(np.ceil(active_cells[1] / cube_dim)))
          if cube_overlap == 0 else
          get_full_indices(active_cells[1], cube_dim, cube_dim - cube_overlap))
    xs = (linspace_int(0, active_cells[2] - cube_dim,
                       int(np.ceil(active_cells[2] / cube_dim)))
          if cube_overlap == 0 else
          get_full_indices(active_cells[2], cube_dim, cube_dim - cube_overlap))
    return zs, ys, xs


def build_cube_volume_indices(zs, ys, xs, cube_dim, n_cells, n_ghost):
    """Rebuilds cube_volume_indices_ from scratch — same loop order as C++."""
    yz_stride = n_cells[1] * n_cells[2]
    row_stride = n_cells[2]
    mapping_list = []
    for z0 in zs:
        for y0 in ys:
            for x0 in xs:
                mapping = []
                for dz in range(cube_dim):
                    gz = z0 + dz + n_ghost
                    for dy in range(cube_dim):
                        gy = y0 + dy + n_ghost
                        for dx in range(cube_dim):
                            gx = x0 + dx + n_ghost
                            mapping.append(gz * yz_stride + gy * row_stride + gx)
                mapping_list.append(mapping)
    return mapping_list


def extract_field_cubes(field_flat: np.ndarray, cube_volume_indices: list,
                        cube_size: int) -> np.ndarray:
    """Mirrors C++ extract_field_cubes."""
    num_cubes = len(cube_volume_indices)
    cubes = np.empty(num_cubes * cube_size, dtype=field_flat.dtype)
    for c, mapping in enumerate(cube_volume_indices):
        cubes[c * cube_size:(c + 1) * cube_size] = field_flat[mapping]
    return cubes


def assemble_input_tensor(history: list, cube_volume_indices: list,
                           num_cubes: int, cube_size: int,
                           input_sequence_length: int,
                           field_count: int = 3) -> np.ndarray:
    """
    Mirrors the C++ preprocess() loop.
    history: list of length <= input_sequence_length, each element is
             list of 3 field-cube arrays (one per field).
    Returns array of shape [field_count * num_cubes, input_sequence_length, cube_size].
    """
    total = field_count * num_cubes * input_sequence_length * cube_size
    buf = np.zeros(total, dtype=np.float32)

    hist_size = len(history)
    for seq in range(input_sequence_length):
        # resolve_history_index logic
        missing = input_sequence_length - hist_size
        if seq < missing:
            history_index = 0
        else:
            history_index = seq - missing

        step_fields = history[history_index]  # list of 3 field-cube arrays
        for field in range(field_count):
            field_cubes = step_fields[field]  # shape: [num_cubes * cube_size]
            for cube in range(num_cubes):
                batch_index = field * num_cubes + cube
                dst_off = (batch_index * input_sequence_length + seq) * cube_size
                src_off = cube * cube_size
                buf[dst_off:dst_off + cube_size] = field_cubes[src_off:src_off + cube_size]

    return buf.reshape(field_count * num_cubes, input_sequence_length, cube_size)


def reconstruct_output(model_output_flat: np.ndarray,
                       cube_volume_indices: list,
                       num_cubes: int, cube_size: int,
                       n_cells: list, n_ghost: int,
                       weight: np.ndarray,
                       forecast_window: int,
                       field_count: int = 3) -> list:
    """
    Mirrors C++ postprocess():
      - clear active region
      - scatter model output into output fields using cube_volume_indices
      - divide by weight
    Returns list of 3 field arrays (flat, shape [nz*ny*nx]).
    """
    N = n_cells[0] * n_cells[1] * n_cells[2]
    # model_output_flat should be [field_count * num_cubes * forecast_window * cube_size]
    model_output = model_output_flat.reshape(field_count * num_cubes, forecast_window, cube_size)

    fields = []
    for field in range(field_count):
        dst = np.zeros(N, dtype=np.float32)
        for cube in range(num_cubes):
            batch_index = field * num_cubes + cube
            # Take last forecast window slot (forecast_window - 1)
            src_off_base = batch_index * forecast_window + (forecast_window - 1)
            src_vals = model_output[batch_index, forecast_window - 1, :]
            mapping = cube_volume_indices[cube]
            for local in range(cube_size):
                dst[mapping[local]] += src_vals[local]
        # Divide by weight
        nonzero = weight > 0.0
        dst[nonzero] /= weight[nonzero].astype(np.float32)
        fields.append(dst)
    return fields


def build_weight(cube_volume_indices: list, cube_size: int, N: int) -> np.ndarray:
    w = np.zeros(N, dtype=np.float64)
    for mapping in cube_volume_indices:
        for idx in mapping:
            w[idx] += 1.0
    return w


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

RESET = '\033[0m'
GREEN = '\033[32m'
RED   = '\033[31m'
YELLOW = '\033[33m'


def fmt(ok: bool, max_abs: float, l2: float, label: str):
    color = GREEN if ok else RED
    mark = '✅' if ok else '❌'
    print(f"  {mark} {label:50s}  max_abs={max_abs:.3e}  L2={l2:.3e}  "
          f"{color}{'OK' if ok else 'FAIL'}{RESET}")
    return ok


def compare_arrays(a: np.ndarray, b: np.ndarray, label: str, tol: float) -> bool:
    if a.shape != b.shape:
        print(f"  ❌ {label}: shape mismatch {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(diff.max())
    l2 = float(np.sqrt((diff ** 2).sum()) / max(1, np.sqrt(diff.size)))
    return fmt(max_abs <= tol, max_abs, l2, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Verify FlowExtrapolator step-15 divergence')
    parser.add_argument('--ref', default='/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5',
                        help='Reference HDF5 file with step groups (default: %(default)s)')
    parser.add_argument('--dumps', default='debug_dumps',
                        help='Directory with C++ debug dumps (default: %(default)s)')
    parser.add_argument('--model', default='input/transformer_inference_scripted_fw2.pt',
                        help='TorchScript model path (default: %(default)s)')
    parser.add_argument('--inference-step', type=int, default=15,
                        help='Global solver step where inference occurs (default: %(default)s)')
    parser.add_argument('--send-steps', type=str, default='11,12,13,14,15',
                        help='Comma-separated list of send steps (default: %(default)s)')
    parser.add_argument('--dump-step-offset', type=int, default=0,
                        help='Offset between global solver step and debug_step_counter_ '
                             'in filenames (default: %(default)s)')
    parser.add_argument('--inference-dump-idx', type=int, default=None,
                        help='Explicit debug_step_counter_ value used for the inference '
                             'dump files (assembled_input, raw_output, reconstructed_output). '
                             'Overrides the default formula (len(send_steps)-1 + offset). '
                             'Use this when MPI rank races cause unexpected counter values '
                             '(e.g. --inference-dump-idx 1 for the current debug run).')
    parser.add_argument('--tol', type=float, default=1e-5,
                        help='Float comparison tolerance (default: %(default)s)')
    parser.add_argument('--rank', type=int, default=0,
                        help='MPI rank of C++ dumps to compare (default: %(default)s)')
    parser.add_argument('--no-model', action='store_true',
                        help='Skip model inference (compare input assembly only)')
    args = parser.parse_args()

    dump_dir = args.dumps
    tol = args.tol
    inference_step = args.inference_step
    send_steps = [int(s) for s in args.send_steps.split(',')]

    print("=" * 70)
    print("FlowExtrapolator Step-15 Divergence Analysis")
    print("=" * 70)

    # ---- 1. Load cube layout ------------------------------------------------
    layout_path = os.path.join(dump_dir, 'cube_layout.txt')
    if not os.path.exists(layout_path):
        print(f"ERROR: cube_layout.txt not found at {layout_path}")
        print("  → Run the debug build first (Steps 2+3 in DEBUG_PLAN.md)")
        sys.exit(1)

    print(f"\n[1/5] Loading cube layout from {layout_path}")
    layout = load_cube_layout(layout_path)
    cube_dim      = layout['cube_dimension']
    cube_overlap  = layout['cube_overlap']
    num_cubes     = layout['n_cubes_per_field']
    cube_size     = layout['cube_size']
    input_seq_len = layout['input_sequence_length']
    forecast_win  = layout['forecast_window']
    n_ghost       = layout['n_ghost_layers']
    grid_dims     = layout.get('grid_dims', None)
    active_cells  = layout.get('active_cells', None)

    if isinstance(grid_dims, str):
        grid_dims = parse_int_list(grid_dims)
    if isinstance(active_cells, str):
        active_cells = parse_int_list(active_cells)

    print(f"  cube_dimension={cube_dim}, overlap={cube_overlap}, "
          f"num_cubes={num_cubes}, cube_size={cube_size}")
    print(f"  input_seq_len={input_seq_len}, forecast_win={forecast_win}, "
          f"n_ghost={n_ghost}")
    print(f"  grid_dims={grid_dims}, active_cells={active_cells}")

    # Use stored cube_volume_indices if available, else rebuild
    if 'cube_volume_indices' in layout and layout['cube_volume_indices']:
        print("  Using cube_volume_indices from layout file.")
        cube_volume_indices = layout['cube_volume_indices']
    else:
        print("  Rebuilding cube_volume_indices from starts+ghost logic.")
        zs, ys, xs = build_cube_starts(active_cells, cube_dim, cube_overlap)
        cube_volume_indices = build_cube_volume_indices(
            zs, ys, xs, cube_dim, grid_dims, n_ghost)

    N = grid_dims[0] * grid_dims[1] * grid_dims[2]
    weight = build_weight(cube_volume_indices, cube_size, N)

    # ---- 2. Load reference snapshots ----------------------------------------
    print(f"\n[2/5] Loading reference snapshots from {args.ref}")
    if not HAS_H5PY:
        print("  WARNING: h5py not available. Skipping reference comparison.")
        ref_fields = {}
    elif not os.path.exists(args.ref):
        print(f"  WARNING: Reference file not found: {args.ref}")
        print("  → Reference comparisons will be skipped.")
        ref_fields = {}
    else:
        ref_fields = {}
        with h5py.File(args.ref, 'r') as f:
            all_groups = list(f.keys())
            print(f"  Available groups: {all_groups[:10]}{'...' if len(all_groups)>10 else ''}")

            # Build index from (globalTimeStep, type) -> group name by scanning attrs
            attr_index = {}
            for gname in all_groups:
                grp = f[gname]
                ts = grp.attrs.get('globalTimeStep', None)
                tp = grp.attrs.get('type', None)
                if ts is not None and tp is not None:
                    if isinstance(tp, bytes):
                        tp = tp.decode()
                    attr_index[(int(ts), tp)] = gname

            print(f"  Attribute index: {len(attr_index)} entries. Sample: {list(attr_index.items())[:4]}")

            for step in send_steps:
                for snap_type in ('sent', 'received'):
                    gname = attr_index.get((step, snap_type), None)
                    if gname is not None:
                        grp = f[gname]
                        ref_fields[(step, snap_type)] = {
                            field: np.array(grp[field]).flatten().astype(np.float32)
                            for field in ('U', 'V', 'W') if field in grp
                        }
                        print(f"  Loaded ({step}, {snap_type}) from '{gname}': "
                              f"shapes = { {k: v.shape for k,v in ref_fields[(step,snap_type)].items()} }")
                    else:
                        print(f"  WARNING: group for step {step} {snap_type} not found.")

    field_names = ['U', 'V', 'W']

    # ---- 3. Build Python input tensor from reference "sent" snapshots --------
    print(f"\n[3/5] Assembling Python input tensor for step {inference_step}")

    history = []
    for step in send_steps:
        snap = ref_fields.get((step, 'sent'), None)
        if snap is None:
            print(f"  WARNING: No reference 'sent' snapshot for step {step}. "
                  f"Using C++ dump if available.")
            # Fall back to C++ raw_input dump
            step_dump_idx = step - send_steps[0] + args.dump_step_offset
            field_cubes_list = []
            for fi in range(3):
                p = find_dump(dump_dir, step_dump_idx, 'raw_input', fi, rank=args.rank)
                if p:
                    raw = load_bin(p)
                    cubes = extract_field_cubes(raw, cube_volume_indices, cube_size)
                    field_cubes_list.append(cubes)
                else:
                    print(f"    No dump for step {step} raw_input field {fi} either. Using zeros.")
                    field_cubes_list.append(np.zeros(num_cubes * cube_size, dtype=np.float32))
            history.append(field_cubes_list)
        else:
            field_cubes_list = []
            for fi, fname in enumerate(field_names):
                raw = snap.get(fname, np.zeros(N, dtype=np.float32))
                cubes = extract_field_cubes(raw, cube_volume_indices, cube_size)
                field_cubes_list.append(cubes)
            history.append(field_cubes_list)

    py_assembled = assemble_input_tensor(
        history[-input_seq_len:] if len(history) >= input_seq_len else history,
        cube_volume_indices, num_cubes, cube_size, input_seq_len)
    print(f"  Python assembled_input shape: {py_assembled.shape}")

    # ---- 4. Compare input assembly with C++ dump ----------------------------
    print(f"\n[4/5] Comparing intermediate states")
    all_ok = True

    # Determine which dump step index corresponds to inference_step.
    # debug_step_counter_ increments once per ml_step() call (send or inference).
    # With 24 MPI ranks all writing to the same /tmp dir, only some step indices
    # survive. Use --inference-dump-idx to override if the empirical counter differs.
    if args.inference_dump_idx is not None:
        infer_dump_idx = args.inference_dump_idx
    else:
        infer_dump_idx = (len(send_steps) - 1) + args.dump_step_offset

    # 4a. raw_input for send steps vs reference "sent"
    print(f"\n  --- raw_input vs reference 'sent' ---")
    for si, step in enumerate(send_steps):
        dump_idx = si + args.dump_step_offset
        snap = ref_fields.get((step, 'sent'), None)
        for fi, fname in enumerate(field_names):
            cpp_path = find_dump(dump_dir, dump_idx, 'raw_input', fi, rank=args.rank)
            if cpp_path is None:
                print(f"  ⚠️  step{dump_idx:04d}_raw_input_field{fi}.bin not found — skipping.")
                continue
            cpp_raw = load_bin(cpp_path)
            if snap is not None and fname in snap:
                ref_raw = snap[fname]
                ok = compare_arrays(cpp_raw, ref_raw,
                                    f"step {step} raw_input field{fi}({fname}) vs ref sent", tol)
                all_ok = all_ok and ok

    # 4b. assembled_input: Python vs C++
    print(f"\n  --- assembled_input: Python vs C++ ---")
    cpp_ai_path = find_dump(dump_dir, infer_dump_idx, 'assembled_input', rank=args.rank)
    if cpp_ai_path is not None:
        cpp_ai = load_bin(cpp_ai_path)
        ok = compare_arrays(py_assembled.flatten(), cpp_ai,
                            f"Python assembled_input vs C++ step{infer_dump_idx:04d}", tol)
        all_ok = all_ok and ok
        input_match = ok
    else:
        print(f"  ⚠️  step{infer_dump_idx:04d}_assembled_input.bin not found — skipping.")
        input_match = None

    # 4c. Model inference
    py_raw_output = None
    if not args.no_model and HAS_TORCH and os.path.exists(args.model):
        print(f"\n  --- Model inference ---")
        print(f"  Loading model from {args.model}")
        model = torch.jit.load(args.model, map_location='cpu')
        model.eval()
        with torch.no_grad():
            # Match the C++ provider input exactly: [batch, seq, features]
            # = [field_count * num_cubes, input_seq_len, cube_size]. Do not add
            # an outer batch dimension; the TorchScript wrapper treats dim 0 as
            # the cube batch and returns [batch, forecast_window, cube_size].
            try:
                inp_tensor = torch.from_numpy(py_assembled)
                out_tensor = model(inp_tensor)
                py_raw_output = out_tensor.numpy().flatten()
                print(f"  Model inference succeeded (batch-first). Output shape: {out_tensor.shape}")
            except Exception as e1:
                print(f"  WARNING: batch-first inference failed: {e1}")
                # Fallback for future model variants using seq-first layout.
                try:
                    inp_tensor2 = torch.from_numpy(np.transpose(py_assembled, (1, 0, 2)))
                    out_tensor2 = model(inp_tensor2)
                    py_raw_output = out_tensor2.permute(1, 0, 2).numpy().flatten()
                    print(f"  Model inference succeeded (seq-first fallback). Output shape: {out_tensor2.shape}")
                except Exception as e2:
                    print(f"  WARNING: Model inference failed: {e2}")

        print(f"\n  --- raw_output: Python vs C++ ---")
        cpp_ro_path = find_dump(dump_dir, infer_dump_idx, 'raw_output', rank=args.rank)
        if cpp_ro_path is not None and py_raw_output is not None:
            cpp_ro = load_bin(cpp_ro_path)
            ok = compare_arrays(py_raw_output, cpp_ro,
                                f"Python raw_output vs C++ step{infer_dump_idx:04d}", tol)
            all_ok = all_ok and ok
            output_match = ok
        else:
            print(f"  ⚠️  raw_output dump or Python output not available — skipping.")
            output_match = None
    else:
        if args.no_model:
            print(f"\n  --- Skipping model inference (--no-model) ---")
        elif not HAS_TORCH:
            print(f"\n  --- Skipping model inference (torch not available) ---")
        else:
            print(f"\n  --- Skipping model inference (model not found: {args.model}) ---")
        output_match = None

    # 4d. Reconstructed output: Python vs C++ and vs reference "received"
    py_recon_fields = None
    if py_raw_output is not None:
        print(f"\n  --- Reconstructing Python output fields ---")
        try:
            py_recon_fields = reconstruct_output(
                py_raw_output, cube_volume_indices,
                num_cubes, cube_size, grid_dims, n_ghost,
                weight, forecast_win)
            print(f"  Python reconstructed field shapes: {[f.shape for f in py_recon_fields]}")
        except Exception as e:
            print(f"  WARNING: Reconstruction failed: {e}")

    print(f"\n  --- reconstructed_output: Python vs C++ ---")
    recon_match = True
    for fi, fname in enumerate(field_names):
        cpp_path = find_dump(dump_dir, infer_dump_idx, 'reconstructed_output', fi, rank=args.rank)
        if cpp_path is None:
            print(f"  ⚠️  step{infer_dump_idx:04d}_reconstructed_output_field{fi}.bin not found.")
            recon_match = False
            continue
        cpp_recon = load_bin(cpp_path)
        if py_recon_fields is not None:
            ok = compare_arrays(py_recon_fields[fi], cpp_recon,
                                f"Python recon field{fi}({fname}) vs C++", tol)
            all_ok = all_ok and ok
            recon_match = recon_match and ok
        else:
            print(f"  ⚠️  Python reconstructed output not available.")
            recon_match = False

    print(f"\n  --- reconstructed_output: Python vs reference 'received' ---")
    ref_recv = ref_fields.get((inference_step, 'received'), None)
    py_vs_ref_ok = True
    if ref_recv and py_recon_fields is not None:
        for fi, fname in enumerate(field_names):
            if fname in ref_recv:
                ok = compare_arrays(py_recon_fields[fi], ref_recv[fname],
                                    f"Python recon field{fi}({fname}) vs ref received", tol)
                all_ok = all_ok and ok
                py_vs_ref_ok = py_vs_ref_ok and ok
    else:
        print(f"  ⚠️  Reference 'received' snapshot or Python output not available.")
        py_vs_ref_ok = None

    print(f"\n  --- reconstructed_output: C++ vs reference 'received' ---")
    cpp_vs_ref_ok = True
    if ref_recv:
        for fi, fname in enumerate(field_names):
            cpp_path = find_dump(dump_dir, infer_dump_idx, 'reconstructed_output', fi, rank=args.rank)
            if cpp_path is not None:
                cpp_recon = load_bin(cpp_path)
                if fname in ref_recv:
                    ok = compare_arrays(cpp_recon, ref_recv[fname],
                                        f"C++ recon field{fi}({fname}) vs ref received", tol)
                    all_ok = all_ok and ok
                    cpp_vs_ref_ok = cpp_vs_ref_ok and ok
            else:
                print(f"  ⚠️  step{infer_dump_idx:04d}_reconstructed_output_field{fi}.bin not found.")
                cpp_vs_ref_ok = False
    else:
        print(f"  ⚠️  Reference 'received' snapshot not available.")
        cpp_vs_ref_ok = None

    # ---- 5. Decision tree ---------------------------------------------------
    print("\n" + "=" * 70)
    print("DECISION TREE CONCLUSION")
    print("=" * 70)
    print(f"  Python input == C++ assembled?       {_fmt_bool(input_match)}")
    print(f"  Python output == C++ raw_output?     {_fmt_bool(output_match)}")
    print(f"  Python recon == reference received?  {_fmt_bool(py_vs_ref_ok)}")
    print(f"  C++ recon == reference received?     {_fmt_bool(cpp_vs_ref_ok)}")
    print()

    if input_match is False:
        print(RED + "→ INPUT ASSEMBLY DIVERGES despite algorithmic match." + RESET)
        print("  Likely cause: subtle indexing/offset bug.")
        print("  Action: examine cube_layout.txt carefully, check n_ghost_layers,")
        print("          compare xs_/ys_/zs_ starts between Python and C++.")
    elif input_match is True and output_match is False and py_vs_ref_ok is True:
        print(RED + "→ PROVIDER/MODEL EXECUTION DIVERGES." + RESET)
        print("  Input assembly is correct; Python model matches reference.")
        print("  Likely cause: AIxeleratorService or TorchScript provider version.")
        print("  Action: investigate provider code changes between old and new CMI.")
    elif input_match is True and output_match is True and py_vs_ref_ok is False:
        print(YELLOW + "→ REFERENCE MISMATCH — algorithm may differ from old pipeline." + RESET)
        print("  C++ and Python are consistent with each other but not with reference.")
        print("  Action: re-examine reference provenance; old CMI may have had a")
        print("          different normalization or cube ordering.")
    elif input_match is True and output_match is True and py_vs_ref_ok is True:
        print(GREEN + "→ NEW CODE IS CORRECT." + RESET)
        print("  All stages match. The step-15 'received' difference previously")
        print("  observed is consistent with the expected model output.")
        print("  Next: investigate the HDF-safe scheduling bug (Step 5 in DEBUG_PLAN.md).")
    else:
        print(YELLOW + "→ INCOMPLETE DATA — some comparisons could not be made." + RESET)
        print("  Run the debug build (Steps 2+3) to generate all .bin dumps, then")
        print("  re-run this script.")

    print()
    return 0 if all_ok else 1


def _fmt_bool(v) -> str:
    if v is True:
        return GREEN + '✅  Yes' + RESET
    if v is False:
        return RED + '❌  No' + RESET
    return YELLOW + '⚠️  N/A (missing data)' + RESET


if __name__ == '__main__':
    sys.exit(main())
