import numpy as np
import h5py
import torch
import os

cmi_dir = 'scratch/current_aix_cpu_2210454/debug/cmi'
ref_path = '/rwthfs/rz/cluster/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5'
model_path = 'input/transformer_inference_scripted_fw2.pt'

print("=== STEP 15 RECONSTRUCTION & MODEL DIFFERENCE INVESTIGATION ===")

# 1. Load manifest metadata
manifest_file = os.path.join(cmi_dir, 'current_rank_0_inference_1_manifest.txt')
manifest = {}
with open(manifest_file, 'r') as f:
    for line in f:
        if '=' in line:
            k, v = line.strip().split('=', 1)
            manifest[k.strip()] = v.strip()

grid_dims = [int(x) for x in manifest['n_cells'].split(',')]
total_cells = np.prod(grid_dims)
print(f"Grid dimensions (with ghost layers): {grid_dims} -> total cells = {total_cells}")

# Load CMI reconstructed fields (float32)
cmi_u = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_reconstructed_fields_field0.bin'), dtype=np.float32)
cmi_v = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_reconstructed_fields_field1.bin'), dtype=np.float32)
cmi_w = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_reconstructed_fields_field2.bin'), dtype=np.float32)

# Load Legacy Reference step 15 received
with h5py.File(ref_path, 'r') as f_ref:
    ref_grp = None
    for gname in f_ref.keys():
        grp = f_ref[gname]
        if grp.attrs.get('globalTimeStep') == 15 and grp.attrs.get('type') in [b'received', 'received']:
            ref_grp = grp
            break
    
    ref_u = np.array(ref_grp['U']).flatten()
    ref_v = np.array(ref_grp['V']).flatten()
    ref_w = np.array(ref_grp['W']).flatten()

print("\n--- 1. Comparing CMI Reconstructed Output vs Legacy Reference (Rank 0, Step 15 received) ---")
diff_u = np.abs(cmi_u - ref_u)
diff_v = np.abs(cmi_v - ref_v)
diff_w = np.abs(cmi_w - ref_w)

print(f"U diff vs Ref: max = {np.max(diff_u):.6e}, mean = {np.mean(diff_u):.6e}, rel_L2 = {np.linalg.norm(diff_u)/np.linalg.norm(ref_u):.6e}")
print(f"V diff vs Ref: max = {np.max(diff_v):.6e}, mean = {np.mean(diff_v):.6e}, rel_L2 = {np.linalg.norm(diff_v)/np.linalg.norm(ref_v):.6e}")
print(f"W diff vs Ref: max = {np.max(diff_w):.6e}, mean = {np.mean(diff_w):.6e}, rel_L2 = {np.linalg.norm(diff_w)/np.linalg.norm(ref_w):.6e}")

top_u_idx = np.argsort(diff_u)[-5:][::-1]
print("\nTop 5 U differences vs Ref (index, CMI_val, Ref_val, AbsDiff):")
for idx in top_u_idx:
    print(f"  cell index {idx:6d}: CMI={cmi_u[idx]:.8f}, Ref={ref_u[idx]:.8f}, Diff={diff_u[idx]:.6e}")

# 2. Test Model Execution: PyTorch Python vs C++ LibTorch output
print("\n--- 2. Testing TorchScript Model Execution (Python PyTorch vs C++ LibTorch output) ---")
assembled_input = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_assembled_input.bin'), dtype=np.float32)

num_cubes = int(manifest['num_cubes']) * 3
seq_len = int(manifest['input_sequence_length'])
cube_size = int(manifest['cube_size'])

assembled_tensor = torch.from_numpy(assembled_input.reshape(num_cubes, seq_len, cube_size))

model = torch.jit.load(model_path)
model.eval()

with torch.no_grad():
    python_output = model(assembled_tensor).numpy().flatten()

cpp_output = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_raw_provider_output.bin'), dtype=np.float32)

model_diff = np.abs(python_output - cpp_output)
print(f"Python PyTorch model output vs C++ LibTorch model output:")
print(f"  Max abs diff: {np.max(model_diff):.6e}")
print(f"  Mean abs diff: {np.mean(model_diff):.6e}")

# 3. Test Exact C++ Reconstruction Logic in Python
print("\n--- 3. Testing Exact C++ Reconstruction Algorithm in Python ---")
denorm_output = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_denormalized_output.bin'), dtype=np.float32)
weights = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_cube_weights.bin'), dtype=np.float64)

vol_indices_raw = np.fromfile(os.path.join(cmi_dir, 'current_rank_0_inference_1_cube_volume_indices.bin'), dtype=np.int32)
# vol_indices is shape (num_cubes_per_field, cube_size)
n_cubes_per_field = int(manifest['num_cubes'])
forecast_window = int(manifest['forecast_window'])
vol_indices = vol_indices_raw.reshape(n_cubes_per_field, cube_size)

python_recon_fields = np.zeros((3, total_cells), dtype=np.float32)

for field in range(3):
    field_sum = np.zeros(total_cells, dtype=np.float32)
    for c in range(n_cubes_per_field):
        batch_index = field * n_cubes_per_field + c
        src_offset = (batch_index * forecast_window + (forecast_window - 1)) * cube_size
        mapping = vol_indices[c]
        cube_vals = denorm_output[src_offset : src_offset + cube_size]
        field_sum[mapping] += cube_vals
    
    nz = weights > 0.0
    field_sum[nz] = (field_sum[nz] / weights[nz].astype(np.float32)).astype(np.float32)
    python_recon_fields[field] = field_sum

print("Python-reconstructed U vs CMI C++ reconstructed U:")
diff_py_cmi = np.abs(python_recon_fields[0] - cmi_u)
print(f"  Max abs diff: {np.max(diff_py_cmi):.6e}")

print("Python-reconstructed U vs Legacy Ref U:")
diff_py_ref = np.abs(python_recon_fields[0] - ref_u)
print(f"  Max abs diff: {np.max(diff_py_ref):.6e}")
