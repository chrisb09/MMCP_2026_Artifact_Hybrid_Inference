#!/usr/bin/zsh
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=direct-torch240-abi1
#SBATCH --output=logs/direct_torch240_abi1_%J.out
#SBATCH --error=logs/direct_torch240_abi1_%J.err

set -euxo pipefail
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_folder}"
source setup_env_claix23.sh

torch_pkg="${project_folder}/analysis/direct_torch_runs/libtorch_2.4.0_abi1_pkg/libtorch"
torch_include="${torch_pkg}/include"
torch_api_include="${torch_include}/torch/csrc/api/include"
torch_lib="${torch_pkg}/lib"
output_dir="${project_folder}/analysis/direct_torch_runs"

# 1. Compile inspection executable
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"${torch_include}" -I"${torch_api_include}" \
    analysis/inspect_aix_torch.cpp \
    -L"${torch_lib}" -Wl,-rpath,"${torch_lib}" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
    -o "${output_dir}/inspect_torch_240_abi1"

# 2. Compile inference executable
g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"${torch_include}" -I"${torch_api_include}" \
    analysis/direct_libtorch_inference.cpp \
    -L"${torch_lib}" -Wl,-rpath,"${torch_lib}" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
    -o "${output_dir}/direct_libtorch_240_abi1"

export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"

echo "=== Torch 2.4.0 ABI-1 Introspection ==="
"${output_dir}/inspect_torch_240_abi1"

echo "=== Torch 2.4.0 ABI-1 Inference (Batch 3456) ==="
"${output_dir}/direct_libtorch_240_abi1" \
    input/transformer_inference_scripted_fw2.pt \
    scratch/current_aix_cpu_2314057/debug/cmi/current_rank_0_inference_1_normalized_input.bin \
    "${output_dir}/cpp_240_abi1_b3456.bin" 3456

echo "=== Comparison Results ==="
source "${project_folder}/phydll_py_venv/bin/activate"
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path('.')
runs = root / 'analysis/direct_torch_runs'
refs = {
    'aix_260_abi1': root / 'scratch/current_aix_cpu_2314057/debug/cmi/current_rank_0_inference_1_raw_provider_output.bin',
    'smartsim_240_abi0': root / 'debug_dumps/smartsim/cmi_2390071/current_rank_0_inference_1_raw_provider_output.bin',
    'phydll_py_240_abi0': root / 'debug_dumps/phydll_py/cmi_2387199/current_rank_0_inference_1_raw_provider_output.bin',
    'cpp_260_abi1': runs / 'cpp_b3456.bin',
    'python_240_abi0': runs / 'python_b3456_t1.bin',
    'cpp_240_abi0': runs / 'cpp_smartsim_b3456.bin',
    'cpp_240_abi1': runs / 'cpp_240_abi1_b3456.bin',
}

data = {k: np.fromfile(p, dtype=np.float32) for k, p in refs.items() if p.exists()}

for k, v in data.items():
    print(f'{k:20s} size={v.size}')

print('\n' + '='*70)
print('COMPARING NEW: cpp_240_abi1 against all references:')
print('='*70)

target = data['cpp_240_abi1']
for name, arr in data.items():
    if name == 'cpp_240_abi1': continue
    diff = np.abs(target.astype(np.float64) - arr.astype(np.float64))
    print(f"cpp_240_abi1 vs {name:20s}: same={np.array_equal(target, arr)} max={diff.max():.9e} mean={diff.mean():.9e} nonzero={np.count_nonzero(diff)}")

PY
