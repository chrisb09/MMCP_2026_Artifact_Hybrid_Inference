#!/usr/bin/zsh
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=direct-torch260-py39
#SBATCH --output=logs/direct_torch260_py39_%J.out
#SBATCH --error=logs/direct_torch260_py39_%J.err

set -euxo pipefail
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_folder}"
source setup_env_claix23.sh

output_dir="${project_folder}/analysis/direct_torch_runs"
venv_dir="${output_dir}/venv_py39_torch260"
py39_bin="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3.9"

if [ ! -d "${venv_dir}" ]; then
    echo "Creating Python 3.9 venv in ${venv_dir}..."
    "${py39_bin}" -m venv "${venv_dir}"
    source "${venv_dir}/bin/activate"
    pip install --no-cache-dir torch==2.6.0 numpy --index-url https://download.pytorch.org/whl/cu124
else
    source "${venv_dir}/bin/activate"
    pip install --no-cache-dir numpy || true
fi

python -c "import torch; print('Loaded PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

echo "=== Running Direct Python 3.9 + PyTorch 2.6.0 Inference (Batch 3456) ==="
python -u analysis/direct_torch_inference.py \
    --model input/transformer_inference_scripted_fw2.pt \
    --input scratch/current_aix_cpu_2314057/debug/cmi/current_rank_0_inference_1_normalized_input.bin \
    --output "${output_dir}/python_260_py39_b3456.bin" \
    --batch-size 3456 --threads 1

echo "=== Comparison Results ==="
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
    'cpp_240_abi1': runs / 'cpp_240_abi1_b3456.bin',
    'python_260_py39_abi0': runs / 'python_260_py39_b3456.bin',
}

data = {k: np.fromfile(p, dtype=np.float32) for k, p in refs.items() if p.exists()}

for k, v in data.items():
    print(f'{k:22s} size={v.size}')

print('\n' + '='*70)
print('COMPARING NEW: python_260_py39_abi0 against all references:')
print('='*70)

target = data['python_260_py39_abi0']
for name, arr in data.items():
    if name == 'python_260_py39_abi0': continue
    diff = np.abs(target.astype(np.float64) - arr.astype(np.float64))
    print(f"python_260_py39_abi0 vs {name:22s}: same={np.array_equal(target, arr)} max={diff.max():.9e} mean={diff.mean():.9e} nonzero={np.count_nonzero(diff)}")

PY
