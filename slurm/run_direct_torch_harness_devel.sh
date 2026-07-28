#!/usr/bin/zsh
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=direct-torch-harness
#SBATCH --output=logs/direct_torch_harness_%J.out
#SBATCH --error=logs/direct_torch_harness_%J.err

set -euxo pipefail
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_folder}"
source setup_env_claix23.sh

base_ld_library_path="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${project_folder}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-SCOREP/lib:${base_ld_library_path}"
model="${project_folder}/input/transformer_inference_scripted_fw2.pt"
input="${project_folder}/scratch/current_aix_cpu_2314057/debug/cmi/current_rank_0_inference_1_normalized_input.bin"
output_dir="${project_folder}/analysis/direct_torch_runs"

"${output_dir}/direct_libtorch_inference" "${model}" "${input}" "${output_dir}/cpp_b1.bin" 1

# Do not let the AIX LibTorch libraries override the Python wheel's libraries.
export LD_LIBRARY_PATH="${base_ld_library_path}"
source "${project_folder}/phydll_py_venv/bin/activate"
python -u analysis/direct_torch_inference.py \
    --model "${model}" --input "${input}" \
    --output "${output_dir}/python_b1_t1.bin" \
    --batch-size 1 --threads 1
