#!/usr/bin/env bash
# Diagnostic: per-model forward GPU memory, torch accounting vs nvidia-smi.
#SBATCH --job-name=cmi-diag-mem
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/cmi-diag-mem_%j.out
#SBATCH --error=logs/cmi-diag-mem_%j.err

set -uo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
cd "${root}"
source ./setup_env_claix23.sh
source /hpcwork/ro092286/smartsim/python/smartsim_cuda-12/bin/activate

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

python3 scripts/diag_forward_mem.py input/transformer_inference_scripted_fw2.pt 50000
python3 scripts/diag_forward_mem.py input/transformer_inference_scripted_fw2.pt 65000
python3 scripts/diag_forward_mem.py input/mmcp_test_mlp_m5_cpu.pt 125550
python3 scripts/diag_forward_mem.py input/mmcp_test_mlp_m5_cpu.pt 1000000
