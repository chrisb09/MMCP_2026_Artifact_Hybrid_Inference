#!/usr/bin/env bash
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
set -euo pipefail
: "${CMI_PERGPU_MANIFEST:?Use the dedicated per-GPU submitter}"
exec bash "${PROJECT_DIR}/slurm/run_cmi_smoke_gpu.sh" smartsim
