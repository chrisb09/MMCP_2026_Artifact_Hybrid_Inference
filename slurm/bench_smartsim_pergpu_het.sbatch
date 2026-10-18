#!/usr/bin/env bash
# Both components deliberately match the existing surrogate wrapper.
#SBATCH --account=rwth0792
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
set -euo pipefail
: "${CMI_PERGPU_MANIFEST:?Use the dedicated per-GPU submitter}"
export HET_CPU_TASKS=96 HET_GPU_TASKS=96 HET_GPUS=4
exec bash "${PROJECT_DIR}/slurm/cmi_bench_het_engine.sh"
