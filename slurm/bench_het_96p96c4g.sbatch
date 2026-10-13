#!/usr/bin/env bash
# Surrogate heterogeneous cell: 96 (CPU-side) + 96 (GPU-side) solver ranks / 4 GPUs.
# Both het components are c23g nodes; the CPU-side component hides its GPUs.
#SBATCH --job-name=cmi-bench-het96p96c4g
#SBATCH --account=rwth0792
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/cmi-bench-het96p96c4g_%j.out
#SBATCH --error=logs/cmi-bench-het96p96c4g_%j.err

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
export HET_CPU_TASKS="${HET_CPU_TASKS:-96}"
export HET_GPU_TASKS="${HET_GPU_TASKS:-96}"
export HET_GPUS="${HET_GPUS:-4}"
exec bash "${SLURM_SUBMIT_DIR}/slurm/cmi_bench_het_engine.sh"
