#!/usr/bin/env bash
# Surrogate heterogeneous cell: 24 (CPU-side) + 24 (GPU-side) solver ranks / 1 GPU.
# Both het components are c23g nodes; the CPU-side component hides its GPUs.
#SBATCH --job-name=cmi-bench-het24p24c1g
#SBATCH --account=rwth0792
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/cmi-bench-het24p24c1g_%j.out
#SBATCH --error=logs/cmi-bench-het24p24c1g_%j.err

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

set -euo pipefail
export HET_CPU_TASKS="${HET_CPU_TASKS:-24}"
export HET_GPU_TASKS="${HET_GPU_TASKS:-24}"
export HET_GPUS="${HET_GPUS:-1}"
exec bash "${SLURM_SUBMIT_DIR}/slurm/cmi_bench_het_engine.sh"
