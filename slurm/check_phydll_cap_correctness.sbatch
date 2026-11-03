#!/usr/bin/env bash
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=phydll-cap-check
#SBATCH --output=logs/phydll_cap_check_%j.out
#SBATCH --error=logs/phydll_cap_check_%j.err
set -euo pipefail
project="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run="${PHYDLL_CHECK_RUN_DIR:-${project}/scratch/phydll_cap50000_pilots/pilot_20261027T060511Z_GRijgH/scratch/phydll-blocking_gpu_5251529_WEA4nn}"
source "/hpcwork/$(whoami)/smartsim/python/smartsim_cuda-12/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
cd "$project"
mkdir -p logs
pids=()
for block in 0 1 2 3 4 5; do
    first=$(( block * 4 ))
    ranks="${first},$((first + 1)),$((first + 2)),$((first + 3))"
    python scripts/check_phydll_cap_correctness.py \
        --pilot-run-dir "$run" \
        --model input/transformer_inference_scripted_fw2.pt \
        --ranks 24 --rank-list "$ranks" --inferences 2 \
        --atol 1e-4 --rtol 1e-4 --pilot-cap 50000 \
        > "logs/phydll_cap_check_${SLURM_JOB_ID}_${block}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
for block in 0 1 2 3 4 5; do
    printf '\n== block %s ==\n' "$block"
    tail -n 4 "logs/phydll_cap_check_${SLURM_JOB_ID}_${block}.log"
done
exit "$status"
