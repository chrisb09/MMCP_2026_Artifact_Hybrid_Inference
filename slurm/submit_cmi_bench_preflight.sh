#!/usr/bin/env bash
# Preflight M1: compact, output-free schedule.
#
# Validates that a single exclusive four-H100 c23g node with 24 AIX collective
# solver ranks yields exactly 7 inference calls under a compact schedule:
#   solutionInterval=10000, RUN_STEPS=200
#   warmups at global steps 15 and 44
#   measured at 73, 102, 131, 160, 189
#
# Cases:
#   compact-transformer-ov0 : Transformer, overlap 0, AIX batch 50000
#   compact-mlp-ov6         : MLP, overlap 6, AIX batch 1000000
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results
jobids="results/cmi_bench_preflight_jobids.txt"
: > "${jobids}"

steps="${PREFLIGHT_STEPS:-200}"
gpus="${PREFLIGHT_GPUS:-4}"
solver_ranks="${NP_SOLVER:-24}"

submit_one() {
    local case="$1" model="$2" overlap="$3" batch="$4"
    local jid
    jid="$(sbatch --parsable \
        --partition=c23g --account=rwth0792 \
        --gres="gpu:${gpus}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${solver_ranks}" --cpus-per-task=1 \
        --time=01:00:00 \
        --job-name="cmi-bench-${case}" \
        --output="${root}/logs/cmi-bench-${case}_%j.out" \
        --error="${root}/logs/cmi-bench-${case}_%j.err" \
        --export="ALL,PILOT_MODEL_FILE=${model},ML_CUBE_OVERLAP=${overlap},MAIA_AIX_BATCHSIZE=${batch},RUN_STEPS=${steps},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,NP_SOLVER=${solver_ranks}" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" aix-collective)"
    printf '%-24s overlap=%-2s batch=%-8s job=%s\n' "${case}" "${overlap}" "${batch}" "${jid}"
    printf '%s,%s,%s,%s\n' "${case}" "${overlap}" "${batch}" "${jid}" >> "${jobids}"
}

submit_one compact-transformer-ov0 "./input/transformer_inference_scripted_fw2.pt" 0 50000
submit_one compact-mlp-ov6         "./input/mmcp_test_mlp_m5_cpu.pt"                 6 1000000

echo "Job IDs in ${jobids}"
