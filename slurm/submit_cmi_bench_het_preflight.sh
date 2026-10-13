#!/usr/bin/env bash
# Preflight M2: surrogate-heterogeneous launch validation (AIX collective).
#
# Validates that MAIA spans both c23g het components in one MPI world, that the
# CPU-side component hides its GPUs (AIX sees only the GPU-side device count),
# and that the compact schedule still yields 7 inference calls.
#
# Cases:
#   het24p24c1g-transformer-ov0 : 48 solver ranks, 1 GPU
#   het96p96c4g-mlp-ov6         : 192 solver ranks, 4 GPUs (memory-critical)
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results
jobids="results/cmi_bench_het_preflight_jobids.txt"
: > "${jobids}"

submit_one() {
    local case="$1" script="$2" model="$3" overlap="$4" batch="$5"
    local jid
    jid="$(sbatch --parsable \
        --export="ALL,PILOT_MODEL_FILE=${model},ML_CUBE_OVERLAP=${overlap},MAIA_AIX_BATCHSIZE=${batch},RUN_STEPS=200,SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,AIX_COMM=collective,CELL_NAME=${case}" \
        "${root}/slurm/${script}")"
    printf '%-28s overlap=%-2s batch=%-8s job=%s\n' "${case}" "${overlap}" "${batch}" "${jid}"
    printf '%s,%s\n' "${case}" "${jid}" >> "${jobids}"
}

submit_one het24p24c1g-transformer-ov0 bench_het_24p24c1g.sbatch "./input/transformer_inference_scripted_fw2.pt" 0 50000
submit_one het96p96c4g-mlp-ov6         bench_het_96p96c4g.sbatch "./input/mmcp_test_mlp_m5_cpu.pt"                 6 1000000

echo "Job IDs in ${jobids}"
