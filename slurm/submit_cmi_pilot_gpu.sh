#!/usr/bin/env bash
# Submit the MAIA [TBL] ML-step pilot on a full c23g GPU node.
#
# Characterizes the ML step for the Transformer and the untrained MLP at
# controlled cube-overlap settings. Uses the validated AIX collective path with
# 24 solver ranks on one exclusive four-H100 node, ML step timing enabled, and
# debug exports disabled. This is a sizing pilot, not the campaign.
#
# Cases:
#   transformer-ov0 : Transformer, overlap 0, AIX batch 50000 (default)
#   mlp-ov0         : MLP, overlap 0, AIX batch 1000000
#   mlp-ov6         : MLP, overlap 6, AIX batch 1000000
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results
jobids="results/cmi_pilot_jobids.txt"
: > "${jobids}"

steps="${PILOT_STEPS:-200}"
gpus="${PILOT_GPUS:-4}"
solver_ranks="${NP_SOLVER:-24}"

submit_one() {
    local case="$1" model="$2" overlap="$3" batch="$4"
    local jid
    jid="$(sbatch --parsable \
        --partition=c23g --account=rwth0792 \
        --gres="gpu:${gpus}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${solver_ranks}" --cpus-per-task=1 \
        --time=01:00:00 \
        --job-name="cmi-pilot-${case}" \
        --output="${root}/logs/cmi-pilot-${case}_%j.out" \
        --error="${root}/logs/cmi-pilot-${case}_%j.err" \
        --export="ALL,PILOT_MODEL_FILE=${model},ML_CUBE_OVERLAP=${overlap},MAIA_AIX_BATCHSIZE=${batch},RUN_STEPS=${steps},ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,NP_SOLVER=${solver_ranks}" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" aix-collective)"
    printf '%-16s overlap=%-2s batch=%-8s job=%s\n' "${case}" "${overlap}" "${batch}" "${jid}"
    printf '%s,%s,%s,%s\n' "${case}" "${overlap}" "${batch}" "${jid}" >> "${jobids}"
}

submit_one transformer-ov0 "./input/transformer_inference_scripted_fw2.pt" 0 50000
submit_one mlp-ov0         "./input/mmcp_test_mlp_m5_cpu.pt"                 0 1000000
submit_one mlp-ov6         "./input/mmcp_test_mlp_m5_cpu.pt"                 6 1000000

echo "Job IDs in ${jobids}"
