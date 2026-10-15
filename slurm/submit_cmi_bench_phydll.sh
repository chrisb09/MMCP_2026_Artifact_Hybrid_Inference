#!/usr/bin/env bash
# Submit PhyDLL (native C++ DL client) benchmark cells:
#   models x placements x wait modes x repeats
#
# Wait modes: default (receive-blocking) and readiness (100 us poll).
# Placements: 24c1g, 96c4g (single node), 24p24c1g, 96p96c4g (surrogate het).
# One DL client per GPU; het launches via a single srun MPMD with a rank
# dispatch wrapper on the GPU-side component.
#
# Override via env, e.g.:
#   MODELS=mlp PLACEMENTS='24c1g 24p24c1g' WAITS=default slurm/submit_cmi_bench_phydll.sh
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results

REPS="${REPS:-1}"
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
WAITS="${WAITS:-default readiness}"
STEPS="${STEPS:-200}"
jobids="results/cmi_bench_phydll_jobids.txt"
: > "${jobids}"

model_file()   { case "$1" in transformer) echo "./input/transformer_inference_scripted_fw2.pt" ;; mlp) echo "./input/mmcp_test_mlp_m5_cpu.pt" ;; esac; }
model_overlap(){ case "$1" in transformer) echo 0 ;; mlp) echo 4 ;; esac; }
# Per-DL-client forward sub-batch cap: keep the transformer below 2^16, let the
# tiny MLP run one large batch.
model_phydll_batch() { case "$1" in transformer) echo 2048 ;; mlp) echo 1000000 ;; esac; }

record() { printf '%-52s %s\n' "$1" "$2"; printf '%s,%s\n' "$1" "$2" >> "${jobids}"; }

submit_single() {
    local placement="$1" model="$2" wait="$3" rep="$4"
    local gres ntasks npdl case_name
    case "${placement}" in
        24c1g) gres=1; ntasks=24; npdl=1 ;;
        96c4g) gres=4; ntasks=96; npdl=4 ;;
        *) return 1 ;;
    esac
    case "${wait}" in default) case_name=phydll-blocking ;; readiness) case_name=phydll-readiness ;; esac
    local cell="phydll-${placement}-${wait}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --partition=c23g --account=rwth0792 --gres="gpu:${gres}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1 --time=00:20:00 \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),MAIA_PHYDLL_BATCHSIZE=$(model_phydll_batch "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,NP_SOLVER=${ntasks},NP_DL=${npdl}" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" "${case_name}")"
    record "${cell}" "${jid}"
}

submit_het() {
    local placement="$1" model="$2" wait="$3" rep="$4"
    local script cpu gpu gpus npdl
    case "${placement}" in
        24p24c1g) script=bench_het_24p24c1g.sbatch; cpu=24; gpu=24; gpus=1; npdl=1 ;;
        96p96c4g) script=bench_het_96p96c4g.sbatch; cpu=96; gpu=96; gpus=4; npdl=4 ;;
        *) return 1 ;;
    esac
    local cell="phydll-${placement}-${wait}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),MAIA_PHYDLL_BATCHSIZE=$(model_phydll_batch "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,NP_DL=${npdl},PHYDLL_WAIT=${wait},CELL_NAME=${cell}" \
        "${root}/slurm/${script}")"
    record "${cell}" "${jid}"
}

for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for wait in ${WAITS}; do
            for rep in $(seq 1 "${REPS}"); do
                case "${placement}" in
                    24c1g|96c4g) submit_single "${placement}" "${model}" "${wait}" "${rep}" ;;
                    24p24c1g|96p96c4g) submit_het "${placement}" "${model}" "${wait}" "${rep}" ;;
                esac
            done
        done
    done
done

echo "Job IDs in ${jobids}"
