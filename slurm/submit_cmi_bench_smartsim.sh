#!/usr/bin/env bash
# Submit SmartSim/RedisAI benchmark cells:
#   models x placements x db modes x repeats
#
# DB modes: shared (one DB for all solver ranks). Per-GPU DB is a follow-up.
# Placements: 24c1g, 96c4g (single node), 24p24c1g, 96p96c4g (surrogate het;
# the DB is launched on the GPU-side het group via SmartSim's Slurm launcher).
#
# Override via env, e.g.:
#   MODELS=mlp PLACEMENTS='24c1g 24p24c1g' slurm/submit_cmi_bench_smartsim.sh
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results

REPS="${REPS:-1}"
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
DBS="${DBS:-shared}"
STEPS="${STEPS:-200}"
jobids="results/cmi_bench_smartsim_jobids.txt"
: > "${jobids}"

model_file()   { case "$1" in transformer) echo "./input/transformer_inference_scripted_fw2.pt" ;; mlp) echo "./input/mmcp_test_mlp_m5_cpu.pt" ;; esac; }
model_overlap(){ case "$1" in transformer) echo 0 ;; mlp) echo 4 ;; esac; }
db_nodes()     { case "$1" in shared) echo 1 ;; *) echo 1 ;; esac; }

record() { printf '%-52s %s\n' "$1" "$2"; printf '%s,%s\n' "$1" "$2" >> "${jobids}"; }

submit_single() {
    local placement="$1" model="$2" db="$3" rep="$4"
    local gres ntasks numgpus
    case "${placement}" in
        24c1g) gres=1; ntasks=24; numgpus=1 ;;
        96c4g) gres=4; ntasks=96; numgpus=4 ;;
        *) return 1 ;;
    esac
    local cell="smartsim-${placement}-${db}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --partition=c23g --account=rwth0792 --gres="gpu:${gres}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1 --time=00:20:00 \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,NP_SOLVER=${ntasks},MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=$(db_nodes "${db}")" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" smartsim)"
    record "${cell}" "${jid}"
}

submit_het() {
    local placement="$1" model="$2" db="$3" rep="$4"
    local script cpu gpu gpus numgpus
    case "${placement}" in
        24p24c1g) script=bench_het_24p24c1g.sbatch; cpu=24; gpu=24; gpus=1; numgpus=1 ;;
        96p96c4g) script=bench_het_96p96c4g.sbatch; cpu=96; gpu=96; gpus=4; numgpus=4 ;;
        *) return 1 ;;
    esac
    local cell="smartsim-${placement}-${db}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=$(db_nodes "${db}"),CELL_NAME=${cell}" \
        "${root}/slurm/${script}")"
    record "${cell}" "${jid}"
}

for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for db in ${DBS}; do
            for rep in $(seq 1 "${REPS}"); do
                case "${placement}" in
                    24c1g|96c4g) submit_single "${placement}" "${model}" "${db}" "${rep}" ;;
                    24p24c1g|96p96c4g) submit_het "${placement}" "${model}" "${db}" "${rep}" ;;
                esac
            done
        done
    done
done

echo "Job IDs in ${jobids}"
