#!/usr/bin/env bash
# Submit AIX coupling benchmark cells:
#   models x placements x layouts x comm modes x repeats
#
# Placements:
#   24c1g     single c23g node, 24 ranks / 1 GPU
#   96c4g     single c23g node, 96 ranks / 4 GPUs
#   24p24c1g  surrogate het, 24 (CPU-side) + 24 (GPU-side) / 1 GPU
#   96p96c4g  surrogate het, 96 (CPU-side) + 96 (GPU-side) / 4 GPUs
#
# Layouts map to AIX prebuilts swapped at runtime via AIX_LAYOUT_LIB:
#   default -> INSTALL (round-robin), affine -> INSTALL-AFFINE, numa -> INSTALL-NUMA
#
# Override via env to run a subset, e.g.:
#   MODELS=transformer PLACEMENTS='24c1g 96c4g' LAYOUTS=default COMMS=coll \
#     slurm/submit_cmi_bench_aix.sh
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results

REPS="${REPS:-1}"
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
LAYOUTS="${LAYOUTS:-default affine numa}"
COMMS="${COMMS:-coll p2p1 p2pfull}"
STEPS="${STEPS:-200}"
jobids="results/cmi_bench_aix_jobids.txt"
: > "${jobids}"

model_file()    { case "$1" in transformer) echo "./input/transformer_inference_scripted_fw2.pt" ;; mlp) echo "./input/mmcp_test_mlp_m5_cpu.pt" ;; esac; }
model_overlap() { case "$1" in transformer) echo 0 ;; mlp) echo 4 ;; esac; }
model_batch()   { case "$1" in transformer) echo 50000 ;; mlp) echo 1000000 ;; esac; }
layout_lib()    { case "$1" in default) echo "" ;; affine) echo "${root}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-AFFINE" ;; numa) echo "${root}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-NUMA" ;; esac; }
comm_case()     { case "$1" in coll) echo aix-collective ;; p2p1) echo aix-p2p1 ;; p2pfull) echo aix-p2pfull ;; esac; }
comm_aix_comm() { case "$1" in coll) echo collective ;; *) echo pipelined ;; esac; }

record() { printf '%-52s %s\n' "$1" "$2"; printf '%s,%s\n' "$1" "$2" >> "${jobids}"; }

submit_single() {
    local placement="$1" model="$2" layout="$3" comm="$4" rep="$5"
    local gres ntasks
    case "${placement}" in 24c1g) gres=1; ntasks=24 ;; 96c4g) gres=4; ntasks=96 ;; *) return 1 ;; esac
    local cell="aix-${placement}-${layout}-${comm}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --partition=c23g --account=rwth0792 --gres="gpu:${gres}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1 --time=01:00:00 \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),MAIA_AIX_BATCHSIZE=$(model_batch "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,AIX_LAYOUT_LIB=$(layout_lib "${layout}"),NP_SOLVER=${ntasks}" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" "$(comm_case "${comm}")")"
    record "${cell}" "${jid}"
}

submit_het() {
    local placement="$1" model="$2" layout="$3" comm="$4" rep="$5"
    local script cpu gpu gpus
    case "${placement}" in
        24p24c1g) script=bench_het_24p24c1g.sbatch; cpu=24; gpu=24; gpus=1 ;;
        96p96c4g) script=bench_het_96p96c4g.sbatch; cpu=96; gpu=96; gpus=4 ;;
        *) return 1 ;;
    esac
    local credits=""
    [[ "${comm}" == p2pfull ]] && credits=$((cpu + gpu))
    local cell="aix-${placement}-${layout}-${comm}-${model}-r${rep}"
    local jid
    jid="$(sbatch --parsable \
        --export="ALL,PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),MAIA_AIX_BATCHSIZE=$(model_batch "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,AIX_COMM=$(comm_aix_comm "${comm}"),AIX_CREDITS=${credits},AIX_LAYOUT_LIB=$(layout_lib "${layout}"),CELL_NAME=${cell}" \
        "${root}/slurm/${script}")"
    record "${cell}" "${jid}"
}

for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for layout in ${LAYOUTS}; do
            for comm in ${COMMS}; do
                for rep in $(seq 1 "${REPS}"); do
                    case "${placement}" in
                        24c1g|96c4g) submit_single "${placement}" "${model}" "${layout}" "${comm}" "${rep}" ;;
                        24p24c1g|96p96c4g) submit_het "${placement}" "${model}" "${layout}" "${comm}" "${rep}" ;;
                    esac
                done
            done
        done
    done
done

echo "Job IDs in ${jobids}"
