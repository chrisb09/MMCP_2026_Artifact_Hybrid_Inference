#!/usr/bin/env bash
# Calibration submitter for SmartSim RedisAI aggregation.
#
# For each model × placement combination submits 3 cells:
#   ctrl      – control (batch_size=0, aggregation disabled)
#   t1ms      – aggregation candidate, timeout=1 ms
#   t5ms      – aggregation candidate, timeout=5 ms
#
# Models: transformer mlp
# Placements: 24c1g 24p24c1g 96c4g 96p96c4g
# Total cells: 2 models × 4 placements × 3 variants = 24 cells
#
# Safe aggregation targets (above max single-rank request, below 2^16):
#   transformer: batch_size=7000, min_batch_size=7000
#   mlp:         batch_size=45000, min_batch_size=45000
#
# Usage:
#   slurm/submit_smartsim_calibration.sh [--dry-run | --submit]
#   Default: --dry-run
#
# Override models/placements via env:
#   MODELS='transformer' PLACEMENTS='24c1g' slurm/submit_smartsim_calibration.sh --submit
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
dry_run=1
for arg in "$@"; do
    case "${arg}" in
        --dry-run)  dry_run=1 ;;
        --submit)   dry_run=0 ;;
        *) printf 'Unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
    esac
done

# ── Knobs ─────────────────────────────────────────────────────────────────────
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
STEPS="${STEPS:-200}"
SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"
TIMEOUTS_MS="${TIMEOUTS_MS:-1 5}"   # timeout candidates in ms

# Validate models and placements
for _model in ${MODELS}; do
    [[ "${_model}" == transformer || "${_model}" == mlp ]] || { printf 'Unknown model: %s\n' "${_model}" >&2; exit 2; }
done
for _placement in ${PLACEMENTS}; do
    [[ "${_placement}" == 24c1g || "${_placement}" == 24p24c1g || \
       "${_placement}" == 96c4g || "${_placement}" == 96p96c4g ]] || \
        { printf 'Unknown placement: %s\n' "${_placement}" >&2; exit 2; }
done
unset _model _placement

mkdir -p logs

# ── Per-model helpers ─────────────────────────────────────────────────────────
model_file() {
    case "$1" in
        transformer) printf './input/transformer_inference_scripted_fw2.pt' ;;
        mlp)         printf './input/mmcp_test_mlp_m5_cpu.pt' ;;
    esac
}
model_overlap() {
    case "$1" in transformer) printf '0' ;; mlp) printf '4' ;; esac
}
# Safe aggregation batch sizes (above max per-rank request for the given model)
model_batch_size() {
    case "$1" in transformer) printf '7000' ;; mlp) printf '45000' ;; esac
}

# ── Per-placement helpers ─────────────────────────────────────────────────────
placement_ntasks() {
    case "$1" in 24c1g) printf '24' ;; 24p24c1g) printf '24' ;; 96c4g) printf '96' ;; 96p96c4g) printf '96' ;; esac
}
placement_gpus() {
    case "$1" in 24c1g) printf '1' ;; 24p24c1g) printf '1' ;; 96c4g) printf '4' ;; 96p96c4g) printf '4' ;; esac
}
placement_is_het() {
    case "$1" in 24p24c1g|96p96c4g) return 0 ;; *) return 1 ;; esac
}
placement_het_script() {
    case "$1" in
        24p24c1g) printf 'bench_het_24p24c1g.sbatch' ;;
        96p96c4g) printf 'bench_het_96p96c4g.sbatch' ;;
    esac
}

# ── Campaign directory ────────────────────────────────────────────────────────
calib_root="${SMARTSIM_CALIB_ROOT:-${root}/results/smartsim_calibration}"
mkdir -p "${calib_root}"
campaign="$(mktemp -d "${calib_root}/campaign_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
manifest="${campaign}/manifest.csv"
scratch_root="${SMARTSIM_CALIB_SCRATCH:-${root}/scratch/smartsim_calibration}"
# Scratch dir is only needed when actually launching jobs; skip in dry-run to
# avoid creating artefacts under the project tree during preview passes.
[[ "${dry_run}" == 1 ]] || mkdir -p "${scratch_root}"

# Capture git revision (best effort)
revision="$(git -C "${root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"

# Manifest header
printf 'cell,jobid,placement,model,batch_size,min_batch_size,timeout_ms,revision,dry_run\n' > "${manifest}"

printf 'Campaign: %s\n' "${campaign}"
printf 'Manifest: %s\n' "${manifest}"
[[ "${dry_run}" == 1 ]] && printf '(dry-run; pass --submit to actually submit)\n'

# ── Submit a single-node cell ─────────────────────────────────────────────────
submit_single() {
    local placement="$1" model="$2" cell="$3" bs="$4" min_bs="$5" timeout_ms="$6"
    local ntasks gres numgpus
    ntasks="$(placement_ntasks "${placement}")"
    gres="$(placement_gpus "${placement}")"
    numgpus="${gres}"
    local exports
    exports="ALL,PROJECT_DIR=${root},CELL_NAME=${cell},PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=${SOLUTION_INTERVAL},ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=1,DEBUG_EXPORT_RANK=0,NP_SOLVER=${ntasks},MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=1,SMARTSIM_BATCH_SIZE=${bs},SMARTSIM_MIN_BATCH_SIZE=${min_bs},SMARTSIM_MIN_BATCH_TIMEOUT_MS=${timeout_ms},SMOKE_SCRATCH_ROOT=${scratch_root}"
    local cmd=(sbatch --parsable \
        --partition=c23g --account=rwth0792 --gres="gpu:${gres}" --exclusive --mem=0 \
        --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1 --time=00:20:00 \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" \
        --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="${exports}" \
        "${root}/slurm/run_cmi_smoke_gpu.sh" smartsim)
    if [[ "${dry_run}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
        printf '%s\n' "${cell}"
        jid=DRY_RUN
    else
        jid="$("${cmd[@]}")"
        jid="${jid%%;*}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || { printf 'sbatch returned unexpected output: %s\n' "${jid}" >&2; exit 2; }
        printf 'Submitted %s -> %s\n' "${cell}" "${jid}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cell}" "${jid}" "${placement}" "${model}" \
        "${bs}" "${min_bs}" "${timeout_ms}" "${revision}" "${dry_run}" >> "${manifest}"
}

# ── Submit a het cell ─────────────────────────────────────────────────────────
submit_het() {
    local placement="$1" model="$2" cell="$3" bs="$4" min_bs="$5" timeout_ms="$6"
    local script
    script="$(placement_het_script "${placement}")"
    local numgpus
    numgpus="$(placement_gpus "${placement}")"
    local exports
    exports="ALL,PROJECT_DIR=${root},CELL_NAME=${cell},PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=${SOLUTION_INTERVAL},ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=1,DEBUG_EXPORT_RANK=0,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=1,SMARTSIM_BATCH_SIZE=${bs},SMARTSIM_MIN_BATCH_SIZE=${min_bs},SMARTSIM_MIN_BATCH_TIMEOUT_MS=${timeout_ms},SMOKE_SCRATCH_ROOT=${scratch_root}"
    local cmd=(sbatch --parsable \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" \
        --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="${exports}" \
        "${root}/slurm/${script}")
    if [[ "${dry_run}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
        printf '%s\n' "${cell}"
        jid=DRY_RUN
    else
        jid="$("${cmd[@]}")"
        jid="${jid%%;*}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || { printf 'sbatch returned unexpected output: %s\n' "${jid}" >&2; exit 2; }
        printf 'Submitted %s -> %s\n' "${cell}" "${jid}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cell}" "${jid}" "${placement}" "${model}" \
        "${bs}" "${min_bs}" "${timeout_ms}" "${revision}" "${dry_run}" >> "${manifest}"
}

# ── Main submission loop ──────────────────────────────────────────────────────
for model in ${MODELS}; do
    bs="$(model_batch_size "${model}")"
    min_bs="${bs}"       # min_batch_size == batch_size (saturate as fast as possible)
    for placement in ${PLACEMENTS}; do
        # Cell 1: control (batch_size=0, min_batch_size=0, timeout=0)
        cell="smartsim-${placement}-calib-${model}-ctrl"
        if placement_is_het "${placement}"; then
            submit_het  "${placement}" "${model}" "${cell}" 0 0 0
        else
            submit_single "${placement}" "${model}" "${cell}" 0 0 0
        fi

        # Cells 2 & 3: aggregation candidates with timeout=1ms and timeout=5ms
        for timeout_ms in ${TIMEOUTS_MS}; do
            cell="smartsim-${placement}-calib-${model}-t${timeout_ms}ms"
            if placement_is_het "${placement}"; then
                submit_het  "${placement}" "${model}" "${cell}" "${bs}" "${min_bs}" "${timeout_ms}"
            else
                submit_single "${placement}" "${model}" "${cell}" "${bs}" "${min_bs}" "${timeout_ms}"
            fi
        done
    done
done

printf 'Campaign manifest: %s\n' "${manifest}"
