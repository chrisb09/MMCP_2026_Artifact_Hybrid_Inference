#!/usr/bin/env bash
# Production submitter for 24 SmartSim aggregation reruns:
#   2 models × 4 placements × 3 repetitions
#
# Aggregation parameters are taken from the environment (required):
#   SMARTSIM_BATCH_SIZE          – RedisAI batch_size (must be > 0)
#   SMARTSIM_MIN_BATCH_SIZE      – RedisAI min_batch_size (must be <= batch_size)
#   SMARTSIM_MIN_BATCH_TIMEOUT_MS – RedisAI min_batch_timeout in ms (must be > 0)
#
# Usage:
#   SMARTSIM_BATCH_SIZE=7000 SMARTSIM_MIN_BATCH_SIZE=7000 \
#   SMARTSIM_MIN_BATCH_TIMEOUT_MS=1 slurm/submit_smartsim_aggregation_reruns.sh [--dry-run|--submit]
#   Default: --dry-run
#
# Override models/placements/reps via env:
#   MODELS='transformer' PLACEMENTS='24c1g' REPS=1 slurm/submit_smartsim_aggregation_reruns.sh --submit
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
dry_run=1
for arg in "$@"; do
    case "${arg}" in
        --dry-run) dry_run=1 ;;
        --submit)  dry_run=0 ;;
        *) printf 'Unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
    esac
done

# ── Validate required aggregation parameters ──────────────────────────────────
: "${SMARTSIM_BATCH_SIZE:?SMARTSIM_BATCH_SIZE is required and must be a positive integer}"
: "${SMARTSIM_MIN_BATCH_SIZE:?SMARTSIM_MIN_BATCH_SIZE is required and must be a non-negative integer}"
: "${SMARTSIM_MIN_BATCH_TIMEOUT_MS:?SMARTSIM_MIN_BATCH_TIMEOUT_MS is required and must be a positive integer}"

for _var in "${SMARTSIM_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_TIMEOUT_MS}"; do
    [[ "${_var}" =~ ^[0-9]+$ ]] || { printf 'submit_smartsim_aggregation_reruns: non-negative integer required, got: %s\n' "${_var}" >&2; exit 2; }
done
unset _var

(( SMARTSIM_BATCH_SIZE > 0 )) || { printf 'submit_smartsim_aggregation_reruns: SMARTSIM_BATCH_SIZE must be > 0 for production reruns.\n' >&2; exit 2; }
(( SMARTSIM_MIN_BATCH_SIZE <= SMARTSIM_BATCH_SIZE )) || \
    { printf 'submit_smartsim_aggregation_reruns: SMARTSIM_MIN_BATCH_SIZE (%s) must be <= SMARTSIM_BATCH_SIZE (%s).\n' \
        "${SMARTSIM_MIN_BATCH_SIZE}" "${SMARTSIM_BATCH_SIZE}" >&2; exit 2; }
(( SMARTSIM_MIN_BATCH_TIMEOUT_MS > 0 )) || \
    { printf 'submit_smartsim_aggregation_reruns: SMARTSIM_MIN_BATCH_TIMEOUT_MS must be > 0.\n' >&2; exit 2; }

# ── Knobs ─────────────────────────────────────────────────────────────────────
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
REPS="${REPS:-3}"
STEPS="${STEPS:-200}"
SOLUTION_INTERVAL="${SOLUTION_INTERVAL:-10000}"

# Validate models and placements
for _model in ${MODELS}; do
    [[ "${_model}" == transformer || "${_model}" == mlp ]] || { printf 'Unknown model: %s\n' "${_model}" >&2; exit 2; }
done
for _placement in ${PLACEMENTS}; do
    [[ "${_placement}" == 24c1g || "${_placement}" == 24p24c1g || \
       "${_placement}" == 96c4g || "${_placement}" == 96p96c4g ]] || \
        { printf 'Unknown placement: %s\n' "${_placement}" >&2; exit 2; }
done
[[ "${REPS}" =~ ^[1-9][0-9]*$ ]] || { printf 'REPS must be a positive integer.\n' >&2; exit 2; }
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
campaign_root="${SMARTSIM_AGG_RERUNS_ROOT:-${root}/results/smartsim_aggregation_reruns}"
mkdir -p "${campaign_root}"
campaign="$(mktemp -d "${campaign_root}/campaign_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
manifest="${campaign}/manifest.csv"
scratch_root="${SMARTSIM_AGG_SCRATCH:-${root}/scratch/smartsim_aggregation_reruns}"
# Scratch dir only needed when actually launching jobs.
[[ "${dry_run}" == 1 ]] || mkdir -p "${scratch_root}"

# Capture git revision (best effort)
revision="$(git -C "${root}" rev-parse HEAD 2>/dev/null || printf 'unknown')"

# Manifest header
printf 'cell,jobid,placement,model,rep,batch_size,min_batch_size,timeout_ms,revision,dry_run\n' > "${manifest}"

printf 'Campaign: %s\n' "${campaign}"
printf 'Manifest: %s\n' "${manifest}"
printf 'Aggregation: batch_size=%s min_batch_size=%s timeout_ms=%s\n' \
    "${SMARTSIM_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_TIMEOUT_MS}"
[[ "${dry_run}" == 1 ]] && printf '(dry-run; pass --submit to actually submit)\n'

# ── Submit a single-node cell ─────────────────────────────────────────────────
submit_single() {
    local placement="$1" model="$2" rep="$3" cell="$4"
    local ntasks gres numgpus
    ntasks="$(placement_ntasks "${placement}")"
    gres="$(placement_gpus "${placement}")"
    numgpus="${gres}"
    # Clean environment: export only the required knobs, do not inherit debug settings.
    local exports
    exports="ALL,PROJECT_DIR=${root},CELL_NAME=${cell},REP=${rep},PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=${SOLUTION_INTERVAL},ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,NP_SOLVER=${ntasks},MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=1,SMARTSIM_BATCH_SIZE=${SMARTSIM_BATCH_SIZE},SMARTSIM_MIN_BATCH_SIZE=${SMARTSIM_MIN_BATCH_SIZE},SMARTSIM_MIN_BATCH_TIMEOUT_MS=${SMARTSIM_MIN_BATCH_TIMEOUT_MS},SMOKE_SCRATCH_ROOT=${scratch_root}"
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
        jid=DRY_RUN
    else
        jid="$("${cmd[@]}")"
        jid="${jid%%;*}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || { printf 'sbatch returned unexpected output: %s\n' "${jid}" >&2; exit 2; }
        printf 'Submitted %s -> %s\n' "${cell}" "${jid}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cell}" "${jid}" "${placement}" "${model}" "${rep}" \
        "${SMARTSIM_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_TIMEOUT_MS}" \
        "${revision}" "${dry_run}" >> "${manifest}"
}

# ── Submit a het cell ─────────────────────────────────────────────────────────
submit_het() {
    local placement="$1" model="$2" rep="$3" cell="$4"
    local script numgpus
    script="$(placement_het_script "${placement}")"
    numgpus="$(placement_gpus "${placement}")"
    local exports
    exports="ALL,PROJECT_DIR=${root},CELL_NAME=${cell},REP=${rep},PILOT_MODEL_FILE=$(model_file "${model}"),ML_CUBE_OVERLAP=$(model_overlap "${model}"),RUN_STEPS=${STEPS},SOLUTION_INTERVAL=${SOLUTION_INTERVAL},ML_STEP_TIMING_PRINT=1,DEBUG_EXPORT=0,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,MAIA_SMARTSIM_NUM_GPUS=${numgpus},SMARTSIM_DB_NODES=1,SMARTSIM_BATCH_SIZE=${SMARTSIM_BATCH_SIZE},SMARTSIM_MIN_BATCH_SIZE=${SMARTSIM_MIN_BATCH_SIZE},SMARTSIM_MIN_BATCH_TIMEOUT_MS=${SMARTSIM_MIN_BATCH_TIMEOUT_MS},SMOKE_SCRATCH_ROOT=${scratch_root}"
    local cmd=(sbatch --parsable \
        --job-name="cmi-${cell}" \
        --output="${root}/logs/cmi-${cell}_%j.out" \
        --error="${root}/logs/cmi-${cell}_%j.err" \
        --export="${exports}" \
        "${root}/slurm/${script}")
    if [[ "${dry_run}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
        jid=DRY_RUN
    else
        jid="$("${cmd[@]}")"
        jid="${jid%%;*}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || { printf 'sbatch returned unexpected output: %s\n' "${jid}" >&2; exit 2; }
        printf 'Submitted %s -> %s\n' "${cell}" "${jid}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${cell}" "${jid}" "${placement}" "${model}" "${rep}" \
        "${SMARTSIM_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_SIZE}" "${SMARTSIM_MIN_BATCH_TIMEOUT_MS}" \
        "${revision}" "${dry_run}" >> "${manifest}"
}

# ── Main submission loop ──────────────────────────────────────────────────────
for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for rep in $(seq 1 "${REPS}"); do
            cell="smartsim-${placement}-agg-${model}-r${rep}"
            if placement_is_het "${placement}"; then
                submit_het  "${placement}" "${model}" "${rep}" "${cell}"
            else
                submit_single "${placement}" "${model}" "${rep}" "${cell}"
            fi
        done
    done
done

printf 'Campaign manifest: %s\n' "${manifest}"
