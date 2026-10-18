#!/usr/bin/env bash
# Dedicated supplement; never writes the shared campaign's ledgers.
set -euo pipefail
root="$(realpath "$(dirname "$0")/..")"
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-96c4g 96p96c4g}"
REPS="${REPS:-3}"
FIRST_REP="${FIRST_REP:-1}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-600}"
[[ "${CASE_TIMEOUT_SECONDS}" == 300 || "${CASE_TIMEOUT_SECONDS}" == 600 ]] || { printf 'CASE_TIMEOUT_SECONDS must be 300 or 600.\n' >&2; exit 2; }
[[ "${REPS}" =~ ^[1-3]$ ]] || { printf 'REPS must be 1, 2 or 3 (repetitions 1..N).\n' >&2; exit 2; }
[[ "${FIRST_REP}" =~ ^[1-3]$ ]] && (( FIRST_REP <= REPS )) || { printf 'FIRST_REP must be between 1 and REPS.\n' >&2; exit 2; }
[[ "${DRY_RUN:-0}" =~ ^[01]$ ]] || exit 2
for model in ${MODELS}; do [[ "${model}" == transformer || "${model}" == mlp ]] || exit 2; done
for placement in ${PLACEMENTS}; do [[ "${placement}" == 96c4g || "${placement}" == 96p96c4g ]] || exit 2; done
campaign_root="${CMI_PERGPU_CAMPAIGN_ROOT:-${root}/results/cmi_smartsim_pergpu}"
mkdir -p "${campaign_root}" "${root}/logs"
campaign="$(mktemp -d "${campaign_root}/campaign_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
mkdir "${campaign}/status"
manifest="${campaign}/manifest.csv"
printf 'cell,jobid,placement,model,rep,tpq,status_dir,revision,dry_run,case_timeout_seconds\n' > "${manifest}"
revision="$(git -C "${root}" rev-parse HEAD)"
sha256sum "${root}/slurm/"{submit_cmi_bench_smartsim_pergpu.sh,bench_smartsim_pergpu_single.sbatch,bench_smartsim_pergpu_het.sbatch,cmi_smartsim_pergpu.sh,run_cmi_smoke_gpu.sh,cmi_bench_het_engine.sh,cmi_node_sampler.sh} \
    "${root}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
    "${root}/scripts/validate_cmi_smartsim_pergpu.py" "${root}/config_smartsim.toml" > "${campaign}/source.sha256"
git -C "${root}" diff --binary > "${campaign}/source.diff"
for model in ${MODELS}; do
    if [[ "${model}" == transformer ]]; then
        mf=./input/transformer_inference_scripted_fw2.pt; ov=0
    else
        mf=./input/mmcp_test_mlp_m5_cpu.pt; ov=4
    fi
    for placement in ${PLACEMENTS}; do
        tpq=1; wrapper=het
        [[ "${placement}" != 96c4g ]] || wrapper=single
        [[ "${model}/${placement}" != transformer/96c4g ]] || tpq=4
        for ((rep=FIRST_REP; rep<=REPS; ++rep)); do
            cell="smartsim-${placement}-pergpu-${model}-r${rep}"
            exports="ALL,PROJECT_DIR=${root},CELL_NAME=${cell},REP=${rep},PILOT_MODEL_FILE=${mf},ML_CUBE_OVERLAP=${ov},SMARTSIM_TPQ=${tpq},SMARTSIM_DB_LAYOUT=per-ml-node,SMARTSIM_DB_NODES=4,SMARTSIM_DB_GPU_COUNT=4,MAIA_SMARTSIM_NUM_GPUS=1,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,NP_SOLVER=96,RUN_STEPS=200,SOLUTION_INTERVAL=10000,ML_STEP_TIMING_PRINT=1,MAIA_ML_WORKLOAD_LOG=1,DEBUG_EXPORT=0,CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS},READY_TIMEOUT_SECONDS=120,SMARTSIM_INTERFACE=ib0,CMI_PERGPU_MANIFEST=${manifest},CMI_PERGPU_STATUS_DIR=${campaign}/status,SMOKE_SCRATCH_ROOT=${root}/scratch/cmi_smartsim_pergpu"
            cmd=(sbatch --parsable --chdir="${root}" --job-name="cmi-${cell}"
                --output="${root}/logs/cmi-${cell}_%j.out" --error="${root}/logs/cmi-${cell}_%j.err"
                --export="${exports}" "${root}/slurm/bench_smartsim_pergpu_${wrapper}.sbatch")
            printf '%q ' "${cmd[@]}" >> "${campaign}/commands.sh"; printf '\n' >> "${campaign}/commands.sh"
            if [[ "${DRY_RUN:-0}" == 1 ]]; then
                jid=DRY_RUN
                printf '%q ' "${cmd[@]}"; printf '\n'
            else
                jid="$("${cmd[@]}")"; jid="${jid%%;*}"
                [[ "${jid}" =~ ^[0-9]+$ ]] || exit 2
            fi
            printf '%s,%s\n' "${cell}" "${jid}" >> "${campaign}/jobids.csv"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${cell}" "${jid}" "${placement}" "${model}" "${rep}" "${tpq}" "${campaign}/status" "${revision}" "${DRY_RUN:-0}" "${CASE_TIMEOUT_SECONDS}" >> "${manifest}"
        done
    done
done
printf 'Campaign manifest: %s\n' "${manifest}"
