#!/usr/bin/env bash
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=cmi-gpu-smoke
#SBATCH --output=logs/cmi-gpu-smoke_%j.out
#SBATCH --error=logs/cmi-gpu-smoke_%j.err

set -euo pipefail

# Submit from the artifact repository, or export PROJECT_DIR explicitly.
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
project_dir="$(realpath "${project_dir}")"
smoke_case="${1:-${CMI_SMOKE_CASE:-aix-collective}}"

provider=""
aix_comm="collective"
aix_credits=""
case "${smoke_case}" in
    phydll-blocking|phydll-readiness) provider=PHYDLL ;;
    aix-collective|aix-sync-off) provider=AIX; aix_comm="collective" ;;
    aix-p2p1) provider=AIX; aix_comm="pipelined"; aix_credits=1 ;;
    aix-p2pfull) provider=AIX; aix_comm="pipelined"; aix_credits="${NP_SOLVER:-24}" ;;
    smartsim) provider=SMARTSIM ;;
    *) printf 'Unknown case: %s\n' "${smoke_case}" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?Run this script inside a Slurm GPU allocation}"
run_steps="${RUN_STEPS:-60}"
timeout_s="${CASE_TIMEOUT_SECONDS:-900}"
ready_s="${READY_TIMEOUT_SECONDS:-180}"
np_solver="${NP_SOLVER:-24}"
np_dl="${NP_DL:-1}"
for value in "${run_steps}" "${timeout_s}" "${ready_s}" "${np_solver}" "${np_dl}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { printf 'Expected a positive integer: %s\n' "${value}" >&2; exit 2; }
done
(( run_steps >= 51 )) || { printf 'RUN_STEPS must be >=51 to reach the second inference.\n' >&2; exit 2; }

tasks="${np_solver}"
[[ "${provider}" != PHYDLL ]] || tasks=$((np_solver + np_dl))
maia_build_dir="$(realpath "${MAIA_BUILD_DIR:-${project_dir}/maia/build_gnu_production}")"
maia="${maia_build_dir}/bin/maia"
[[ -x "${maia}" ]] || { printf 'MAIA not executable: %s\n' "${maia}" >&2; exit 1; }

source "${project_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${maia_build_dir}/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}:${LD_LIBRARY_PATH:-}"

# Preserve Slurm's allocation-aware GPU mask: never clear CUDA_VISIBLE_DEVICES here.
export CPP_ML_INTERFACE_PROVIDER_ENV="${provider}"
export CPP_ML_INTERFACE_DEVICE=GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
export MAIA_ML_STEP_TIMING_SYNC="${MAIA_ML_STEP_TIMING_SYNC:-1}"
[[ "${smoke_case}" != aix-sync-off ]] || export MAIA_ML_STEP_TIMING_SYNC=0
# Number of DL fields for auto/uniform transport (five input, two output frames).
export PHYDLL_DL_COUNT=2 PHYDLL_DL_FIELD_COUNT=2
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
export SCOREP_ENABLE_TRACING=false SCOREP_ENABLE_PROFILING=false
unset FLOW_DEBUG_DUMP_DIR MAIA_SNAPSHOT_DIR MLCOUPLING_DEBUG_EXPORT
unset MLCOUPLING_DEBUG_ALL_RANKS MLCOUPLING_DEBUG_EXPORT_DIR MLCOUPLING_DEBUG_RANK
unset MLCOUPLING_DEBUG_MAX_INFERENCES DUMP_TENSOR_DIR AIX_P2P_TIMELINE_DIR
unset SSDB AIX_DIAGNOSTICS AIX_DIAGNOSTIC_BARRIERS AIX_P2P_INITIAL_CREDITS
if [[ -n "${aix_credits}" ]]; then
    export AIX_P2P_INITIAL_CREDITS="${aix_credits}"
fi

scratch_root="${SMOKE_SCRATCH_ROOT:-${project_dir}/scratch/cmi_smokes}"
mkdir -p "${scratch_root}"
scratch_root="$(realpath "${scratch_root}")"
run_dir="$(mktemp -d "${scratch_root}/${smoke_case}_gpu_${SLURM_JOB_ID}_XXXXXX")"
cd "${run_dir}"
mkdir out auxdata boxes planes tmp
export TMPDIR="${run_dir}/tmp"
if [[ "${provider}" == AIX ]]; then
    # Capture CUDA execution evidence; GPU visibility alone does not prove use.
    if [[ "${aix_comm}" == pipelined ]]; then
        export AIX_P2P_TIMELINE_DIR="${run_dir}/aix_gpu_diagnostics"
    else
        export AIX_DIAGNOSTICS=1 AIX_DIAGNOSTICS_DIR="${run_dir}/aix_gpu_diagnostics"
    fi
fi
ln -s "${project_dir}/input" input
ln -s "${project_dir}/input/grid_les_medium.hdf5" grid_les_medium.hdf5
cp --reflink=auto "${project_dir}/input/restart_les_init_medium.hdf5" out/restart_les_ref_medium.hdf5
cp "${project_dir}/input/properties_run_les_ref_medium.toml" properties.toml
sed -i \
    -e "s/^timeSteps *=.*/timeSteps = ${run_steps}/" \
    -e 's/^hostFraction *=.*/hostFraction = "1.00"/' \
    -e 's/^mlInterval *=.*/mlInterval = 5/' \
    -e 's/^mlInputLength *=.*/mlInputLength = 5/' \
    -e 's/^mlStepCoefficient *=.*/mlStepCoefficient = 12/' \
    -e 's/^mlForecastWindow *=.*/mlForecastWindow = 2/' \
    -e 's/^mlScalingFactor *=.*/mlScalingFactor = 1/' \
    -e 's/^mlInputStepDistance *=.*/mlInputStepDistance = 1/' properties.toml
model_path="$(python3 -c 'import tomllib; print(tomllib.load(open("properties.toml", "rb"))["modelPath"])')"
[[ -r "${model_path}" ]] || { printf 'Input model not readable: %s\n' "${model_path}" >&2; exit 1; }

config_name="config_$(printf '%s' "${provider}" | tr '[:upper:]' '[:lower:]').toml"
cp "${project_dir}/${config_name}" "${config_name}"
if [[ "${provider}" == PHYDLL ]]; then
    sed -i '/^[[:space:]]*solver_readiness_wait[[:space:]]*=/d' "${config_name}"
    if [[ "${smoke_case}" == phydll-readiness ]]; then
        sed -i '/^\[library\]/a solver_readiness_wait = true' "${config_name}"
    fi
elif [[ "${provider}" == AIX ]]; then
    sed -i '/^[[:space:]]*communication_mode[[:space:]]*=/d' "${config_name}"
    sed -i "/^\[library\]/a communication_mode = \"${aix_comm}\"" "${config_name}"
fi
if [[ "${DEBUG_EXPORT:-0}" == 1 ]]; then
    mkdir debug_dumps
    export MLCOUPLING_DEBUG_EXPORT=1 MLCOUPLING_DEBUG_ALL_RANKS=1
    export MLCOUPLING_DEBUG_EXPORT_DIR="${run_dir}/debug_dumps"
    export MLCOUPLING_DEBUG_MAX_INFERENCES=2
fi

{
    printf 'CASE=%s\n' "${smoke_case}"
    printf 'RUN_DIR=%s\n' "${run_dir}"
    printf 'MAIA=%s\n' "${maia}"
    printf 'MODEL=%s\n' "${model_path}"
    printf 'RUN_STEPS=%s\n' "${run_steps}"
    printf 'SOLVER_RANKS=%s\n' "${np_solver}"
    printf 'DL_RANKS=%s\n' "$((tasks - np_solver))"
    printf 'AIX_COMM=%s\n' "${aix_comm}"
    printf 'AIX_CREDITS=%s\n' "${aix_credits:-unset}"
    printf 'MAIA_ML_STEP_TIMING_SYNC=%s\n' "${MAIA_ML_STEP_TIMING_SYNC}"
    printf 'SLURM_CPUS_ON_NODE=%s\n' "${SLURM_CPUS_ON_NODE:-unknown}"
} | tee run.info

# Confirm the allocated GPU is actually visible before inference.
nvidia-smi -L > gpu_list.log 2>&1 || true
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" >> run.info

controller_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${controller_pid}" ]]; then
        touch .solver_done
        for ((i = 0; i < 30; ++i)); do
            kill -0 "${controller_pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${controller_pid}" 2>/dev/null; then
            kill -TERM -- "-${controller_pid}" 2>/dev/null || true
            sleep 2
            kill -KILL -- "-${controller_pid}" 2>/dev/null || true
        fi
        wait "${controller_pid}" 2>/dev/null || true
    fi
    printf 'EXIT_STATUS=%s\n' "${status}" >> run.info
    printf 'Preserved smoke artifacts: %s\n' "${run_dir}"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

pin="${project_dir}/slurm/pin_cmi_rank.sh"
# Oversubscribe inside the allocation: more ranks than allocated cores, with
# explicit per-rank affinity instead of Slurm's 1:1 binding.
solver_cmd=(srun --label --mpi=pmix --nodes=1 --ntasks="${tasks}" --overcommit
    --cpu-bind=none --kill-on-bad-exit=1)

if [[ "${provider}" == PHYDLL ]]; then
    # The site's 5-second exit grace can kill the GPU client during teardown.
    # Let all ranks exit normally; the outer case timeout still bounds the step.
    solver_cmd+=(--wait=0)
    export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
    dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
    [[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
    [[ -x "${dl_client}" ]] || { printf 'C++ PhyDLL client missing in %s\n' "${maia_build_dir}" >&2; exit 1; }
    [[ "${maia}${dl_client}" != *[[:space:]]* ]] || { printf 'Multi-prog paths must not contain whitespace.\n' >&2; exit 2; }
    printf '0-%s /bin/bash %s %s ./properties.toml\n%s-%s /bin/bash %s %s\n' \
        "$((np_solver - 1))" "${pin}" "${maia}" \
        "${np_solver}" "$((tasks - 1))" "${pin}" "${dl_client}" > phydll.conf
    solver_cmd+=(--multi-prog ./phydll.conf)
elif [[ "${provider}" == SMARTSIM ]]; then
    smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/$(whoami)/smartsim/python/smartsim_cuda-12/bin/activate}"
    [[ -r "${smart_env}" ]] || { printf 'SmartSim environment missing: %s\n' "${smart_env}" >&2; exit 1; }
    source "${smart_env}"
    export SR_CMD_TIMEOUT=600 SR_SOCKET_TIMEOUT=600000 SR_MODEL_TIMEOUT=600000
    export SMARTSIM_MPI_SEQUENTIAL_PUT=0
    export SR_LOG_LEVEL=INFO SR_LOG_FILE="${run_dir}/smartredis.log"
    timeout --signal=TERM --kill-after=10 "$((timeout_s + ready_s + 60))s" \
        python3 "${project_dir}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
        --launcher local --interface lo --db-nodes 1 --auto-port --use-gpu \
        --intra-op-threads 4 --inter-op-threads 1 --threads-per-queue 4 \
        --cpu-cores-per-node "${np_solver}" --timeout-s "${ready_s}" \
        --endpoint-file .ssdb_endpoint --done-file .solver_done --exp-dir ./ssdb_exp \
        > controller.log 2>&1 &
    controller_pid=$!
    for ((i = 0; i < ready_s; ++i)); do
        [[ ! -s .ssdb_endpoint ]] || break
        kill -0 "${controller_pid}" 2>/dev/null || { printf 'SmartSim controller exited; see controller.log.\n' >&2; exit 1; }
        sleep 1
    done
    [[ -s .ssdb_endpoint ]] || { printf 'SmartSim readiness timeout; see controller.log.\n' >&2; exit 1; }
    IFS= read -r SSDB < .ssdb_endpoint
    export SSDB
    solver_cmd+=(/bin/bash "${pin}" "${maia}" ./properties.toml)
else
    solver_cmd+=(/bin/bash "${pin}" "${maia}" ./properties.toml)
fi

start_seconds=$(date +%s)
set +e
timeout --signal=TERM --kill-after=20 "${timeout_s}s" "${solver_cmd[@]}" > solver.log 2>&1
status=$?
set -e
printf 'BENCHMARK_SOLVER_WALL_SECONDS=%s\nSOLVER_EXIT_STATUS=%s\n' \
    "$(( $(date +%s) - start_seconds ))" "${status}" | tee -a run.info
if [[ "${status}" == 0 && "${provider}" == AIX ]]; then
    python3 "${project_dir}/scripts/check_aix_gpu_evidence.py" \
        "${aix_comm}" "${run_dir}/aix_gpu_diagnostics"
fi
exit "${status}"
