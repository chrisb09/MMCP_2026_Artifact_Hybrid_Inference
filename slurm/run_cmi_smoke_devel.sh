#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=00:20:00
#SBATCH --job-name=cmi-smoke
#SBATCH --output=cmi_smoke_%j.out
#SBATCH --error=cmi_smoke_%j.err

set -euo pipefail

# Submit from the parent workspace, or export PROJECT_DIR explicitly.
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
project_dir="$(realpath "${project_dir}")"
smoke_case="${1:-${CMI_SMOKE_CASE:-phydll-blocking}}"
case "${smoke_case}" in
    phydll-blocking|phydll-readiness) provider=PHYDLL ;;
    aix-collective|aix-sync-off) provider=AIX ;;
    smartsim) provider=SMARTSIM ;;
    *) printf 'Unknown case: %s\n' "${smoke_case}" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run this script inside a Slurm CPU allocation}"
run_steps="${RUN_STEPS:-60}"
timeout_s="${CASE_TIMEOUT_SECONDS:-900}"
ready_s="${READY_TIMEOUT_SECONDS:-120}"
np_dl="${NP_DL:-24}"
for value in "${run_steps}" "${timeout_s}" "${ready_s}" "${np_dl}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { printf 'Expected a positive integer: %s\n' "${value}" >&2; exit 2; }
done
(( run_steps >= 51 )) || { printf 'RUN_STEPS must be >=51 to reach the second inference.\n' >&2; exit 2; }
tasks=24
[[ "${provider}" != PHYDLL ]] || tasks=$((24 + np_dl))
(( ${SLURM_NTASKS:-0} >= tasks )) || { printf 'Need at least %s allocated tasks.\n' "${tasks}" >&2; exit 2; }
maia_build_dir="$(realpath "${MAIA_BUILD_DIR:-${project_dir}/maia/build_gnu_production}")"
maia="${maia_build_dir}/bin/maia"
[[ -x "${maia}" ]] || { printf 'MAIA not executable: %s\n' "${maia}" >&2; exit 1; }

source "${project_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${maia_build_dir}/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}/stubs:${cuda_lib}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
export CPP_ML_INTERFACE_PROVIDER_ENV="${provider}"
export CPP_ML_INTERFACE_DEVICE=CPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
export MAIA_ML_STEP_TIMING_SYNC="${MAIA_ML_STEP_TIMING_SYNC:-1}"
[[ "${smoke_case}" != aix-sync-off ]] || export MAIA_ML_STEP_TIMING_SYNC=0
# This is the number of DL fields, not the number of MPI DL ranks.
# Auto/uniform transport for five input frames and two output frames uses 5/2 fields.
export PHYDLL_DL_COUNT=2 PHYDLL_DL_FIELD_COUNT=2
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
export SCOREP_ENABLE_TRACING=false SCOREP_ENABLE_PROFILING=false
unset FLOW_DEBUG_DUMP_DIR MAIA_SNAPSHOT_DIR MLCOUPLING_DEBUG_EXPORT
unset MLCOUPLING_DEBUG_ALL_RANKS MLCOUPLING_DEBUG_EXPORT_DIR MLCOUPLING_DEBUG_RANK
unset MLCOUPLING_DEBUG_MAX_INFERENCES DUMP_TENSOR_DIR AIX_P2P_TIMELINE_DIR
unset SSDB AIX_DIAGNOSTICS AIX_DIAGNOSTIC_BARRIERS

scratch_root="${SMOKE_SCRATCH_ROOT:-${project_dir}/scratch/cmi_smokes}"
mkdir -p "${scratch_root}"
scratch_root="$(realpath "${scratch_root}")"
run_dir="$(mktemp -d "${scratch_root}/${smoke_case}_${SLURM_JOB_ID}_XXXXXX")"
cd "${run_dir}"
mkdir out auxdata boxes planes tmp
export TMPDIR="${run_dir}/tmp"
ln -s "${project_dir}/input" input
ln -s "${project_dir}/input/grid_les_medium.hdf5" grid_les_medium.hdf5
# A private restart copy also prevents accidental writes through a shared symlink.
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
# Keep modelPath from the input properties: MAIA overrides library.model_file/path.
model_path="$(python3 -c 'import tomllib; print(tomllib.load(open("properties.toml", "rb"))["modelPath"])')"
[[ -r "${model_path}" ]] || { printf 'Input model not readable: %s\n' "${model_path}" >&2; exit 1; }
config_name="config_$(printf '%s' "${provider}" | tr '[:upper:]' '[:lower:]').toml"
cp "${project_dir}/${config_name}" "${config_name}"
if [[ "${provider}" == PHYDLL ]]; then
    # Omission tests the new blocking default, rather than explicitly setting false.
    sed -i '/^[[:space:]]*solver_readiness_wait[[:space:]]*=/d' "${config_name}"
    if [[ "${smoke_case}" == phydll-readiness ]]; then
        sed -i '/^\[library\]/a solver_readiness_wait = true' "${config_name}"
    fi
elif [[ "${provider}" == AIX ]]; then
    sed -i '/^[[:space:]]*communication_mode[[:space:]]*=/d' "${config_name}"
    sed -i '/^\[library\]/a communication_mode = "collective"' "${config_name}"
fi
if [[ "${DEBUG_EXPORT:-0}" == 1 ]]; then
    mkdir debug_dumps
    export MLCOUPLING_DEBUG_EXPORT=1 MLCOUPLING_DEBUG_ALL_RANKS=1
    export MLCOUPLING_DEBUG_EXPORT_DIR="${run_dir}/debug_dumps"
    export MLCOUPLING_DEBUG_MAX_INFERENCES=2
fi
printf 'CASE=%s\nRUN_DIR=%s\nMAIA=%s\nMODEL=%s\nRUN_STEPS=%s\nSOLVER_RANKS=24\n' \
    "${smoke_case}" "${run_dir}" "${maia}" "${model_path}" "${run_steps}" | tee run.info
printf 'AIX_DIAGNOSTIC_BARRIERS=%s\n' "${AIX_DIAGNOSTIC_BARRIERS:-unset}" >> run.info
printf 'MAIA_ML_STEP_TIMING_SYNC=%s\n' "${MAIA_ML_STEP_TIMING_SYNC}" >> run.info
printf 'DL_RANKS=%s\n' "$((tasks - 24))" >> run.info

controller_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${controller_pid}" ]]; then
        touch .solver_done
        # Bound shutdown too: the controller can hang while stopping Redis.
        for ((i=0; i<30; i++)); do
            kill -0 "${controller_pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${controller_pid}" 2>/dev/null; then
            # timeout owns a process group containing the local controller/DB.
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

solver_cmd=(srun --label --mpi=pmix --nodes=1 --ntasks="${tasks}" --ntasks-per-node="${tasks}"
    --cpus-per-task=1 --cpu-bind=cores --kill-on-bad-exit=1)
# Do not halve the step's memory when a 48-CPU allocation runs 24 solver ranks.
solver_cmd+=(--mem="${SLURM_MEM_PER_NODE:?A per-node memory allocation is required}M")
if [[ "${provider}" == PHYDLL ]]; then
    # Match MAIA's world barrier after the terminal PhyDLL coupling signal.
    export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
    dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
    [[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
    [[ -x "${dl_client}" ]] || { printf 'C++ PhyDLL client missing in %s\n' "${maia_build_dir}" >&2; exit 1; }
    # Slurm multi-prog tokenizes whitespace, so reject ambiguous executable paths.
    [[ "${maia}${dl_client}" != *[[:space:]]* ]] || { printf 'Multi-prog executable paths must not contain whitespace.\n' >&2; exit 2; }
    printf '0-23 %s ./properties.toml\n24-%s %s\n' "${maia}" "$((tasks - 1))" "${dl_client}" > phydll.conf
    solver_cmd+=(--multi-prog ./phydll.conf)
elif [[ "${provider}" == SMARTSIM ]]; then
    smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/$(whoami)/smartsim/python/smartsim_cpu/bin/activate}"
    [[ -r "${smart_env}" ]] || { printf 'SmartSim environment missing: %s\n' "${smart_env}" >&2; exit 1; }
    source "${smart_env}"
    export SR_CMD_TIMEOUT=600 SR_SOCKET_TIMEOUT=600000 SR_MODEL_TIMEOUT=600000
    export SMARTSIM_MPI_SEQUENTIAL_PUT=0
    export SR_LOG_LEVEL=INFO SR_LOG_FILE="${run_dir}/smartredis.log"
    timeout --signal=TERM --kill-after=10 "$((timeout_s + ready_s + 60))s" \
        python3 "${project_dir}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
        --launcher local --interface lo --db-nodes 1 --auto-port \
        --intra-op-threads 8 --inter-op-threads 1 --threads-per-queue 8 \
        --cpu-cores-per-node 24 --timeout-s "${ready_s}" \
        --endpoint-file .ssdb_endpoint --done-file .solver_done --exp-dir ./ssdb_exp \
        > controller.log 2>&1 &
    controller_pid=$!
    for ((i=0; i<ready_s; i++)); do
        [[ ! -s .ssdb_endpoint ]] || break
        kill -0 "${controller_pid}" 2>/dev/null || { printf 'SmartSim controller exited; see controller.log.\n' >&2; exit 1; }
        sleep 1
    done
    [[ -s .ssdb_endpoint ]] || { printf 'SmartSim readiness timeout; see controller.log.\n' >&2; exit 1; }
    IFS= read -r SSDB < .ssdb_endpoint
    export SSDB
    solver_cmd+=("${maia}" ./properties.toml)
else
    solver_cmd+=("${maia}" ./properties.toml)
fi
start_seconds=$(date +%s)
set +e
timeout --signal=TERM --kill-after=20 "${timeout_s}s" "${solver_cmd[@]}" > solver.log 2>&1
status=$?
set -e
printf 'BENCHMARK_SOLVER_WALL_SECONDS=%s\nSOLVER_EXIT_STATUS=%s\n' \
    "$(( $(date +%s) - start_seconds ))" "${status}" | tee -a run.info
exit "${status}"
