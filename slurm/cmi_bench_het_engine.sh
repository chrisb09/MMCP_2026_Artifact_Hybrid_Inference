#!/usr/bin/env bash
# Shared engine for surrogate-heterogeneous MAIA benchmark cells.
#
# Both het components are c23g nodes; the CPU-side component hides its GPUs via
# HET_SURROGATE_HIDE_GPU=1 on the rank wrapper, so device detection sees a
# CPU-only node there. The solver spans both components in one MPI world using a
# single srun MPMD launch (one component per het group).
#
# Required env:
#   HET_CPU_TASKS  solver ranks on the CPU-side component (het-group 0)
#   HET_GPU_TASKS  solver ranks on the GPU-side component (het-group 1)
#   HET_GPUS       GPUs allocated on the GPU-side component
# Benchmark env (defaults preserve a compact AIX collective cell):
#   PILOT_MODEL_FILE, ML_CUBE_OVERLAP, MAIA_AIX_BATCHSIZE, RUN_STEPS,
#   SOLUTION_INTERVAL, ML_STEP_TIMING_PRINT, AIX_COMM, AIX_CREDITS, DEBUG_EXPORT
set -euo pipefail

project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
project_dir="$(realpath "${project_dir}")"
: "${SLURM_HET_SIZE:?Run inside a heterogeneous Slurm allocation}"

provider="${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}"
case "${provider}" in
    AIX|PHYDLL|SMARTSIM) ;;
    *) printf 'cmi_bench_het_engine supports AIX, PHYDLL, SMARTSIM (got %s).\n' "${provider}" >&2; exit 2 ;;
esac
np_dl="${NP_DL:-1}"
phydll_wait="${PHYDLL_WAIT:-default}"
ready_s="${READY_TIMEOUT_SECONDS:-180}"
controller_pid=""

cpu_tasks="${HET_CPU_TASKS:?HET_CPU_TASKS must be set}"
gpu_tasks="${HET_GPU_TASKS:?HET_GPU_TASKS must be set}"
gpus="${HET_GPUS:?HET_GPUS must be set}"
aix_comm="${AIX_COMM:-collective}"
aix_credits="${AIX_CREDITS:-}"
run_steps="${RUN_STEPS:-200}"
solution_interval="${SOLUTION_INTERVAL:-10000}"
timeout_s="${CASE_TIMEOUT_SECONDS:-1200}"
ml_model_file="${PILOT_MODEL_FILE:-./input/transformer_inference_scripted_fw2.pt}"
ml_cube_overlap="${ML_CUBE_OVERLAP:-0}"

for v in "${cpu_tasks}" "${gpu_tasks}" "${gpus}" "${run_steps}" "${solution_interval}" "${timeout_s}"; do
    [[ "${v}" =~ ^[0-9]+$ ]] || { printf 'Expected a non-negative integer: %s\n' "${v}" >&2; exit 2; }
done
(( ml_cube_overlap < 8 )) || { printf 'ML_CUBE_OVERLAP must be < mlCubeD (8).\n' >&2; exit 2; }

maia_build_dir="$(realpath "${MAIA_BUILD_DIR:-${project_dir}/maia/build_gnu_production}")"
maia="${maia_build_dir}/bin/maia"
[[ -x "${maia}" ]] || { printf 'MAIA not executable: %s\n' "${maia}" >&2; exit 1; }

source "${project_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${maia_build_dir}/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}:${LD_LIBRARY_PATH:-}"

# Runtime AIX distribution-layout selection (RUNPATH yields to LD_LIBRARY_PATH).
if [[ -n "${AIX_LAYOUT_LIB:-}" ]]; then
    aix_layout_libdir="${AIX_LAYOUT_LIB}"
    [[ -f "${aix_layout_libdir}/libAIxeleratorService.so" ]] || aix_layout_libdir="${AIX_LAYOUT_LIB}/lib"
    [[ -f "${aix_layout_libdir}/libAIxeleratorService.so" ]] || { printf 'AIX_LAYOUT_LIB has no libAIxeleratorService.so: %s\n' "${AIX_LAYOUT_LIB}" >&2; exit 1; }
    export LD_LIBRARY_PATH="${aix_layout_libdir}:${LD_LIBRARY_PATH}"
fi

export CPP_ML_INTERFACE_PROVIDER_ENV="${provider}"
export CPP_ML_INTERFACE_DEVICE=GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
export MAIA_ML_STEP_TIMING_SYNC="${MAIA_ML_STEP_TIMING_SYNC:-1}"
if [[ "${ML_STEP_TIMING_PRINT:-0}" == 1 ]]; then
    export MAIA_ML_STEP_TIMING_PRINT=1
fi
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
# Defensive: exclude UCX's CMA transport, which aborts with "Bad address" on
# very large intra-node transfers. This does not change OpenMPI's PML and does
# NOT fix invalid (overflowed) gather displacements; the workload size does.
export UCX_TLS="${UCX_TLS:-^cma}"
export SCOREP_ENABLE_TRACING=false SCOREP_ENABLE_PROFILING=false
unset FLOW_DEBUG_DUMP_DIR MAIA_SNAPSHOT_DIR MLCOUPLING_DEBUG_EXPORT
unset MLCOUPLING_DEBUG_ALL_RANKS MLCOUPLING_DEBUG_EXPORT_DIR MLCOUPLING_DEBUG_RANK
unset MLCOUPLING_DEBUG_MAX_INFERENCES DUMP_TENSOR_DIR AIX_P2P_TIMELINE_DIR
unset SSDB AIX_DIAGNOSTICS AIX_DIAGNOSTIC_BARRIERS AIX_P2P_INITIAL_CREDITS
if [[ -n "${aix_credits}" ]]; then
    export AIX_P2P_INITIAL_CREDITS="${aix_credits}"
fi

scratch_root="${SMOKE_SCRATCH_ROOT:-${project_dir}/scratch/cmi_bench}"
mkdir -p "${scratch_root}"
scratch_root="$(realpath "${scratch_root}")"
cell="${CELL_NAME:-het_${cpu_tasks}p${gpu_tasks}c${gpus}g_${aix_comm}}"
run_dir="$(mktemp -d "${scratch_root}/${cell}_${SLURM_JOB_ID}_XXXXXX")"
cd "${run_dir}"
mkdir out auxdata boxes planes tmp
export TMPDIR="${run_dir}/tmp"
# Per-node resource sampler + opt-in NUMA pinner, wired into the rank wrapper.
export CMI_SAMPLER_SCRIPT="${project_dir}/slurm/cmi_node_sampler.sh"
export CMI_SAMPLER_DIR="${run_dir}"
export CMI_NUMA_PIN_SCRIPT="${project_dir}/slurm/cmi_numa_pin.py"
export CMI_BIND_LOG="${CMI_BIND_LOG:-1}"
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
    -e 's/^mlInputStepDistance *=.*/mlInputStepDistance = 1/' \
    -e "s|^modelPath *=.*|modelPath = \"${ml_model_file}\"|" \
    -e "s/^mlCubeOverlap *=.*/mlCubeOverlap = ${ml_cube_overlap}/" \
    -e "s/^solutionInterval *=.*/solutionInterval = ${solution_interval}/" properties.toml
model_path="$(python3 -c 'import tomllib; print(tomllib.load(open("properties.toml","rb"))["modelPath"])')"
[[ -r "${model_path}" ]] || { printf 'Input model not readable: %s\n' "${model_path}" >&2; exit 1; }

if [[ "${provider}" == AIX ]]; then
    cp "${project_dir}/config_aix.toml" config_aix.toml
    sed -i '/^[[:space:]]*communication_mode[[:space:]]*=/d' config_aix.toml
    sed -i "/^\[library\]/a communication_mode = \"${aix_comm}\"" config_aix.toml
    if [[ "${aix_comm}" == pipelined ]]; then
        export AIX_P2P_TIMELINE_DIR="${run_dir}/aix_gpu_diagnostics"
    else
        export AIX_DIAGNOSTICS=1 AIX_DIAGNOSTICS_DIR="${run_dir}/aix_gpu_diagnostics"
    fi
elif [[ "${provider}" == PHYDLL ]]; then
    cp "${project_dir}/config_phydll.toml" config_phydll.toml
    sed -i '/^[[:space:]]*solver_readiness_wait[[:space:]]*=/d' config_phydll.toml
    if [[ "${phydll_wait}" == readiness ]]; then
        sed -i '/^\[library\]/a solver_readiness_wait = true' config_phydll.toml
    fi
else
    cp "${project_dir}/config_smartsim.toml" config_smartsim.toml
fi
if [[ "${DEBUG_EXPORT:-0}" == 1 ]]; then
    mkdir debug_dumps
    export MLCOUPLING_DEBUG_EXPORT=1 MLCOUPLING_DEBUG_ALL_RANKS=1
    export MLCOUPLING_DEBUG_EXPORT_DIR="${run_dir}/debug_dumps"
    export MLCOUPLING_DEBUG_MAX_INFERENCES=2
fi

{
    printf 'CELL=%s\n' "${cell}"
    printf 'REP=%s\n' "${REP:-}"
    printf 'RUN_DIR=%s\n' "${run_dir}"
    printf 'MAIA=%s\n' "${maia}"
    printf 'MODEL=%s\n' "${model_path}"
    printf 'RUN_STEPS=%s\n' "${run_steps}"
    printf 'SOLUTION_INTERVAL=%s\n' "${solution_interval}"
    printf 'ML_CUBE_OVERLAP=%s\n' "${ml_cube_overlap}"
    printf 'MAIA_AIX_BATCHSIZE=%s\n' "${MAIA_AIX_BATCHSIZE:-default}"
    printf 'MAIA_PHYDLL_BATCHSIZE=%s\n' "${MAIA_PHYDLL_BATCHSIZE:-default}"
    printf 'MAIA_SMARTSIM_NUM_GPUS=%s\n' "${MAIA_SMARTSIM_NUM_GPUS:-default}"
    printf 'MAIA_ML_WORKLOAD_LOG=%s\n' "${MAIA_ML_WORKLOAD_LOG:-default}"
    printf 'PROVIDER=%s\n' "${provider}"
    printf 'HET_CPU_TASKS=%s\n' "${cpu_tasks}"
    printf 'HET_GPU_TASKS=%s\n' "${gpu_tasks}"
    printf 'HET_GPUS=%s\n' "${gpus}"
    printf 'NP_DL=%s\n' "${np_dl}"
    printf 'PHYDLL_WAIT=%s\n' "${phydll_wait}"
    printf 'AIX_COMM=%s\n' "${aix_comm}"
    printf 'AIX_CREDITS=%s\n' "${aix_credits:-unset}"
    printf 'AIX_LAYOUT_LIB=%s\n' "${AIX_LAYOUT_LIB:-default}"
    printf 'MAIA_ML_STEP_TIMING_SYNC=%s\n' "${MAIA_ML_STEP_TIMING_SYNC}"
    printf 'MAIA_ML_STEP_TIMING_PRINT=%s\n' "${MAIA_ML_STEP_TIMING_PRINT:-0}"
    printf 'DEBUG_EXPORT=%s\n' "${DEBUG_EXPORT:-0}"
    printf 'BATCH_HOST=%s\n' "$(hostname -s 2>/dev/null || hostname)"
    if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
        printf 'HET_GPU_NODELIST=%s\n' "${SLURM_JOB_NODELIST_HET_GROUP_1:-unknown}"
    fi
} | tee run.info

nvidia-smi -L > gpu_list.log 2>&1 || true

# Fold the per-node sampler peaks (written by cmi_node_sampler.sh on each node)
# into run.info.
collect_peaks() {
    local f host mib idx peak
    for f in "${run_dir}"/mem_*.txt; do
        [[ -r "${f}" ]] || continue
        host="$(basename "${f}" .txt)"; host="${host#mem_}"
        mib="$(cat "${f}" 2>/dev/null || echo 0)"
        [[ "${mib}" =~ ^[0-9]+$ ]] || mib=0
        printf 'NODE_RAM_PEAK_MIB_%s=%s\n' "${host}" "${mib}" >> run.info
    done
    for f in "${run_dir}"/gpu_*.txt; do
        [[ -r "${f}" ]] || continue
        host="$(basename "${f}" .txt)"; host="${host#gpu_}"
        while read -r idx peak; do
            [[ -n "${idx}" && -n "${peak}" ]] || continue
            printf 'GPU_VRAM_PEAK_MIB_%s_gpu%s=%s\n' "${host}" "${idx}" "${peak}" >> run.info
        done < "${f}"
    done
    for f in "${run_dir}"/bind_*.txt; do
        [[ -r "${f}" ]] || continue
        host="$(basename "${f}" .txt)"; host="${host#bind_}"
        ranks="$(wc -l < "${f}")"
        masks="$(sed -E 's/.*mask=//' "${f}" | sort -u | wc -l)"
        printf 'BIND_HOST_%s_ranks=%s\nBIND_HOST_%s_distinct_masks=%s\n' "${host}" "${ranks}" "${host}" "${masks}" >> run.info
    done
    for f in "${run_dir}"/gpubase_*.txt; do
        [[ -r "${f}" ]] || continue
        host="$(basename "${f}" .txt)"; host="${host#gpubase_}"
        while read -r idx base; do
            [[ -n "${idx}" && -n "${base}" ]] || continue
            printf 'GPU_VRAM_BASE_MIB_%s_gpu%s=%s\n' "${host}" "${idx}" "${base}" >> run.info
        done < "${f}"
    done
    for f in "${run_dir}"/gpuproc_*.txt; do
        [[ -r "${f}" ]] || continue
        host="$(basename "${f}" .txt)"; host="${host#gpuproc_}"
        printf 'GPU_PROC_PEAK_%s=%s\n' "${host}" "$(cat "${f}" 2>/dev/null || echo 0)" >> run.info
    done
}
cleanup() {
    local status=$?
    if [[ -n "${controller_pid}" ]]; then
        touch .solver_done
        kill -TERM -- "-${controller_pid}" 2>/dev/null || kill "${controller_pid}" 2>/dev/null || true
        wait "${controller_pid}" 2>/dev/null || true
    fi
    collect_peaks
    printf 'EXIT_STATUS=%s\n' "${status}" >> run.info
    printf 'Preserved bench artifacts: %s\n' "${run_dir}"
    exit "${status}"
}
trap cleanup EXIT

pin="${project_dir}/slurm/pin_cmi_rank.sh"
if [[ "${provider}" == AIX ]]; then
    solver_cmd=(srun --label --mpi=pmix --cpu-bind=cores --distribution=block --kill-on-bad-exit=1
        --export=ALL,HET_SURROGATE_HIDE_GPU=1
        --het-group=0 --nodes=1 --ntasks="${cpu_tasks}" --cpus-per-task=1
        /bin/bash "${pin}" "${maia}" ./properties.toml
        :
        --export=ALL
        --het-group=1 --nodes=1 --ntasks="${gpu_tasks}" --cpus-per-task=1
        /bin/bash "${pin}" "${maia}" ./properties.toml)
elif [[ "${provider}" == PHYDLL ]]; then
    # PhyDLL: group 0 hosts the CPU-side solver ranks; group 1 hosts the GPU-side
    # solver ranks plus the DL clients, dispatched by local rank (Slurm cannot
    # reference a het group twice in one srun MPMD).
    dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
    [[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
    [[ -x "${dl_client}" ]] || { printf 'C++ PhyDLL client missing in %s\n' "${maia_build_dir}" >&2; exit 1; }
    printf '#!/bin/bash\nexec /bin/bash "%s" "%s" ./properties.toml\n' "${pin}" "${maia}" > solver_cmd.sh
    printf '#!/bin/bash\nexec /bin/bash "%s" "%s"\n' "${pin}" "${dl_client}" > dl_cmd.sh
    chmod +x solver_cmd.sh dl_cmd.sh
    # Number of DL fields for uniform-chunk transport (five input, two output
    # frames); must match the solver provider or the client aborts at startup.
    export PHYDLL_DL_COUNT=2 PHYDLL_DL_FIELD_COUNT=2
    export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
    solver_cmd=(srun --label --mpi=pmix --cpu-bind=none --kill-on-bad-exit=1 --wait=0
        --export=ALL,HET_SURROGATE_HIDE_GPU=1
        --het-group=0 --nodes=1 --ntasks="${cpu_tasks}" --cpus-per-task=1
        /bin/bash "${pin}" "${maia}" ./properties.toml
        :
        --export=ALL
        --het-group=1 --nodes=1 --ntasks="$((gpu_tasks + np_dl))" --cpus-per-task=1 --overcommit
        /bin/bash "${project_dir}/slurm/cmi_bench_dispatch.sh" "${gpu_tasks}" ./solver_cmd.sh ./dl_cmd.sh)
else
    # SmartSim: the RedisAI database runs on the GPU-side het group (SmartSim's
    # Slurm launcher with --het-group), the solver spans both groups.
    smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/$(whoami)/smartsim/python/smartsim_cuda-12/bin/activate}"
    [[ -r "${smart_env}" ]] || { printf 'SmartSim environment missing: %s\n' "${smart_env}" >&2; exit 1; }
    source "${smart_env}"
    export SR_CMD_TIMEOUT=900 SR_SOCKET_TIMEOUT=900000 SR_MODEL_TIMEOUT=900000
    export SMARTSIM_MPI_SEQUENTIAL_PUT=0
    export SR_LOG_LEVEL=INFO SR_LOG_FILE="${run_dir}/smartredis.log"
    iface="${SMARTSIM_INTERFACE:-lo}"
    timeout --signal=TERM --kill-after=10 "$((timeout_s + ready_s + 60))s" \
        python3 "${project_dir}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
        --launcher slurm --het-group=1 --interface "${iface}" --db-nodes "${SMARTSIM_DB_NODES:-1}" --auto-port --use-gpu \
        --intra-op-threads 4 --inter-op-threads 1 --threads-per-queue "${SMARTSIM_TPQ:-1}" \
        --cpu-cores-per-node "${gpu_tasks}" --timeout-s "${ready_s}" \
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
    solver_cmd=(srun --label --mpi=pmix --cpu-bind=cores --distribution=block --kill-on-bad-exit=1
        --export=ALL,HET_SURROGATE_HIDE_GPU=1
        --het-group=0 --nodes=1 --ntasks="${cpu_tasks}" --cpus-per-task=1
        /bin/bash "${pin}" "${maia}" ./properties.toml
        :
        --export=ALL
        --het-group=1 --nodes=1 --ntasks="${gpu_tasks}" --cpus-per-task=1
        /bin/bash "${pin}" "${maia}" ./properties.toml)
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
