#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-hybrid-phydll
#SBATCH --output=logs/hybrid_phydll_%j.out
#SBATCH --error=logs/hybrid_phydll_%j.err
# Group 0: solver (24 ranks, c23g node — GPU present but not used by solver)
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH hetjob
# Group 1: PhyDLL DL client + GPU
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

set -euo pipefail

client_kind="${PHYDLL_CLIENT:-cpp}"
[[ "${client_kind}" == "cpp" || "${client_kind}" == "python" ]] || { echo "PHYDLL_CLIENT must be cpp or python." >&2; exit 2; }
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-20}"
build_variant="${BUILD_VARIANT:-plain}"
build_suffix=""
[[ "${build_variant}" == "scorep" ]] && build_suffix="_scorep"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production_cmi${build_suffix}}"
phydll_python_env="${PHYDLL_PYTHON_ENV:-/hpcwork/${USER}/smartsim/python/smartsim_cuda-12/bin/activate}"
run_dir="${project_folder}/scratch/hybrid_phydll_${client_kind}_${SLURM_JOB_ID}"

if [[ "${build_variant}" == "scorep" ]]; then
    export USE_SCOREP=1
    export SCOREP_METRIC_PAPI=""
fi
source "${project_folder}/setup_env_claix23.sh"

if [[ "${client_kind}" == "python" ]]; then
    [[ -f "${phydll_python_env}" ]] || { echo "PhyDLL Python environment not found: ${phydll_python_env}" >&2; exit 1; }
    source "${phydll_python_env}"
fi

mkdir -p "${run_dir}/out"
ln -s "${project_folder}/input" "${run_dir}/input"
cp "${project_folder}/config_phydll.toml" "${run_dir}/"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
export CPP_ML_INTERFACE_DEVICE=GPU
if [[ "${MLCOUPLING_DEBUG_EXPORT:-0}" == "1" ]]; then
    debug_export_dir="${MLCOUPLING_DEBUG_EXPORT_DIR:-${run_dir}/debug}"
    mkdir -p "${debug_export_dir}"
    export FLOW_DEBUG_DUMP_DIR="${debug_export_dir}/cmi"
    export MAIA_SNAPSHOT_DIR="${debug_export_dir}/snapshots"
else
    unset FLOW_DEBUG_DUMP_DIR
    unset MAIA_SNAPSHOT_DIR
fi
export LD_LIBRARY_PATH="${project_folder}/CPP-ML-Interface/extern/phydll/build/lib:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_INTRA_OP_THREADS=24
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_DL_COUNT=1
export PHYDLL_DL_FIELD_COUNT=1
export OMPI_MCA_pmix="^s1,s2"

persist_snapshots() {
    local status=$?
    [[ -n "${MAIA_SNAPSHOT_DIR:-}" ]] || return "${status}"
    local snapshot_tmp="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
    local snapshot_dest="${MAIA_SNAPSHOT_DIR}/snapshots_${SLURM_JOB_ID}.h5"

    if [[ -s "${snapshot_tmp}" ]]; then
        mkdir -p "${MAIA_SNAPSHOT_DIR}"
        cp -f "${snapshot_tmp}" "${snapshot_dest}"
        echo "Snapshot copy complete: ${snapshot_dest} ($(stat --printf='%s bytes' "${snapshot_dest}"))"
        if command -v h5ls >/dev/null; then
            echo "Snapshot contents:"
            h5ls -r "${snapshot_dest}"
        fi
    else
        echo "WARNING: snapshot file was not found or empty: ${snapshot_tmp}" >&2
    fi
    return "${status}"
}
trap persist_snapshots EXIT

dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
[[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
[[ -x "${dl_client}" ]] || { echo "PhyDLL C++ client not found." >&2; exit 1; }

if [[ "${build_variant}" == "scorep" ]]; then
    export SCOREP_ENABLE_TRACING=false
    export SCOREP_ENABLE_PROFILING=true
    export SCOREP_MPI_ENABLE_GROUPS="NONE"
    mkdir -p "${run_dir}/scorep-results"
fi

cd "${run_dir}"

if [[ "${client_kind}" == "cpp" ]]; then
    if [[ "${build_variant}" == "scorep" ]]; then
        srun --label --mpi=pmix --export=ALL --preserve-env \
            --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
                bash -c 'export SCOREP_EXPERIMENT_DIRECTORY="'"${run_dir}"'/scorep-results/solver_rank_${SLURM_PROCID}"; exec "'"${maia_build_dir}"'/bin/maia" ./properties.toml' : \
            --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
                bash -c 'export SCOREP_EXPERIMENT_DIRECTORY="'"${run_dir}"'/scorep-results/ml_rank_${SLURM_PROCID}"; exec "'"${dl_client}"'"'
    else
        srun --label --mpi=pmix --export=ALL --preserve-env \
            --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
                "${maia_build_dir}/bin/maia" ./properties.toml : \
            --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
                "${dl_client}"
    fi
else
    # Python DL client
    if [[ "${build_variant}" == "scorep" ]]; then
        dl_cmd="python3 -m scorep --keep-files --instrumenter-type=dummy --noinstrumenter --mpp=none ${project_folder}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
    else
        dl_cmd="python3 ${project_folder}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
    fi

    if [[ "${build_variant}" == "scorep" ]]; then
        srun --label --mpi=pmix --export=ALL --preserve-env \
            --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
                bash -c 'export SCOREP_EXPERIMENT_DIRECTORY="'"${run_dir}"'/scorep-results/solver_rank_${SLURM_PROCID}"; exec "'"${maia_build_dir}"'/bin/maia" ./properties.toml' : \
            --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
                bash -c 'export SCOREP_EXPERIMENT_DIRECTORY="'"${run_dir}"'/scorep-results/ml_rank_${SLURM_PROCID}"; source "'"${phydll_python_env}"'" && '"${dl_cmd}"
    else
        srun --label --mpi=pmix --export=ALL --preserve-env \
            --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
                "${maia_build_dir}/bin/maia" ./properties.toml : \
            --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
                bash -c "source '${phydll_python_env}' && ${dl_cmd}"
    fi
fi
