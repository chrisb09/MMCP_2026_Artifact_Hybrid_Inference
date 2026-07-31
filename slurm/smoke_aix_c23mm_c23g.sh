#!/bin/zsh

############################
# Global job options
############################
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-smoke-aix-c23mm-c23g
#SBATCH --output=logs/output_smoke_aix_%J.txt
#SBATCH --error=logs/error_smoke_aix_%J.txt

############################
# Component 0: CPU solver (c23mm)
############################
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1

#SBATCH hetjob

############################
# Component 1: ML side / GPU (c23g)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

set -euxo pipefail

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"
toml_folder="${project_folder}/input"
maia_build_dir="${project_folder}/maia/build_gnu_production"
run_steps="${RUN_STEPS:-50}"

run_dir="${project_folder}/scratch/smoke_aix_${SLURM_JOB_ID}"
mkdir -p "${run_dir}/out" "${run_dir}/auxdata" "${run_dir}/logs" "${project_folder}/logs"

source "${project_folder}/setup_env_claix23.sh"
cuda_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${cuda_lib}:${LD_LIBRARY_PATH:-}"

export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
export CPP_ML_INTERFACE_DEVICE=GPU
if [[ "${CAPTURE_SNAPSHOTS:-1}" == "1" ]]; then
    export MAIA_SNAPSHOT_DIR="${run_dir}/snapshots"
    mkdir -p "${MAIA_SNAPSHOT_DIR}"
else
    unset MAIA_SNAPSHOT_DIR
fi

TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
temp_toml="${run_dir}/properties.toml"
cp "${TOML_FILE}" "${temp_toml}"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${temp_toml}"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${temp_toml}"
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "${temp_toml}"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "${temp_toml}"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "${temp_toml}"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "${temp_toml}"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "${temp_toml}"

ln -sf "${toml_folder}/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -sf "${toml_folder}/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"
ln -sf "${project_folder}/input" "${run_dir}/input"

cp "${project_folder}/config_aix.toml" "${run_dir}/config_aix.toml"
cp "${project_folder}/config_aix.toml" "${run_dir}/config.toml"

persist_snapshots() {
    local exit_code=$?
    [[ -n "${MAIA_SNAPSHOT_DIR:-}" ]] || return "${exit_code}"
    local snapshot_tmp="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
    local snapshot_dest="${MAIA_SNAPSHOT_DIR}/snapshots_50.h5"

    if [[ -s "${snapshot_tmp}" ]]; then
        cp -f "${snapshot_tmp}" "${snapshot_dest}"
        echo "Snapshot copy complete: ${snapshot_dest} ($(stat --printf='%s bytes' "${snapshot_dest}"))"
    else
        echo "WARNING: snapshot file not found or empty: ${snapshot_tmp}" >&2
    fi
    return "${exit_code}"
}
trap persist_snapshots EXIT

cd "${run_dir}"

echo "=== Starting AIX GPU smoke test (24 CPU ranks on c23mm + 1 GPU rank on c23g) ==="
start_seconds=$(date +%s)
srun --label --mpi=pmix \
    --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        "${project_folder}/CPP-ML-Interface/dl_clients/maia_runner.sh" "${maia_build_dir}/bin/maia" ./properties.toml : \
    --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
        "${project_folder}/CPP-ML-Interface/dl_clients/maia_runner.sh" "${maia_build_dir}/bin/maia" ./properties.toml
echo "=== AIX GPU smoke test complete ==="
echo "BENCHMARK_SOLVER_WALL_SECONDS=$(( $(date +%s) - start_seconds ))"
