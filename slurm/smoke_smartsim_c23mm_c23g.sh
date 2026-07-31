#!/bin/zsh

############################
# Global job options
############################
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-smoke-smartsim-c23mm-c23g
#SBATCH --output=logs/output_smoke_smartsim_%J.txt
#SBATCH --error=logs/error_smoke_smartsim_%J.txt

############################
# Component 0: CPU solver (c23mm)
############################
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1

#SBATCH hetjob

############################
# Component 1: SmartSim controller / GPU (c23g)
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
smart_env="/hpcwork/${username}/smartsim/python/smartsim_cuda-12/bin/activate"
network_interface="ib0"
run_steps="${RUN_STEPS:-50}"

run_dir="${project_folder}/scratch/smoke_smartsim_${SLURM_JOB_ID}"
mkdir -p "${run_dir}/out" "${run_dir}/auxdata" "${run_dir}/logs" "${project_folder}/logs"

source "${project_folder}/setup_env_claix23.sh"

if [[ ! -f "${smart_env}" ]]; then
    echo "SmartSim Python environment not found: ${smart_env}" >&2
    exit 1
fi
source "${smart_env}"

export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
export CPP_ML_INTERFACE_DEVICE=GPU
export MLCOUPLING_SMARTSIM_NUM_GPUS=1
if [[ "${CAPTURE_SNAPSHOTS:-1}" == "1" ]]; then
    export MAIA_SNAPSHOT_DIR="${run_dir}/snapshots"
    mkdir -p "${MAIA_SNAPSHOT_DIR}"
else
    unset MAIA_SNAPSHOT_DIR
fi

export SR_CMD_TIMEOUT=600
export SR_SOCKET_TIMEOUT=600000
export SR_MODEL_TIMEOUT=600000

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

cp "${project_folder}/config_smartsim.toml" "${run_dir}/config_smartsim.toml"
cp "${project_folder}/config_smartsim.toml" "${run_dir}/config.toml"

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

cleanup() {
    local exit_code=$?
    touch "${run_dir}/.solver_done"
    [[ -z "${controller_pid:-}" ]] || wait "${controller_pid}" || true
    persist_snapshots
    return "${exit_code}"
}
trap cleanup EXIT

# Launch SmartSim controller on het-group 1 (c23g GPU node)
srun --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
    bash -lc "cd '${run_dir}' && python3 '${project_folder}/CPP-ML-Interface/dl_clients/smartsim_controller.py' \
        --launcher local --interface '${network_interface}' --use-gpu \
        --db-nodes 1 --intra-op-threads 8 --threads-per-queue 1 \
        --port 6780 --endpoint-file .ssdb_endpoint --done-file .solver_done \
        --exp-dir ./ssdb_exp" &
controller_pid=$!

echo "Waiting for SmartSim database endpoint..."
for _ in {1..120}; do
    [[ -s "${run_dir}/.ssdb_endpoint" ]] && break
    sleep 1
done
[[ -s "${run_dir}/.ssdb_endpoint" ]] || { echo "SmartSim endpoint was not created." >&2; exit 1; }
export SSDB="$(<"${run_dir}/.ssdb_endpoint")"
echo "SmartSim DB active at SSDB=${SSDB}"

cd "${run_dir}"

echo "=== Starting SmartSim GPU smoke test (24 CPU ranks on c23mm + SmartSim DB on c23g GPU) ==="
start_seconds=$(date +%s)
srun --label --mpi=pmix \
        --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        "${project_folder}/CPP-ML-Interface/dl_clients/maia_runner.sh" "${maia_build_dir}/bin/maia" ./properties.toml
echo "=== SmartSim GPU smoke test complete ==="
echo "BENCHMARK_SOLVER_WALL_SECONDS=$(( $(date +%s) - start_seconds ))"
