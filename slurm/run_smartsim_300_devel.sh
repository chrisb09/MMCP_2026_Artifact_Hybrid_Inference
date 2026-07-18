#!/usr/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --oversubscribe
#SBATCH --job-name=maia-smartsim-300
#SBATCH --output=logs/output_smartsim_300_%J.txt
#SBATCH --error=logs/error_smartsim_300_%J.txt

set -euxo pipefail

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"
toml_folder="${project_folder}/input"
provider_suffix="smartsim"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production_cmi_scorep}"
run_steps="${RUN_STEPS:-300}"
smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/${username}/smartsim/python/smartsim_cpu/bin/activate}"

cd "${project_folder}"
mkdir -p logs out auxdata "debug_dumps/${provider_suffix}"

source "${project_folder}/setup_env_claix23.sh"
cuda_stubs="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
export LD_LIBRARY_PATH="${cuda_stubs}:${LD_LIBRARY_PATH:-}"
if [[ -f "${smart_env}" ]]; then
    source "${smart_env}"
else
    echo "SmartSim Python environment not found: ${smart_env}"
    exit 1
fi

export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
export FLOW_DEBUG_DUMP_DIR="${project_folder}/debug_dumps/${provider_suffix}"
export MAIA_SNAPSHOT_DIR="${project_folder}/debug_dumps/${provider_suffix}"
export MLCOUPLING_DEBUG_EXPORT=1
export MLCOUPLING_DEBUG_ALL_RANKS="${MLCOUPLING_DEBUG_ALL_RANKS:-1}"
export MLCOUPLING_DEBUG_EXPORT_DIR="${project_folder}/debug_dumps/${provider_suffix}/cmi_${SLURM_JOB_ID}"
mkdir -p "${MLCOUPLING_DEBUG_EXPORT_DIR}" "${MAIA_SNAPSHOT_DIR}"
export SR_CMD_TIMEOUT=600
export SR_SOCKET_TIMEOUT=600000
export SR_MODEL_TIMEOUT=600000
export SR_LOG_LEVEL=DEBUG
export SR_LOG_FILE="${project_folder}/logs/smartredis_\${SLURM_JOB_ID}.log"

TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
temp_toml="temp_smartsim_300_properties.toml"
cp "${TOML_FILE}" "${temp_toml}"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${temp_toml}"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${temp_toml}"
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "${temp_toml}"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "${temp_toml}"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "${temp_toml}"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "${temp_toml}"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "${temp_toml}"

ln -sf "${toml_folder}/grid_les_medium.hdf5" .
ln -sf "${toml_folder}/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5"
rm -f m_log forces.0.dat Residual .ssdb_endpoint .solver_done

cleanup() {
    touch .solver_done || true
    if [[ -n "${controller_pid:-}" ]]; then
        wait "${controller_pid}" || true
    fi
}
trap cleanup EXIT

echo "=== Starting SmartSim controller on 24 ML cores ==="
python3 CPP-ML-Interface/dl_clients/smartsim_controller.py \
    --launcher local \
    --interface lo \
    --db-nodes 1 \
    --intra-op-threads 8 \
    --threads-per-queue 8 \
    --cpu-cores-per-node 24 \
    --port 6780 \
    --endpoint-file .ssdb_endpoint \
    --done-file .solver_done \
    --exp-dir ./ssdb_exp &
controller_pid=$!

for _ in {1..120}; do
    if [[ -s .ssdb_endpoint ]]; then
        break
    fi
    sleep 1
done

if [[ ! -s .ssdb_endpoint ]]; then
    echo "SmartSim endpoint file was not created."
    exit 1
fi

export SSDB="$(cat .ssdb_endpoint)"
echo "SSDB=${SSDB}"

echo "=== Starting MAIA SmartSim run (24 solver ranks) ==="
srun --label --mpi=pmix -n 24 --ntasks-per-node=24 --cpus-per-task=1 --cpu-bind=cores \
    "${maia_build_dir}/bin/maia" ./"${temp_toml}"
echo "=== MAIA SmartSim run complete ==="

SNAPSHOT_TMP="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
SNAPSHOT_DEST="${project_folder}/debug_dumps/${provider_suffix}/snapshots_300.h5"
if [[ -f "${SNAPSHOT_TMP}" ]]; then
    cp "${SNAPSHOT_TMP}" "${SNAPSHOT_DEST}"
    echo "Snapshot copy complete: ${SNAPSHOT_DEST}"
else
    echo "Warning: Snapshot file not found at ${SNAPSHOT_TMP}"
fi
