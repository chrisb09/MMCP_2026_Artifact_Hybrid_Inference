#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-hybrid-smartsim
#SBATCH --output=logs/hybrid_smartsim_%j.out
#SBATCH --error=logs/hybrid_smartsim_%j.err
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-20}"
build_variant="${BUILD_VARIANT:-plain}"
build_suffix=""
[[ "${build_variant}" == "scorep" ]] && build_suffix="_scorep"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production_cmi${build_suffix}}"
smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/${USER}/smartsim/python/smartsim_cuda-12/bin/activate}"
network_interface="${HYBRID_NETWORK_INTERFACE:-ib0}"
run_dir="${project_folder}/scratch/hybrid_smartsim_${SLURM_JOB_ID}"

if [[ "${build_variant}" == "scorep" ]]; then
    export USE_SCOREP=1
    export SCOREP_METRIC_PAPI=""
fi
source "${project_folder}/setup_env_claix23.sh"
if [[ ! -f "${smart_env}" ]]; then
    echo "SmartSim Python environment not found: ${smart_env}" >&2
    exit 1
fi
source "${smart_env}"
mkdir -p "${run_dir}/out"
ln -s "${project_folder}/input" "${run_dir}/input"
cp "${project_folder}/config_smartsim.toml" "${run_dir}/"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
export CPP_ML_INTERFACE_DEVICE=GPU
export MLCOUPLING_SMARTSIM_NUM_GPUS=1
export FLOW_DEBUG_DUMP_DIR="${run_dir}/dumps"
export MAIA_SNAPSHOT_DIR="${run_dir}/snapshots"
export SR_CMD_TIMEOUT=600
export SR_SOCKET_TIMEOUT=600000
export SR_MODEL_TIMEOUT=600000
cleanup() {
    touch "${run_dir}/.solver_done"
    [[ -z "${controller_pid:-}" ]] || wait "${controller_pid}" || true
}
trap cleanup EXIT

srun --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
    bash -lc "cd '${run_dir}' && python3 '${project_folder}/CPP-ML-Interface/dl_clients/smartsim_controller.py' --launcher local --interface '${network_interface}' --use-gpu --db-nodes 1 --intra-op-threads 8 --threads-per-queue 1 --port 6780 --endpoint-file .ssdb_endpoint --done-file .solver_done --exp-dir ./ssdb_exp" &
controller_pid=$!

for _ in {1..120}; do
    [[ -s "${run_dir}/.ssdb_endpoint" ]] && break
    sleep 1
done
[[ -s "${run_dir}/.ssdb_endpoint" ]] || { echo "SmartSim endpoint was not created." >&2; exit 1; }
export SSDB="$(<"${run_dir}/.ssdb_endpoint")"

cd "${run_dir}"
srun --label --mpi=pmix --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
    "${maia_build_dir}/bin/maia" ./properties.toml
