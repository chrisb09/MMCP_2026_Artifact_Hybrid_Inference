#!/usr/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --oversubscribe
#SBATCH --job-name=maia-phydll-py-300
#SBATCH --output=logs/output_phydll_py_300_%J.txt
#SBATCH --error=logs/error_phydll_py_300_%J.txt

set -euxo pipefail

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"
toml_folder="${project_folder}/input"
provider_suffix="phydll"
snapshot_suffix="phydll_py"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production}"
run_steps="${RUN_STEPS:-300}"
np_phy="${NP_PHY:-24}"
np_dl="${NP_DL:-24}"
total_tasks=$((np_phy + np_dl))
phy_last=$((np_phy - 1))
dl_last=$((total_tasks - 1))
smart_env="/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/phydll_py_venv/bin/activate"

cd "${project_folder}"
mkdir -p logs out auxdata "debug_dumps/${snapshot_suffix}"

source "${project_folder}/setup_env_claix23.sh"
cuda_stubs="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
export LD_LIBRARY_PATH="${cuda_stubs}:${LD_LIBRARY_PATH:-}"

export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
export FLOW_DEBUG_DUMP_DIR="${project_folder}/debug_dumps/${snapshot_suffix}"
export MAIA_SNAPSHOT_DIR="${project_folder}/debug_dumps/${snapshot_suffix}"
export MLCOUPLING_DEBUG_EXPORT=1
export MLCOUPLING_DEBUG_ALL_RANKS="${MLCOUPLING_DEBUG_ALL_RANKS:-1}"
export MLCOUPLING_DEBUG_MAX_INFERENCES=100
export MLCOUPLING_DEBUG_EXPORT_DIR="${project_folder}/debug_dumps/${snapshot_suffix}/cmi_${SLURM_JOB_ID}"
mkdir -p "${MLCOUPLING_DEBUG_EXPORT_DIR}" "${MAIA_SNAPSHOT_DIR}"
export LD_LIBRARY_PATH="${project_folder}/CPP-ML-Interface/extern/phydll/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${project_folder}/CPP-ML-Interface/extern/phydll/src/python:${PYTHONPATH:-}"
export MLCOUPLING_INTRA_OP_THREADS=1
export MLCOUPLING_INTER_OP_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PHYDLL_DL_COUNT="${PHYDLL_DL_COUNT:-1}"
export PHYDLL_DL_EXIT_GRACE_SECONDS="${PHYDLL_DL_EXIT_GRACE_SECONDS:-900}"
export SCOREP_ENABLE_TRACING=false
export SCOREP_ENABLE_PROFILING=false
unset SCOREP_MPI_ENABLE_GROUPS
export OMPI_MCA_pmix=pmix3x
export OMPI_MCA_ess=pmi

TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
temp_toml="temp_phydll_py_300_properties.toml"
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
rm -f m_log forces.0.dat Residual phydll_run_py.conf

cat > phydll_run_py.conf <<EOF
0-${phy_last} ${project_folder}/CPP-ML-Interface/dl_clients/maia_runner.sh ${maia_build_dir}/bin/maia ./${temp_toml}
${np_phy}-${dl_last} ${project_folder}/CPP-ML-Interface/dl_clients/python_runner.sh ${project_folder}/CPP-ML-Interface/dl_clients/phydll_dl_client.py
EOF

echo "=== Starting PhyDLL Python run (${np_phy} solver + ${np_dl} DL ranks) ==="
srun --label --mpi=pmix -n "${total_tasks}" --ntasks-per-node="${total_tasks}" --cpus-per-task=1 --multi-prog ./phydll_run_py.conf
echo "=== MAIA PhyDLL Python run complete ==="

SNAPSHOT_TMP="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
SNAPSHOT_DEST="${project_folder}/debug_dumps/${snapshot_suffix}/snapshots_300.h5"
if [[ -f "${SNAPSHOT_TMP}" ]]; then
    cp "${SNAPSHOT_TMP}" "${SNAPSHOT_DEST}"
    echo "Snapshot copy complete: ${SNAPSHOT_DEST}"
else
    echo "Warning: Snapshot file not found at ${SNAPSHOT_TMP}"
fi
