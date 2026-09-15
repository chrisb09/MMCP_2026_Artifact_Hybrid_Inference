#!/usr/bin/zsh
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=maia-cmi240
#SBATCH --output=logs/output_current240_%J.txt
#SBATCH --error=logs/error_current240_%J.txt

set -euxo pipefail

username=$(whoami)
project_folder="/rwthfs/rz/cluster/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"
cd "${project_folder}"
mkdir -p logs out auxdata

toml_folder="${project_folder}/input"
dest="${project_folder}/debug_dumps/current_torch240"
rm -rf "${dest}"
mkdir -p "${dest}"

export MLCOUPLING_DEBUG_EXPORT=1
export MLCOUPLING_DEBUG_EXPORT_DIR="${dest}"
export MLCOUPLING_DEBUG_ALL_RANKS=1
export MLCOUPLING_DEBUG_MAX_INFERENCES=3
export MAIA_SNAPSHOT_DIR="${dest}"

echo "MLCOUPLING_DEBUG_EXPORT_DIR=${dest}"
echo "MAIA_SNAPSHOT_DIR=${dest}"

TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
temp_toml="temp_current240_101steps_properties.toml"
cp "$TOML_FILE" "$temp_toml"

# Run 101 steps to capture inferences 1 (step 15), 2 (step 51), and 3 (step 101)
sed -i "s/^timeSteps *=.*/timeSteps = 101/" "$temp_toml"
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "$temp_toml"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "$temp_toml"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "$temp_toml"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "$temp_toml"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "$temp_toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "$temp_toml"

echo "Using temp TOML: $temp_toml (timeSteps=101, hostFraction=1.00)"

ln -sf "${toml_folder}/grid_les_medium.hdf5" .
ln -sf "${toml_folder}/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5"
rm -f m_log forces.0.dat Residual

source "${project_folder}/setup_env_claix23.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
maia_build_dir="${project_folder}/maia/build_gnu_production"

echo "=== Starting MAIA CMI Torch 2.4.0 run (24 solver ranks, CPU-only devel) ==="
start_time=$(date +%s)
srun --mem-per-cpu=5G --label "${maia_build_dir}/bin/maia" ./"${temp_toml}"
echo "Run elapsed seconds: $(( $(date +%s) - start_time ))"

SNAPSHOT_TMP="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
if [[ -f "$SNAPSHOT_TMP" ]]; then
    echo "Copying snapshot file: ${SNAPSHOT_TMP} -> ${dest}/snapshots_${SLURM_JOB_ID}_rank_0.h5"
    cp "$SNAPSHOT_TMP" "${dest}/snapshots_${SLURM_JOB_ID}_rank_0.h5"
fi

echo "=== Run complete. Files in ${dest}: ==="
ls -lh "${dest}"
