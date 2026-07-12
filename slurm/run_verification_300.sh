#!/usr/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=maia-verification-300
#SBATCH --output=logs/output_verif_300_%J.txt
#SBATCH --error=logs/error_verif_300_%J.txt

set -euxo pipefail

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

export CPP_ML_INTERFACE_PROVIDER_ENV="${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}"
echo "Using CPP_ML_INTERFACE_PROVIDER_ENV=${CPP_ML_INTERFACE_PROVIDER_ENV}"
provider_suffix="${(L)CPP_ML_INTERFACE_PROVIDER_ENV}"
maia_build_dir="${project_folder}/maia/build_gnu_production_${provider_suffix}"

export MAIA_SNAPSHOT_DIR="${project_folder}/debug_dumps"
mkdir -p "${MAIA_SNAPSHOT_DIR}"

TOML_FILE="${project_folder}/input/properties_run_les_ref_medium.toml"
temp_toml="temp_verif_300_properties.toml"
cp "$TOML_FILE" "$temp_toml"

sed -i "s/^timeSteps *=.*/timeSteps = 300/" "$temp_toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "$temp_toml"

# Apply correct ML parameters!
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "$temp_toml"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "$temp_toml"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "$temp_toml"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "$temp_toml"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "$temp_toml"

ln -sf "${project_folder}/input/grid_les_medium.hdf5" .
mkdir -p out auxdata logs
ln -sf "${project_folder}/input/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5"

rm -f m_log forces.0.dat Residual

source "${project_folder}/setup_env_claix23.sh"

echo "=== Starting verification run (300 steps, CPU-only devel partition) ==="
srun --label ${maia_build_dir}/bin/maia ./"${temp_toml}"
echo "=== Run complete ==="

SNAPSHOT_TMP="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
SNAPSHOT_DEST="${MAIA_SNAPSHOT_DIR}/snapshots_300.h5"

if [[ -f "$SNAPSHOT_TMP" ]]; then
    echo "Copying snapshot file: ${SNAPSHOT_TMP} -> ${SNAPSHOT_DEST}"
    cp "$SNAPSHOT_TMP" "$SNAPSHOT_DEST"
    echo "Snapshot copy complete. Size: $(du -h "$SNAPSHOT_DEST" | cut -f1)"
else
    echo "Warning: Snapshot file not found at ${SNAPSHOT_TMP}"
fi
