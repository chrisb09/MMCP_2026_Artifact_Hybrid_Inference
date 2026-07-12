#!/usr/bin/zsh
#SBATCH --job-name=rebuild-verif
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --output=rebuild_verif_%j.txt
set -euxo pipefail
source setup_env_claix23.sh

export CPP_ML_INTERFACE_PROVIDER_ENV="${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}"
echo "Using CPP_ML_INTERFACE_PROVIDER_ENV=${CPP_ML_INTERFACE_PROVIDER_ENV}"
provider_suffix="${(L)CPP_ML_INTERFACE_PROVIDER_ENV}"
maia_build_dir="${PWD}/maia/build_gnu_production_${provider_suffix}"

export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"

echo "=== Building MAIA ==="
cd "${maia_build_dir}"
cmake .
make -j24
cd ../..

echo "=== Running Verification ==="
export MAIA_SNAPSHOT_DIR="${PWD}/debug_dumps"
mkdir -p "${MAIA_SNAPSHOT_DIR}"

TOML_FILE="${PWD}/input/properties_run_les_ref_medium.toml"
temp_toml="temp_verif_300_properties.toml"
cp "$TOML_FILE" "$temp_toml"

sed -i "s/^timeSteps *=.*/timeSteps = 300/" "$temp_toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "$temp_toml"
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "$temp_toml"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "$temp_toml"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "$temp_toml"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "$temp_toml"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "$temp_toml"

ln -sf "${PWD}/input/grid_les_medium.hdf5" .
mkdir -p out auxdata logs
ln -sf "${PWD}/input/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5"

rm -f m_log forces.0.dat Residual

srun --label ${maia_build_dir}/bin/maia ./"${temp_toml}"

SNAPSHOT_TMP="/tmp/maia_snapshots_${SLURM_JOB_ID}.h5"
SNAPSHOT_DEST="${MAIA_SNAPSHOT_DIR}/snapshots_300.h5"

if [[ -f "$SNAPSHOT_TMP" ]]; then
    echo "Copying snapshot file: ${SNAPSHOT_TMP} -> ${SNAPSHOT_DEST}"
    cp "$SNAPSHOT_TMP" "$SNAPSHOT_DEST"
else
    echo "Warning: Snapshot file not found at ${SNAPSHOT_TMP}"
fi
