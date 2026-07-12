#!/usr/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=maia-debug-dump-devel
#SBATCH --output=logs/output_debug_devel_%J.txt
#SBATCH --error=logs/error_debug_devel_%J.txt

set -euxo pipefail

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

if [ ! -d "$project_folder" ]; then
    echo "Project folder '$project_folder' not found!"
    exit 1
fi

toml_folder="${project_folder}/input"

# ---- Debug dump settings ----
export FLOW_DEBUG_DUMP_DIR="/tmp/flow_debug"
export MAIA_SNAPSHOT_DIR="${project_folder}/debug_dumps"
rm -rf "${FLOW_DEBUG_DUMP_DIR}"
mkdir -p "${FLOW_DEBUG_DUMP_DIR}"
echo "FLOW_DEBUG_DUMP_DIR=${FLOW_DEBUG_DUMP_DIR}"

# ---- Create a 20-step TOML override with CPU-only hostFraction ----
TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
if [ ! -f "$TOML_FILE" ]; then
    echo "TOML file '$TOML_FILE' not found!"
    exit 1
fi

temp_toml="temp_debug_devel_20steps_properties_run_les_ref_medium.toml"
cp "$TOML_FILE" "$temp_toml"
# Override timeSteps to 20 — enough to cover steps 11-15 (first inference) plus some margin
sed -i "s/^timeSteps *=.*/timeSteps = 100/" "$temp_toml"
# Match the reference/debug schedule: collect five consecutive sends, then jump 24 steps.
sed -i "s/^mlInterval *=.*/mlInterval = 5/" "$temp_toml"
sed -i "s/^mlStepCoefficient *=.*/mlStepCoefficient = 12/" "$temp_toml"
sed -i "s/^mlForecastWindow *=.*/mlForecastWindow = 2/" "$temp_toml"
sed -i "s/^mlScalingFactor *=.*/mlScalingFactor = 1/" "$temp_toml"
sed -i "s/^mlInputStepDistance *=.*/mlInputStepDistance = 1/" "$temp_toml"
# Override hostFraction to "1.00" to run CPU-only on the devel partition
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "$temp_toml"
echo "Using temp TOML: $temp_toml  (timeSteps=100, mlInterval=5, mlStepCoefficient=12, hostFraction=1.00)"

# ---- Grid / restart symlinks ----
ln -sf "${toml_folder}/grid_les_medium.hdf5" .
mkdir -p out auxdata logs
ln -sf "${toml_folder}/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5"

rm -f m_log forces.0.dat Residual

source "${project_folder}/setup_env_claix23.sh"
# Using system python environment with all required modules

export CPP_ML_INTERFACE_PROVIDER_ENV="${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}"
provider_suffix="${(L)CPP_ML_INTERFACE_PROVIDER_ENV}"
maia_build_dir="${project_folder}/maia/build_gnu_production_cmi"


echo "=== Starting debug run (20 steps, CPU-only devel partition) ==="
srun --label ${maia_build_dir}/bin/maia ./"${temp_toml}"

echo "=== Run complete ==="
echo "Debug dumps in ${FLOW_DEBUG_DUMP_DIR}:"
ls -lh "${FLOW_DEBUG_DUMP_DIR}/" || echo "No dumps generated!"

# ---- Copy dumps to persistent workspace ----
dest="${project_folder}/debug_dumps"
rm -rf "${dest}"
mkdir -p "${dest}"
echo "Copying dumps to ${dest} ..."
cp -r "${FLOW_DEBUG_DUMP_DIR}/." "${dest}/"
echo "Manifest:"
cat "${dest}/manifest.txt" 2>/dev/null || echo "(no manifest found)"
echo "=== Done. Dumps in ${dest} ==="
