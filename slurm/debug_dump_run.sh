#!/bin/zsh

############################
# Global job options
############################
#SBATCH --account=thes2181
#SBATCH --time=00:10:00
#SBATCH --job-name=maia-debug-dump
#SBATCH --exclusive
#SBATCH --output=logs/output_debug_%J.txt
#SBATCH --error=logs/error_debug_%J.txt

############################
# Component 0: CPU (c23mm)
############################
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1

#SBATCH hetjob

############################
# Component 1: GPU (c23g)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4


############################
# Runtime
############################
echo "CPU nodes:"
scontrol show hostname $SLURM_JOB_NODELIST_HET_GROUP_0

echo "GPU nodes:"
scontrol show hostname $SLURM_JOB_NODELIST_HET_GROUP_1

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

if [ ! -d "$project_folder" ]; then
    echo "Project folder '$project_folder' not found!"
    exit 1
fi

toml_folder="${project_folder}/input/"

# ---- Debug dump settings ----
# All FlowExtrapolator intermediates will be written here.
# /tmp is node-local. We dump on ALL ranks but the Python analysis
# only needs rank 0 data; MPI rank-0 outputs will dominate the comparison.
export FLOW_DEBUG_DUMP_DIR="/tmp/flow_debug"
mkdir -p "${FLOW_DEBUG_DUMP_DIR}"
echo "FLOW_DEBUG_DUMP_DIR=${FLOW_DEBUG_DUMP_DIR}"

# ---- Create a 20-step TOML override (minimal run to capture steps 11-15) ----
TOML_FILE="${toml_folder}/properties_run_les_ref_medium.toml"
if [ ! -f "$TOML_FILE" ]; then
    echo "TOML file '$TOML_FILE' not found!"
    exit 1
fi

temp_toml="temp_debug_20steps_properties_run_les_ref_medium.toml"
cp "$TOML_FILE" "$temp_toml"
# Override timeSteps to 20 — enough to cover steps 11-15 (first inference) plus some margin
sed -i "s/^timeSteps *=.*/timeSteps = 20/" "$temp_toml"
echo "Using temp TOML: $temp_toml  (timeSteps overridden to 20)"

# ---- Grid / restart symlinks ----
ln -sf "${toml_folder}/grid_les_medium.hdf5" .
ln -s "${toml_folder}/restart_les_init_medium.hdf5" "./out/restart_les_ref_medium.hdf5" 2>/dev/null || true

mkdir -p out auxdata logs

source "${project_folder}/setup_env_claix23.sh"
source "${project_folder}/CPP-ML-Interface/extern/python/venv/bin/activate"

echo "=== Starting debug run (20 steps) ==="
srun --export=ALL --het-group=0 --mpi=pmix --preserve-env --cpus-per-task=1 \
    ${project_folder}/maia/build_gnu_production/bin/maia ./"${temp_toml}" \
    : \
    --export=ALL --het-group=1 --mpi=pmix --preserve-env --cpus-per-task=1 \
    ${project_folder}/maia/build_gnu_production/bin/maia ./"${temp_toml}"

echo "=== Run complete ==="
echo "Debug dumps in ${FLOW_DEBUG_DUMP_DIR}:"
ls -lh "${FLOW_DEBUG_DUMP_DIR}/" | head -40

# ---- Copy dumps to persistent workspace ----
# /tmp is ephemeral on the compute node; copy to hpcwork before the job ends.
dest="${project_folder}/debug_dumps"
mkdir -p "${dest}"
echo "Copying dumps to ${dest} ..."
cp -r "${FLOW_DEBUG_DUMP_DIR}/." "${dest}/"
echo "Manifest:"
cat "${dest}/manifest.txt" 2>/dev/null || echo "(no manifest found)"
echo "=== Done. Dumps in ${dest} ==="
