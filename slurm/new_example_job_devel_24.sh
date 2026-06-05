#!/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name="maia-hybrid-devel-test-24-5g"
#SBATCH --output=logs/output.%J.txt
#SBATCH --error=logs/error.%J.txt

username=$(whoami)

project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

# Check if project folder exists
if [ ! -d "$project_folder" ]; then
    echo "Project folder '$project_folder' not found!"
    exit 1
fi

toml_folder="${project_folder}/input/"

# Check if toml folder exists
if [ ! -d "$toml_folder" ]; then
    echo "TOML folder '$toml_folder' not found!"
    exit 1
fi

# Usage:
#   sbatch maia.job <SIZE> <CASE_FOLDER> <case_type>
# where:
#   <SIZE> is one of L, M, S, ML, or SH
#   <CASE_FOLDER> is the relative folder path (e.g., "Actuated/L/W1000, L200, T20, A30 S")
#   <case_type> is either "actuated" or "nonactuated"


ROOT_DIR=$(pwd)
SIZE=$1
CASE_FOLDER=$2

mkdir -p out
mkdir -p auxdata

sleep 2

# Set GRID_SIZE and choose appropriate TOML and grid file based on SIZE.
if [[ "$SIZE" == "L" ]]; then
    GRID_SIZE="large"
elif [[ "$SIZE" == "M" ]]; then
    GRID_SIZE="medium"
elif [[ "$SIZE" == "SH" ]]; then
    GRID_SIZE="small_high"
elif [[ "$SIZE" == "ML" ]]; then
    GRID_SIZE="medium_low"
elif [[ "$SIZE" == "S" ]]; then
    GRID_SIZE="small"
else
    echo "Unknown size: $SIZE"
    exit 1
fi

ln -sf ${toml_folder}/grid_les_${GRID_SIZE}.hdf5 .

TOML_FILE="${toml_folder}/properties_run_les_ref_${GRID_SIZE}.toml"

# Check if the TOML file exists
if [ ! -f "$TOML_FILE" ]; then
    echo "TOML file '$TOML_FILE' not found!"
    echo "Available files in ${toml_folder}:"
    ls ${toml_folder} -lh
    exit 1
fi

# Create a temporary TOML file based on the actuated template.
temp_toml="temp_$(basename $TOML_FILE)"
cp "$TOML_FILE" "$temp_toml"

fraction=$(echo "$CASE_FOLDER" | sed -E 's/.*(A[0-9,]+).*/\1/' | sed 's/,/./g')
echo $fraction
sed -i "s/^hostFraction *= *.*/hostFraction = \"$fraction\"/" "$temp_toml"
# Use the temporary file as our TOML file.
TOML_FILE="$temp_toml"

ln -s "${toml_folder}/restart_les_init_${GRID_SIZE}.hdf5" "./out/restart_les_ref_${GRID_SIZE}.hdf5"

source "${project_folder}/setup_env_claix23.sh"

source "${project_folder}/CPP-ML-Interface/extern/python/venv/bin/activate"


if [[ "$fraction" == "1.00" || "$fraction" == "1" ]]; then
    echo "Running non-hybrid version because hostFraction is 1.00"
    # srun /home/thes1961/MAIA/build_interface_aix_scorep_23b/bin/maia ./"$(basename $TOML_FILE)"
    # srun --label /home/rwth0792/mmcp_2026_artifact/MAIA-TOM/m-AIA-Solver/build_gnu_production/bin/maia ./"$(basename $TOML_FILE)"
    srun --label ${project_folder}/maia/build_gnu_production/bin/maia ./"$(basename $TOML_FILE)"
    #mpirun -np 15 /home/thes1961/MAIA/build_interface_aix_scorep_23b/bin/maia ./"$(basename $TOML_FILE)"
else
    echo "Running hybrid version"
    # srun /home/thes1961/HybridInferenceCode/MAIA/build_interface_aix_scorep_23b_hybrid_batch/bin/maia ./"$(basename $TOML_FILE)"
    #srun --label /hpcwork/rwth1859/MMCP_2026_benchmarks/HybridInferenceMAIA/common/bin/maia-hybrid ./"$(basename $TOML_FILE)"
    #mpirun -np 15 /home/thes1961/HybridInferenceCode/MAIA/build_interface_aix_scorep_23b_hybrid_batch/bin/maia ./"$(basename $TOML_FILE)"
fi