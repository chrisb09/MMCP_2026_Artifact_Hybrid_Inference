#!/bin/zsh

############################
# Global job options
############################
#SBATCH --account=p0025821
#SBATCH --time=00:05:00
#SBATCH --job-name=maia-hybrid-phydll-c23mm-c23g-x1
#SBATCH --exclusive
#SBATCH --output=logs/output_maia_%J.txt
#SBATCH --error=logs/error_maia_%J.txt

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
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
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

source "${project_folder}/setup_env_claix23.sh"

source "${project_folder}/CPP-ML-Interface/extern/python/venv/bin/activate"

# Force OpenMPI to use external pmix (required after SLURM update Jan 2026)
export OMPI_MCA_pmix=pmix3x
export OMPI_MCA_ess=pmi

############ Copied from Tom's script ############

cp phydll-tom/hetjob_maia.sh ./
cp phydll-tom/hetjob_dl.sh ./
chmod +x hetjob_maia.sh
chmod +x hetjob_dl.sh


# Before the ln commands - clean up any existing symlinks
rm -f ./grid_les_$GRID_SIZE.hdf5 ./properties_run_les_ref_$GRID_SIZE.toml ./restart_les_init_$GRID_SIZE.hdf5 ./out/restart_les_ref_${GRID_SIZE}.hdf5

# Create symlinks to your own files
ln -s ${toml_folder}/grid_les_${GRID_SIZE}.hdf5 ./grid_les_${GRID_SIZE}.hdf5
ln -s ${toml_folder}/properties_run_les_ref_${GRID_SIZE}.toml ./properties_run_les_ref_${GRID_SIZE}.toml
ln -s ${toml_folder}/restart_les_init_${GRID_SIZE}.hdf5 ./restart_les_init_${GRID_SIZE}.hdf5
ln -s ${toml_folder}/restart_les_init_${GRID_SIZE}.hdf5 ./out/restart_les_ref_${GRID_SIZE}.hdf5


##################################################


if [[ "$fraction" == "1.00" || "$fraction" == "1" ]]; then
    echo "Running non-hybrid version because hostFraction is 1.00"
    # Single unified srun - NO separate --output per component!
    srun --output=logs/output_maia-phydll_%J.txt --mpi=pmix --export=ALL \
        --het-group=0 --cpus-per-task=1 --ntasks-per-node=96 ./hetjob_maia.sh "${project_folder}/maia/build_gnu_production/bin/maia" "./$(basename $TOML_FILE)" \
        : --export=ALL --mpi=pmix --het-group=1 --ntasks=4 --cpus-per-task=24 ./hetjob_dl.sh
else
    echo "Running hybrid version"
    #srun --export=ALL --het-group=0 --mpi=pmix --preserve-env --cpus-per-task=1 ${project_folder}/maia/build_gnu_production/bin/maia ./"$(basename $TOML_FILE)" : --export=ALL --het-group=1 --mpi=pmix --preserve-env --cpus-per-task=1 ${project_folder}/maia/build_gnu_production/bin/maia ./"$(basename $TOML_FILE)"
    echo "Currently no hybrid version available, running non-hybrid instead"
fi

############### Cleanup ############

rm -f ./properties_run_les_ref_$GRID_SIZE.toml
rm -f ./grid_les_$GRID_SIZE.hdf5
rm -f ./restart_les_init_$GRID_SIZE.hdf5
rm -f ./out/restart_les_ref_$GRID_SIZE.hdf5
rm -f ./hetjob_maia.sh
rm -f ./hetjob_dl.sh