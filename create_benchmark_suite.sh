#!/bin/bash
set -e 


if [ -z "$1" ]; then
    echo "ERROR: Please provide Slurm Account as parameter!"
    echo "Usage: ./create_benchmark_suite.sh <ACCOUNT>"
    echo "Example:   ./create_benchmark_suite.sh rwth0792"
    exit 1
fi
SLURM_ACCOUNT="$1"


echo "--- [Phase 1] Starting Benchmark Suite Creation ---"

# 1. Paths & Configuration
export REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VENV_DIR="$REPO_ROOT/.venv"
PREP_DIR="$REPO_ROOT/Templates" # Fetch clean data here
BASE_DIR="$REPO_ROOT/data"
MATERIAL_DIR="$REPO_ROOT/input/MMCP_2026_Material"
VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"
INPUT_DIR="$REPO_ROOT/input"

# =========================================================
# VENV CONFIGURATION
# =========================================================

echo "-> Checking Python Environment (.venv)..."

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found. Creating new one in $VENV_DIR ..."
    
    # 1. Create
    python3 -m venv "$VENV_DIR"
    
    # 2. Activate
    source "$VENV_DIR/bin/activate"
    
    # 3. Pip Upgrade & Dependencies
    echo "Installing dependencies..."
    pip install -r "$REPO_ROOT/requirements.txt"   
    # Adjust your package list here:
    pip install numpy matplotlib h5py torch
    
    echo "Venv successfully created and packages installed."
else
    echo "Venv already exists in $VENV_DIR. Skipping creation."
fi

# =========================================================
# RATIO CONFIGURATION (DICTIONARY STYLE)
# =========================================================

# 1. The base ratios that EVERY config gets (0.00 to 1.00 in 0.1 steps)
BASE_RATIOS=("0.00" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.00")

# 2. Associative array for specific "zoom-in" ranges
# Syntax: declare -A NAME
declare -A EXTRA_RATIOS

# Here you define the additional ratios as a string (space-separated) for each key (Config-Name)
EXTRA_RATIOS["1CPUNodes1GPU"]="0.19 0.21 0.22 0.23 0.24 0.25 0.26"
EXTRA_RATIOS["1CPUNodes4GPU"]="0.03 0.04 0.05 0.06 0.07 0.08 0.09"
EXTRA_RATIOS["2CPUNodes1GPU"]="0.36 0.37 0.38 0.39"
EXTRA_RATIOS["2CPUNodes4GPU"]="0.12 0.13 0.14 0.15 0.16"
EXTRA_RATIOS["4CPUNodes1GPU"]="0.51 0.52 0.53 0.54 0.55"
EXTRA_RATIOS["4CPUNodes4GPU"]="0.21 0.22 0.23 0.24 0.25"
EXTRA_RATIOS["8CPUNodes1GPU"]="0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77"
EXTRA_RATIOS["8CPUNodes4GPU"]="0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39"

# If you have others for different configs, simply add them:
# EXTRA_RATIOS["2CPUNodes1GPU"]="0.xx 0.yy ..."
# If a config is not listed here, it only gets the base ratios.

# Your Configs List
CONFIGS=("1CPUNodes1GPU" "1CPUNodes4GPU" "2CPUNodes1GPU" "2CPUNodes4GPU" "4CPUNodes1GPU" "4CPUNodes4GPU" "8CPUNodes1GPU" "8CPUNodes4GPU")


# =========================================================
# PHASE -1: MATERIAL CHECK & DOWNLOAD
# =========================================================
echo "-> [Phase -1] Checking input material..."

# The 3 files we expect
REQUIRED_FILES=("grid_les_medium.hdf5" "restart_les_init_medium.hdf5" "transformer_inference_scripted_fw2.pt")
MISSING_FILES=false

# 1. Check if folder exists
if [ ! -d "$MATERIAL_DIR" ]; then
    echo "Folder $MATERIAL_DIR is missing."
    MISSING_FILES=true
else
    # 2. Check if all files are inside
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$MATERIAL_DIR/$file" ]; then
            echo "File missing: $file"
            MISSING_FILES=true
        fi
    done
fi

# 3. If something is missing -> Download & Extract
if [ "$MISSING_FILES" = true ]; then
    echo "Material incomplete. Starting download from Sciebo..."
    
    # Ensure input folder exists
    mkdir -p "$INPUT_DIR"
    
    # Download (save as temp.zip)
    # -nv: non-verbose (less text), -O: Output file
    wget -nv -O "$INPUT_DIR/temp_material.zip" "https://rwth-aachen.sciebo.de/s/yxSDGncnbjcA4Tk/download"
    
    if [ $? -ne 0 ]; then
        echo "Download error! Please check link or internet connection."
        exit 1
    fi
    
    echo "Download finished. Extracting..."
    
    # Extract (-q: quiet, -o: overwrite, -d: destination)
    # Sciebo Zips usually contain the folder itself, so we extract into INPUT_DIR
    unzip -q -o "$INPUT_DIR/temp_material.zip" -d "$INPUT_DIR"
    
    # Clean up
    rm "$INPUT_DIR/temp_material.zip"
    
    echo "Material successfully provided in: $MATERIAL_DIR"
else
    echo "Material is fully present. Skipping download."
fi


# Create structure and distribute
mkdir -p "$BASE_DIR"
cp "$PREP_DIR/run_all_experiments.sh" "$BASE_DIR/"
cp "$PREP_DIR/run_all_plotting.sh" "$BASE_DIR/"
cp "$PREP_DIR/calc_single_point_splits.py" "$BASE_DIR/"
cp "$PREP_DIR/calc_splits.py" "$BASE_DIR/"
cp "$PREP_DIR/plotting.py" "$BASE_DIR/"
cp "$PREP_DIR/summary_plots.py" "$BASE_DIR/"
cp "$PREP_DIR/summary_print_speedup.py" "$BASE_DIR/"
for config in "${CONFIGS[@]}"; do
    CONFIG_DIR="$BASE_DIR/$config"

    NODE_COUNT=${config%%CPUNodes*}
    TEMP_SUFFIX=${config##*Nodes}
    GPU_COUNT=${TEMP_SUFFIX%%GPU*}

    echo "Processing Config: $config"
    
    mkdir -p "$CONFIG_DIR/Experiments"
    mkdir -p "$CONFIG_DIR/plots"
    
    # Copy base files from PREP_DIR
    cp "$PREP_DIR/cpu.job" "$CONFIG_DIR/"
    cp "$PREP_DIR/run_step.sh" "$CONFIG_DIR/"
    cp "$REPO_ROOT/input/properties_run_les_ref_medium.toml" "$CONFIG_DIR/"

    if [ "$NODE_COUNT" -eq 1 ]; then 
        
        CPU_PART_NODES=0

        cp "$PREP_DIR/Hybrid.job" "$CONFIG_DIR/"
        # Edit Hybrid.job
        # 1. Set GPU count
        sed -i "s/^GPU_COUNT=4*/GPU_COUNT=$GPU_COUNT/" "$CONFIG_DIR/Hybrid.job"
        # 2. Ensure nodes (is 1 anyway, but just to be safe)
        sed -i "s/^#SBATCH --nodes=.*/#SBATCH --nodes=1/" "$CONFIG_DIR/Hybrid.job"
        sed -i "s/^#SBATCH --account=.*/#SBATCH --account=$SLURM_ACCOUNT/" "$CONFIG_DIR/Hybrid.job"

    else 
        
        CPU_PART_NODES=$((NODE_COUNT - 1))
        cp "$PREP_DIR/Hybrid_2cpu.job" "$CONFIG_DIR/"
        TARGET_JOB="$CONFIG_DIR/Hybrid_2cpu.job"
        sed -i "/^#SBATCH --partition=c23mm/{n;s/^#SBATCH --nodes=.*/#SBATCH --nodes=$CPU_PART_NODES/;}" "$TARGET_JOB"
        sed -i "s/^GPU_COUNT=4*/GPU_COUNT=$GPU_COUNT/" "$TARGET_JOB"
        sed -i "s/^#SBATCH --account=.*/#SBATCH --account=$SLURM_ACCOUNT/" "$TARGET_JOB"
        mv "$TARGET_JOB" "$CONFIG_DIR/Hybrid.job"

    fi

    echo "  Config: $config | Nodes: $NODE_COUNT | GPUs: $GPU_COUNT | CPU-Part: $CPU_PART_NODES"

    sed -i "s/^#SBATCH --nodes=.*/#SBATCH --nodes=$NODE_COUNT/" "$CONFIG_DIR/cpu.job"
    sed -i "s/^#SBATCH --account=.*/#SBATCH --account=$SLURM_ACCOUNT/" "$CONFIG_DIR/cpu.job"




    for file in "$CONFIG_DIR"/*.job; do
        LAST_SBATCH_LINE=$(grep -n "^#SBATCH" "$file" | tail -1 | cut -d: -f1)
        sed -i "${LAST_SBATCH_LINE} a REPO_DIR=\"$REPO_ROOT\"" "$file"
        echo "$file"
    done
    
    # Creates a temporary array for this loop
    CURRENT_RATIOS=( "${BASE_RATIOS[@]}" ${EXTRA_RATIOS[$config]} )

    echo "Config: $config - Generating ${#CURRENT_RATIOS[@]} experiments..."

    # --- EXPERIMENT LOOPS ---
    for ratio in "${CURRENT_RATIOS[@]}"; do
        EXP_DIR="$CONFIG_DIR/Experiments/$ratio"
        mkdir -p "$EXP_DIR"
        
    done
done

sed -i "/^#!\/bin\/bash/a VENV_ACTIVATE=\"$VENV_ACTIVATE\"" "$BASE_DIR/run_all_plotting.sh"

echo "--- [Phase 1] Done! Structure created in '$BASE_DIR'. ---"