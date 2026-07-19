#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=plot-provider-cpu-100
#SBATCH --output=logs/plot_provider_cpu_100_%j.out
#SBATCH --error=logs/plot_provider_cpu_100_%j.err

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_folder}/setup_env_claix23.sh"

python "${project_folder}/analysis/plot_provider_differences.py" \
    --legacy "${project_folder}/scratch/legacy_prepost/scratch/legacy_aix_cpu_2059493/global_snapshots.h5" \
    --provider "AIX=${project_folder}/scratch/current_aix_cpu_2059538/debug/global_snapshots.h5" \
    --provider "SmartSim=${project_folder}/debug_dumps/smartsim/global_2059610.h5" \
    --provider "PhyDLL-C++=${project_folder}/debug_dumps/phydll_cpp/global_2059683.h5" \
    --csv "${project_folder}/analysis/provider_cpu_100_global_differences.csv" \
    --output "${project_folder}/analysis/provider_cpu_100_global_differences.png"
