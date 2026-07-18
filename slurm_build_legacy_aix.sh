#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=build-legacy-aix-debug
#SBATCH --output=logs/build_legacy_aix.%j.out
#SBATCH --error=logs/build_legacy_aix.%j.err

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_folder}/setup_env_claix23.sh"

# Reuse the artifact's CUDA 12.4 LibTorch instead of downloading one.
legacy_libtorch="${project_folder}/CPP-ML-Interface/extern/libtorch"
if [[ ! -e "${legacy_libtorch}" ]]; then
    ln -s "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/CPP-ML-Interface/extern/libtorch" "${legacy_libtorch}"
fi

mkdir -p "${project_folder}/logs"

(
    cd "${project_folder}/CPP-ML-Interface"
    ./install-scorep.sh
)

(
    cd "${project_folder}/maia"
    ./configure.py 1 2 --enable-instrumentation scorep --instrument mpi --instrument user
    cmake --build build_gnu_production -j"${SLURM_CPUS_ON_NODE:-96}"
)
