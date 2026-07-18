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

cpp_ml_root="${project_folder}/CPP-ML-Interface"
scorep_flags="--nocompiler --user --mpp=mpi --thread=none --nocuda"

# The historical installer builds a Python 3.11 mpi4py environment although
# this AIX-only diagnosis uses no Python API. Build only the C++ dependencies.
make -C "${cpp_ml_root}/extern/phydll" \
    BUILD="${cpp_ml_root}/extern/phydll/BUILD-SCOREP" ENABLE_PYTHON=OFF

cmake -S "${cpp_ml_root}/extern/HighFive" -B "${cpp_ml_root}/extern/HighFive/BUILD" \
    -DHIGHFIVE_UNIT_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/extern/HighFive/BUILD/INSTALL"
cmake --build "${cpp_ml_root}/extern/HighFive/BUILD" -j"${SLURM_CPUS_ON_NODE:-96}"
cmake --install "${cpp_ml_root}/extern/HighFive/BUILD"

SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${scorep_flags}" \
cmake -S "${cpp_ml_root}/extern/aixeleratorservice" -B "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP" \
    -DWITH_TORCH=ON -DTORCH_VERSION=2.6.0 \
    -DCMAKE_C_COMPILER=scorep-mpicc -DCMAKE_CXX_COMPILER=scorep-mpicxx \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/extern/aixeleratorservice/INSTALL-SCOREP"
cmake --build "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP" -j"${SLURM_CPUS_ON_NODE:-96}"
cmake --install "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP"

SCOREP_WRAPPER_INSTRUMENTER_FLAGS="${scorep_flags}" \
cmake -S "${cpp_ml_root}" -B "${cpp_ml_root}/BUILD-SCOREP" \
    -DWITH_PHYDLL=OFF -DWITH_AIX=ON -DWITH_REFERENCE_MODEL=ON -DWITH_SCOREP=ON \
    -DCMAKE_C_COMPILER=scorep-mpicc -DCMAKE_CXX_COMPILER=scorep-mpicxx \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/BUILD-SCOREP"
cmake --build "${cpp_ml_root}/BUILD-SCOREP" -j"${SLURM_CPUS_ON_NODE:-96}"
cmake --install "${cpp_ml_root}/BUILD-SCOREP"

(
    cd "${project_folder}/maia"
    ./configure.py 1 2 --enable-instrumentation scorep --instrument mpi --instrument user
    cmake --build build_gnu_production -j"${SLURM_CPUS_ON_NODE:-96}"
)
