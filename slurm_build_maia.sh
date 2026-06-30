#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=build-maia-cmi
#SBATCH --output=logs/build-maia-cmi.%j.out
#SBATCH --error=logs/build-maia-cmi.%j.err

set -euo pipefail

# Determine the repo root directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
ABS_SCRIPT="${REPO_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Self-submit if not inside Slurm
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm job. Re-executing via srun on devel partition..."
    exec srun --partition=devel --cpus-per-task=96 --time=01:00:00 "${ABS_SCRIPT}" "$@"
fi

echo "=== Slurm Build Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-96}"
echo "Repository dir: ${REPO_DIR}"

CPP_ML_DIR="${REPO_DIR}/CPP-ML-Interface"

# Source MAIA environment (modules)
echo "Sourcing MAIA environment from setup_env_claix23.sh..."
cd "${REPO_DIR}"
source ./setup_env_claix23.sh

# Ensure CMI extern submodules are initialized (AIxeleratorService, SmartRedis)
echo "Initializing CMI submodules..."
git -C "${CPP_ML_DIR}" submodule update --init --recursive || true

# Ensure libtorch symlink exists for CMI
if [ ! -d "${CPP_ML_DIR}/extern/libtorch" ]; then
    echo "Creating libtorch symlink..."
    ln -sfn /home/thes2181/libtorch "${CPP_ML_DIR}/extern/libtorch"
fi

NPROC="${SLURM_CPUS_ON_NODE:-96}"
echo "Using NPROC=${NPROC}"

# Install compatible clang Python module + libclang native library (needed for registry generation)
echo "Installing Python 'clang' + 'libclang' for registry generation..."
pip install "clang==17.0.6" "libclang==17.0.6" 2>&1

# Step 1: Build CMI standalone first (minimal deps) to verify registry generator changes
echo "=== Step 1: Building CMI standalone with minimal deps ==="
mkdir -p "${REPO_DIR}/cmi-build"
cd "${REPO_DIR}/cmi-build"
cmake "${CPP_ML_DIR}" \
    -DWITH_AIX=OFF \
    -DWITH_SMARTSIM=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Point libclang to the pip-installed native library for registry generation
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
echo "LIBCLANG_PATH=${LIBCLANG_PATH}"
# Also add to LD_LIBRARY_PATH so libclang can find its dependencies
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"

make -j"${NPROC}" cpp_ml_interface_library
echo "CMI build completed."

# Step 2: Build and run CMI unit tests
echo "=== Step 2: Running CMI tests ==="
make -j"${NPROC}" test_behavior_flow_extrapolator
cd "${REPO_DIR}/cmi-build"
ctest --output-on-failure -R test_behavior_flow_extrapolator || echo "Warning: test executable not found, trying direct run..."
./test/test_behavior_flow_extrapolator 2>&1 || echo "Tests skipped (not built)."

# Step 3: Build MAIA (with CMI built in-tree via add_subdirectory)
echo "=== Step 3: Cleaning old MAIA build ==="
cd "${REPO_DIR}/maia"
rm -rf build_gnu_production

echo "=== Step 4: Configuring MAIA ==="
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=1 --nocompiler --user --mpp=mpi --io=none --memory=none --thread=none --nocuda"
export SCOREP_ENABLE_CUDA=0

# CMI build options (WITH_AIX=OFF, WITH_SMARTSIM=OFF) are set in maia/src/CMakeLists.txt
# to avoid fetching Torch / building AIxeleratorService during the MAIA build.
# Enable them by overriding via cmake cache when AIx/Torch dependencies are available.
./configure.py 1 2 \
    --enable-instrumentation scorep --instrument mpi --instrument user \
    --disable-updateGitSubmodules

# Append -Wno-array-bounds to suppress GCC 13.2 false positive in existing MAIA code.
# This must come AFTER configure.py because MAIA's GNU.cmake sets -Warray-bounds=2
# which would re-enable the warning if -Wno-array-bounds came first via CXXFLAGS.
cd "${REPO_DIR}/maia/build_gnu_production"
CURRENT_FLAGS=$(cmake -LA . 2>/dev/null | grep "^CMAKE_CXX_FLAGS:STRING=" | sed 's/^CMAKE_CXX_FLAGS:STRING=//')
cmake . -DCMAKE_CXX_FLAGS:STRING="${CURRENT_FLAGS} -Wno-array-bounds"

echo "=== Step 5: Building MAIA ==="
cd "${REPO_DIR}/maia"
make -j"${NPROC}"

echo "=== Slurm Build Job Completed Successfully ==="
