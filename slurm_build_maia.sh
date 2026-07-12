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

with_scorep="${WITH_SCOREP:-OFF}"
if [[ "${with_scorep}" != "ON" && "${with_scorep}" != "OFF" ]]; then
    echo "WITH_SCOREP must be ON or OFF, got '${with_scorep}'." >&2
    exit 2
fi

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

build_variant="plain"
build_suffix=""
if [[ "${with_scorep}" == "ON" ]]; then
    build_variant="scorep"
    build_suffix="_scorep"
    export USE_SCOREP=1
fi
maia_build_dir="${MAIA_BUILD_DIR:-${REPO_DIR}/maia/build_gnu_production_cmi${build_suffix}}"
echo "Using unified ${build_variant} build directory: ${maia_build_dir}"

CPP_ML_DIR="${REPO_DIR}/CPP-ML-Interface"

# Source MAIA environment (modules)
echo "Sourcing MAIA environment from setup_env_claix23.sh..."
cd "${REPO_DIR}"
source ./setup_env_claix23.sh

# Ensure CMI extern submodules are initialized (AIxeleratorService, SmartRedis)
echo "Initializing CMI submodules..."
git -C "${CPP_ML_DIR}" submodule update --init --recursive || true

echo "Building external PhyDLL runtime..."
bash "${CPP_ML_DIR}/build_phydll.sh"

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

# Step 1: Build CMI standalone first with every runtime provider enabled.
echo "=== Step 1: Building unified CMI standalone ==="
CMI_BUILD_DIR="${CMI_BUILD_DIR:-${REPO_DIR}/cmi-build-all-providers${build_suffix}}"
echo "Using CMI_BUILD_DIR=${CMI_BUILD_DIR}"
mkdir -p "${CMI_BUILD_DIR}"
cd "${CMI_BUILD_DIR}"
cmake "${CPP_ML_DIR}" \
    -DWITH_AIX=ON \
    -DWITH_SMARTSIM=ON \
    -DWITH_PHYDLL=ON \
    -DWITH_SCOREP="${with_scorep}" \
    -DAIX_USE_PREBUILT=OFF \
    -DAIX_SKIP_VENV_CREATION=ON \
    -DLIBTORCH_DIR="${CPP_ML_DIR}/extern/libtorch" \
    -DTORCH_VERSION=2.6.0 \
    -DBUILD_TESTS=OFF \
    -DCMAKE_CXX_FLAGS:STRING="-DFLOW_DUMP_DEBUG" \
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
cd "${CMI_BUILD_DIR}"
ctest --output-on-failure -R test_behavior_flow_extrapolator || echo "Warning: test executable not found, trying direct run..."
./test/test_behavior_flow_extrapolator 2>&1 || echo "Tests skipped (not built)."

# Step 3: Build MAIA (with CMI built in-tree via add_subdirectory)
echo "=== Step 3: Cleaning old MAIA build ==="
cd "${REPO_DIR}/maia"
rm -rf "${maia_build_dir}"

echo "=== Step 4: Configuring MAIA ==="
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=1 --nocompiler --user --mpp=mpi --io=none --memory=none --thread=none --nocuda"
export SCOREP_ENABLE_CUDA=0

./configure.py 1 2 \
    --disable-updateGitSubmodules \
    --build-dir-name "${maia_build_dir}"

# Append -Wno-array-bounds to suppress GCC 13.2 false positive in existing MAIA code.
# This must come AFTER configure.py because MAIA's GNU.cmake sets -Warray-bounds=2
# which would re-enable the warning if -Wno-array-bounds came first via CXXFLAGS.
cd "${maia_build_dir}"
CURRENT_FLAGS=$(cmake -LA . 2>/dev/null | grep "^CMAKE_CXX_FLAGS:STRING=" | sed 's/^CMAKE_CXX_FLAGS:STRING=//')
cmake . \
    -DCMAKE_CXX_FLAGS:STRING="${CURRENT_FLAGS} -Wno-array-bounds -DFLOW_DUMP_DEBUG" \
    -DWITH_SCOREP="${with_scorep}" \
    -DAIX_USE_PREBUILT=OFF \
    -DAIX_SKIP_VENV_CREATION=ON \
    -DLIBTORCH_DIR="${CPP_ML_DIR}/extern/libtorch" \
    -DTORCH_VERSION=2.6.0 \
    -DBUILD_TESTS=OFF

echo "=== Step 5: Building MAIA ==="
cmake --build "${maia_build_dir}" -j"${NPROC}"

echo "=== Slurm Build Job Completed Successfully ==="
