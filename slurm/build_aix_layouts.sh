#!/usr/bin/env bash
# Build the AIX GPU-affine and NUMA-local-group distribution variants used by
# the [TBL] coupling benchmark layout matrix:
#   default : round-robin                      -> INSTALL (already present)
#   affine  : WITH_HWLOC=ON                    -> INSTALL-AFFINE
#   numa    : WITH_HWLOC=ON + NUMA_LOCAL_GROUPS -> INSTALL-NUMA
#
# Plain (no Score-P) variants. Run on the devel partition.
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=build-aix-layouts
#SBATCH --output=logs/build-aix-layouts.%j.out
#SBATCH --error=logs/build-aix-layouts.%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
cd "${REPO_DIR}"
source "${REPO_DIR}/CPP-ML-Interface/set_env_claix23_cuda12.4.sh"
export CC=gcc CXX=g++

CPP="${REPO_DIR}/CPP-ML-Interface"
AIX_DIR="${CPP}/extern/AIxeleratorService"
LIBTORCH_DIR="${SMARTSIM_LIBTORCH_DIR:-${CPP}/extern/libtorch}"
AIX_VENV_DIR="${SMARTSIM_AIX_VENV_DIR:-${CPP}/extern/python/smartsim_cuda-12}"
TORCH_VERSION="${SMARTSIM_TORCH_VERSION:-2.4.0}"
JOBS="${SLURM_CPUS_ON_NODE:-32}"
TMP="${CPP}/tmp/opencode"

HWLOC_ROOT="${HWLOC_ROOT:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/hwloc/2.9.2-GCCcore-13.2.0}"
NUMA_ROOT="${NUMA_ROOT:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/numactl/2.0.16-GCCcore-13.2.0}"

build_layout() {
    local variant="$1" with_hwloc="$2" numa_local="$3" install_dir="$4"
    local build_dir="${TMP}/aix-${variant}-build"
    echo "=== Building AIX ${variant}: WITH_HWLOC=${with_hwloc} WITH_NUMA_LOCAL_GROUPS=${numa_local} -> ${install_dir}"
    rm -rf "${build_dir}" "${install_dir}"
    mkdir -p "${build_dir}"
    cmake -S "${AIX_DIR}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${install_dir}" \
        -DBUILD_TESTS=OFF \
        -DWITH_TORCH=ON \
        -DTORCH_VERSION="${TORCH_VERSION}" \
        -DLIBTORCH_DIR="${LIBTORCH_DIR}" \
        -DAIX_VENV_DIR="${AIX_VENV_DIR}" \
        -DAIX_SKIP_VENV_CREATION=ON \
        -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF \
        -DWITH_SCOREP=OFF \
        -DWITH_HWLOC="${with_hwloc}" \
        -DWITH_NUMA_LOCAL_GROUPS="${numa_local}" \
        -DHWLOC_ROOT="${HWLOC_ROOT}" \
        -DNUMA_ROOT="${NUMA_ROOT}"
    cmake --build "${build_dir}" -j "${JOBS}"
    cmake --install "${build_dir}"
    git -C "${AIX_DIR}" rev-parse HEAD > "${install_dir}/.git_commit"
    [[ -f "${install_dir}/lib/libAIxeleratorService.so" ]] || { echo "Missing built library in ${install_dir}" >&2; exit 1; }
    echo "AIX ${variant} install ready at ${install_dir} ($(cat "${install_dir}/.git_commit"))"
}

build_layout affine ON OFF "${AIX_DIR}/INSTALL-AFFINE"
build_layout numa ON ON "${AIX_DIR}/INSTALL-NUMA"

echo "AIX_LAYOUT_BUILD_DONE"
