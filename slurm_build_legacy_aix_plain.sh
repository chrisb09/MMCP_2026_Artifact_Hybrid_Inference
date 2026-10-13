#!/usr/bin/env bash
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=build-legacy-plain-240
#SBATCH --output=logs/build_legacy_plain.%j.out
#SBATCH --error=logs/build_legacy_plain.%j.err

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_folder}/setup_env_claix23.sh"
export CXXFLAGS="${CXXFLAGS:-} -DOUTPUT_FIELDS"

legacy_libtorch="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/libtorch"
mkdir -p "${project_folder}/CPP-ML-Interface/extern"
ln -sfn "${legacy_libtorch}" "${project_folder}/CPP-ML-Interface/extern/libtorch"

mkdir -p "${project_folder}/logs"

cpp_ml_root="${project_folder}/CPP-ML-Interface"

# Build PhyDLL C++
make -C "${cpp_ml_root}/extern/phydll" \
    BUILD="${cpp_ml_root}/extern/phydll/BUILD-SCOREP" ENABLE_PYTHON=OFF

# HighFive
cmake -S "${cpp_ml_root}/extern/HighFive" -B "${cpp_ml_root}/extern/HighFive/BUILD" \
    -DHIGHFIVE_UNIT_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/extern/HighFive/BUILD/INSTALL"
cmake --build "${cpp_ml_root}/extern/HighFive/BUILD" -j"${SLURM_CPUS_ON_NODE:-24}"
cmake --install "${cpp_ml_root}/extern/HighFive/BUILD"

# AIxeleratorService (Plain, no Score-P)
rm -rf "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP" "${cpp_ml_root}/extern/aixeleratorservice/INSTALL-SCOREP"
cmake -S "${cpp_ml_root}/extern/aixeleratorservice" -B "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP" \
    -DWITH_TORCH=ON -DTORCH_VERSION=2.4.0 \
    -DLIBTORCH_DIR="${legacy_libtorch}" \
    -DWITH_SCOREP=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/extern/aixeleratorservice/INSTALL-SCOREP"
cmake --build "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP" -j"${SLURM_CPUS_ON_NODE:-24}"
cmake --install "${cpp_ml_root}/extern/aixeleratorservice/BUILD-SCOREP"
ln -sfn INSTALL-SCOREP "${cpp_ml_root}/extern/aixeleratorservice/INSTALL"

# CPP-ML-Interface (Plain, no Score-P)
rm -rf "${cpp_ml_root}/BUILD-SCOREP"
cmake -S "${cpp_ml_root}" -B "${cpp_ml_root}/BUILD-SCOREP" \
    -DWITH_PHYDLL=OFF -DWITH_AIX=ON -DWITH_REFERENCE_MODEL=ON -DWITH_SCOREP=OFF \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/BUILD-SCOREP"
cmake --build "${cpp_ml_root}/BUILD-SCOREP" -j"${SLURM_CPUS_ON_NODE:-24}"
cmake --install "${cpp_ml_root}/BUILD-SCOREP"

# MAIA (Plain, no Score-P)
rm -rf "${project_folder}/maia/build_gnu_production"
(
    cd "${project_folder}/maia"
    ./configure.py 1 2
    cmake --build build_gnu_production -j"${SLURM_CPUS_ON_NODE:-24}"
)

echo "=== Legacy Plain Build Complete ==="
ls -lh "${project_folder}/maia/build_gnu_production/bin/maia"
ldd "${project_folder}/maia/build_gnu_production/bin/maia" | grep -E 'scorep|papi' || echo "Score-P free verified!"
