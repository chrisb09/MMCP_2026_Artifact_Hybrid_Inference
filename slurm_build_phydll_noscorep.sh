#!/bin/zsh
#SBATCH --job-name=build-phydll-noscorep
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=logs/build_phydll_noscorep_%j.out
#SBATCH --error=logs/build_phydll_noscorep_%j.err

set -euxo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${project_folder}/setup_env_claix23.sh"

export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"
maia_build_dir="${project_folder}/maia/build_gnu_production_phydll"

NPROC=${SLURM_CPUS_ON_NODE:-96}

# Build CMI for PhyDLL
cpp_ml_root="${project_folder}/CPP-ML-Interface"
cmake -S "${cpp_ml_root}" -B "${cpp_ml_root}/BUILD-PHYDLL" \
    -DWITH_PHYDLL=ON -DWITH_AIX=OFF -DWITH_SCOREP=OFF \
    -DPHYDLL_BUILD_DIR="${cpp_ml_root}/extern/phydll/build" \
    -DCMAKE_INSTALL_PREFIX="${cpp_ml_root}/BUILD-PHYDLL"
cmake --build "${cpp_ml_root}/BUILD-PHYDLL" -j${NPROC}
cmake --install "${cpp_ml_root}/BUILD-PHYDLL"

cd "${project_folder}/maia"
rm -rf build_gnu_production build_gnu_production_phydll
./configure.py 1 2 --disable-updateGitSubmodules
cmake --build build_gnu_production -j${NPROC}
mkdir -p build_gnu_production_phydll/bin
cp -f build_gnu_production/bin/maia build_gnu_production_phydll/bin/maia

echo "=== Done building non-ScoreP PhyDLL MAIA ==="
ls -lh "${maia_build_dir}/bin/maia"
