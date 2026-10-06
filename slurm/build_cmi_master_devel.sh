#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --job-name=cmi-master-build
#SBATCH --output=logs/cmi-master-build_%j.out
#SBATCH --error=logs/cmi-master-build_%j.err

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
source "${root}/setup_env_claix23.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${cuda_lib}/stubs:${cuda_lib}:${LD_LIBRARY_PATH:-}"
cmi="${root}/CPP-ML-Interface"
cd "${cmi}"
# Build here; the MPMD runtime checks belong in the subsequent smoke allocation.
PHYDLL_TEST_TARGET=all bash -e build_phydll.sh
build="${root}/cmi-build-all-providers"
cmake -S "${cmi}" -B "${build}" \
    -DWITH_AIX=ON -DWITH_SMARTSIM=ON -DWITH_PHYDLL=ON -DWITH_SCOREP=OFF \
    -DAIX_USE_PREBUILT=ON -DAIX_SKIP_VENV_CREATION=ON \
    -DLIBTORCH_DIR="${cmi}/extern/libtorch" -DTORCH_VERSION=2.4.0 \
    -DBUILD_TESTS=OFF -DBUILD_TESTING=ON -DWITH_FORTRAN=ON \
    -DCPPML_RUN_REGISTRY_TESTS=ON \
    -DTEST_PYTHON_EXECUTABLE="/hpcwork/${USER}/smartsim/python/smartsim_cuda-12/bin/python3"
cmake --build "${build}" -j "${BUILD_JOBS:-8}"
ctest --test-dir "${build}" --output-on-failure -E '^test_provider_inference$'
# This test calls MPI_Init: a batch-shell child is not a valid Slurm MPI step.
srun --mpi=pmix --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-24}" \
    "${build}/test_provider_inference" "${build}/test_addition_model.pt"
maia_build="${MAIA_BUILD_DIR:-${root}/maia/build_gnu_production}"
cmake -S "${root}/maia" -B "${maia_build}" \
    -DWITH_SCOREP=OFF -DAIX_USE_PREBUILT=ON -DAIX_SKIP_VENV_CREATION=ON \
    -DLIBTORCH_DIR="${cmi}/extern/libtorch" -DTORCH_VERSION=2.4.0 \
    -DBUILD_TESTS=OFF -DBUILD_TESTING=OFF
cmake --build "${maia_build}" -j "${BUILD_JOBS:-8}"
printf 'MASTER_BUILD_PASS\n'
