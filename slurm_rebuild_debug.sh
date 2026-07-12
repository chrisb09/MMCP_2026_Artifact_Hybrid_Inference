#!/usr/bin/zsh
#SBATCH --job-name=rebuild-debug
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=rebuild_debug_%j.txt
set -euxo pipefail
source setup_env_claix23.sh
export CPP_ML_INTERFACE_PROVIDER_ENV="${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}"
provider_suffix="${(L)CPP_ML_INTERFACE_PROVIDER_ENV}"
maia_build_dir="${PWD}/maia/build_gnu_production_${provider_suffix}"

# libclang for registry generation (if generated_registry.hpp needs regeneration)
pip install "clang==17.0.6" "libclang==17.0.6" 2>&1
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"

NPROC=${SLURM_CPUS_ON_NODE:-96}

# CMI is built in-tree by MAIA via add_subdirectory.
# There is NO separate CMI library linked into MAIA.
# Therefore we only need to add -DFLOW_DUMP_DEBUG to the existing
# maia/build_gnu_production_<provider> cmake cache — the flag propagates to CMI
# sources automatically because the child project inherits parent CMAKE_CXX_FLAGS.
echo "=== Configuring ${maia_build_dir} with FLOW_DUMP_DEBUG ==="
cd "${maia_build_dir}"
CURRENT_FLAGS=$(cmake -LA . 2>/dev/null | grep "^CMAKE_CXX_FLAGS:STRING=" | sed 's/^CMAKE_CXX_FLAGS:STRING=//')
cmake . -DCMAKE_CXX_FLAGS:STRING="${CURRENT_FLAGS} -Wno-array-bounds -DFLOW_DUMP_DEBUG"

echo "=== Building MAIA (incremental, only files affected by header change) ==="
cmake --build "${maia_build_dir}" -j${NPROC}

echo "=== Done ==="
ls -lh "${maia_build_dir}/bin/maia"
