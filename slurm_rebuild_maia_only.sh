#!/usr/bin/zsh
#SBATCH --job-name=rebuild-maia
#SBATCH --partition=devel
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=rebuild_maia_%j.txt
set -euxo pipefail
source setup_env_claix23.sh
maia_build_dir="${PWD}/maia/build_gnu_production_cmi"
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"
NPROC=${SLURM_CPUS_ON_NODE:-96}
cd "${maia_build_dir}"
cmake . -DCMAKE_CXX_FLAGS:STRING="-Wno-array-bounds -DFLOW_DUMP_DEBUG"
cmake --build "${maia_build_dir}" -j${NPROC}
echo "=== Done ==="
ls -lh "${maia_build_dir}/bin/maia"
