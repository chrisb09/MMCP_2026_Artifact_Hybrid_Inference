#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=legacy-aix-debug
#SBATCH --output=logs/legacy_aix_%j.out
#SBATCH --error=logs/legacy_aix_%j.err
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-20}"
run_dir="${project_folder}/scratch/legacy_aix_${SLURM_JOB_ID}"
maia_bin="${project_folder}/maia/build_gnu_production/bin/maia"

source "${project_folder}/setup_env_claix23.sh"
[[ -x "${maia_bin}" ]] || { echo "Legacy MAIA binary not found: ${maia_bin}" >&2; exit 1; }

mkdir -p "${run_dir}/out" "${run_dir}/debug"
ln -s "${project_folder}/input" "${run_dir}/input"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export MLCOUPLING_DEBUG_EXPORT=1
export MLCOUPLING_DEBUG_EXPORT_DIR="${run_dir}/debug"
export MLCOUPLING_DEBUG_RANK=0
export MLCOUPLING_DEBUG_MAX_INFERENCES=1
cuda_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${cuda_lib}:${project_folder}/CPP-ML-Interface/BUILD-SCOREP/lib:${project_folder}/CPP-ML-Interface/extern/aixeleratorservice/INSTALL-SCOREP/lib:${project_folder}/CPP-ML-Interface/extern/phydll/BUILD-SCOREP/lib:${LD_LIBRARY_PATH:-}"
export SCOREP_ENABLE_TRACING=false
export SCOREP_ENABLE_PROFILING=true

cd "${run_dir}"
srun --label --mpi=pmix --export=ALL \
    --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
        /bin/bash -c 'export LD_LIBRARY_PATH="'"${cuda_lib}"':${LD_LIBRARY_PATH:-}"; exec "'"${maia_bin}"'" ./properties.toml' : \
    --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
        /bin/bash -c 'export LD_LIBRARY_PATH="'"${cuda_lib}"':${LD_LIBRARY_PATH:-}"; exec "'"${maia_bin}"'" ./properties.toml'
