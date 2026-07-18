#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:30:00
#SBATCH --job-name=legacy-aix-cpu-debug
#SBATCH --output=logs/legacy_aix_cpu_%j.out
#SBATCH --error=logs/legacy_aix_cpu_%j.err

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
artifact_input="${ARTIFACT_INPUT_DIR:-/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input}"
run_steps="${RUN_STEPS:-20}"
run_dir="${project_folder}/scratch/legacy_aix_cpu_${SLURM_JOB_ID}"
maia_bin="${project_folder}/maia/build_gnu_production/bin/maia"

source "${project_folder}/setup_env_claix23.sh"
[[ -x "${maia_bin}" ]] || { echo "Legacy MAIA binary not found: ${maia_bin}" >&2; exit 1; }

mkdir -p "${run_dir}/out" "${run_dir}/debug"
ln -s "${artifact_input}" "${run_dir}/input"
cp "${artifact_input}/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^mlInterval *=.*/mlInterval = 5/' "${run_dir}/properties.toml"
sed -i 's/^mlInputLength *=.*/mlInputLength = 5/' "${run_dir}/properties.toml"
sed -i 's/^mlStepCoefficient *=.*/mlStepCoefficient = 12/' "${run_dir}/properties.toml"
sed -i 's/^mlForecastWindow *=.*/mlForecastWindow = 2/' "${run_dir}/properties.toml"
sed -i 's/^mlInputStepDistance *=.*/mlInputStepDistance = 1/' "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${run_dir}/properties.toml"
ln -s "${artifact_input}/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${artifact_input}/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export MLCOUPLING_DEBUG_EXPORT=1
export MLCOUPLING_DEBUG_ALL_RANKS="${MLCOUPLING_DEBUG_ALL_RANKS:-1}"
export MLCOUPLING_DEBUG_EXPORT_DIR="${run_dir}/debug"
export MLCOUPLING_DEBUG_RANK=0
export MLCOUPLING_DEBUG_MAX_INFERENCES=1
export SCOREP_ENABLE_TRACING=false
export SCOREP_ENABLE_PROFILING=true
cuda_lib="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/lib"
cuda_stubs="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
export LD_LIBRARY_PATH="${cuda_lib}:${cuda_stubs}:${project_folder}/CPP-ML-Interface/BUILD-SCOREP/lib:${project_folder}/CPP-ML-Interface/extern/aixeleratorservice/INSTALL-SCOREP/lib:${LD_LIBRARY_PATH:-}"

cd "${run_dir}"
srun --label --mpi=pmix --export=ALL --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
    /bin/bash -c 'export SCOREP_EXPERIMENT_DIRECTORY="'"${run_dir}"'/scorep_rank_${SLURM_PROCID}"; exec "'"${maia_bin}"'" ./properties.toml'
