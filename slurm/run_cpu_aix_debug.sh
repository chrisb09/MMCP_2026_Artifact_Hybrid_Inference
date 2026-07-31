#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=current-aix-cpu-debug
#SBATCH --output=logs/current_aix_cpu_%j.out
#SBATCH --error=logs/current_aix_cpu_%j.err

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-300}"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production}"
run_dir="${project_folder}/scratch/current_aix_cpu_${SLURM_JOB_ID}"

source "${project_folder}/setup_env_claix23.sh"
mkdir -p "${run_dir}/out"
ln -s "${project_folder}/input" "${run_dir}/input"
cp "${project_folder}/config_aix.toml" "${run_dir}/"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^mlInterval *=.*/mlInterval = 5/' "${run_dir}/properties.toml"
sed -i 's/^mlInputLength *=.*/mlInputLength = 5/' "${run_dir}/properties.toml"
sed -i 's/^mlStepCoefficient *=.*/mlStepCoefficient = 12/' "${run_dir}/properties.toml"
sed -i 's/^mlForecastWindow *=.*/mlForecastWindow = 2/' "${run_dir}/properties.toml"
sed -i 's/^mlInputStepDistance *=.*/mlInputStepDistance = 1/' "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
unset MLCOUPLING_DEBUG_EXPORT MLCOUPLING_DEBUG_ALL_RANKS MLCOUPLING_DEBUG_EXPORT_DIR
unset MLCOUPLING_DEBUG_RANK MLCOUPLING_DEBUG_MAX_INFERENCES MAIA_SNAPSHOT_DIR
export SCOREP_ENABLE_TRACING=false
export SCOREP_ENABLE_PROFILING=true

# The CPU devel nodes have no driver; LibTorch still links NVML through RPATH.
cuda_stubs="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
export LD_LIBRARY_PATH="${cuda_stubs}:${LD_LIBRARY_PATH:-}"

cd "${run_dir}"
start_seconds=$(date +%s)
srun --label --mpi=pmix --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
    "${project_folder}/CPP-ML-Interface/dl_clients/maia_runner.sh" "${maia_build_dir}/bin/maia" ./properties.toml
echo "BENCHMARK_SOLVER_WALL_SECONDS=$(( $(date +%s) - start_seconds ))"
