#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-hybrid-aix
#SBATCH --output=logs/hybrid_aix_%j.out
#SBATCH --error=logs/hybrid_aix_%j.err
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

set -euo pipefail

project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-20}"
build_variant="${BUILD_VARIANT:-plain}"
build_suffix=""
[[ "${build_variant}" == "scorep" ]] && build_suffix="_scorep"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production_cmi${build_suffix}}"
run_dir="${project_folder}/scratch/hybrid_aix_${SLURM_JOB_ID}"

if [[ "${build_variant}" == "scorep" ]]; then
    export USE_SCOREP=1
    export SCOREP_METRIC_PAPI=""
fi
source "${project_folder}/setup_env_claix23.sh"
mkdir -p "${run_dir}/out"
ln -s "${project_folder}/input" "${run_dir}/input"
cp "${project_folder}/config_aix.toml" "${run_dir}/"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "0.96"/' "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
export FLOW_DEBUG_DUMP_DIR="${run_dir}/dumps"
export MAIA_SNAPSHOT_DIR="${run_dir}/snapshots"
cd "${run_dir}"
srun --label --mpi=pmix --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
    "${maia_build_dir}/bin/maia" ./properties.toml : \
    --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
    "${maia_build_dir}/bin/maia" ./properties.toml
