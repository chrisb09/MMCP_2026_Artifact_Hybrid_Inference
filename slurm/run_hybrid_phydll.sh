#!/usr/bin/env bash
#SBATCH --account=thes2181
#SBATCH --time=00:30:00
#SBATCH --job-name=maia-hybrid-phydll
#SBATCH --output=logs/hybrid_phydll_%j.out
#SBATCH --error=logs/hybrid_phydll_%j.err
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

client_kind="${PHYDLL_CLIENT:-cpp}"
[[ "${client_kind}" == "cpp" || "${client_kind}" == "python" ]] || { echo "PHYDLL_CLIENT must be cpp or python." >&2; exit 2; }
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
run_steps="${RUN_STEPS:-20}"
build_variant="${BUILD_VARIANT:-plain}"
build_suffix=""
[[ "${build_variant}" == "scorep" ]] && build_suffix="_scorep"
maia_build_dir="${MAIA_BUILD_DIR:-${project_folder}/maia/build_gnu_production_cmi${build_suffix}}"
run_dir="${project_folder}/scratch/hybrid_phydll_${client_kind}_${SLURM_JOB_ID}"

if [[ "${build_variant}" == "scorep" ]]; then
    export USE_SCOREP=1
    export SCOREP_METRIC_PAPI=""
fi
source "${project_folder}/setup_env_claix23.sh"
mkdir -p "${run_dir}/out"
cp "${project_folder}/config_phydll.toml" "${run_dir}/"
cp "${project_folder}/input/properties_run_les_ref_medium.toml" "${run_dir}/properties.toml"
sed -i "s/^timeSteps *=.*/timeSteps = ${run_steps}/" "${run_dir}/properties.toml"
sed -i 's/^hostFraction *=.*/hostFraction = "1.00"/' "${run_dir}/properties.toml"
ln -s "${project_folder}/input/grid_les_medium.hdf5" "${run_dir}/grid_les_medium.hdf5"
ln -s "${project_folder}/input/restart_les_init_medium.hdf5" "${run_dir}/out/restart_les_ref_medium.hdf5"

export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
export CPP_ML_INTERFACE_DEVICE=GPU
export FLOW_DEBUG_DUMP_DIR="${run_dir}/dumps"
export MAIA_SNAPSHOT_DIR="${run_dir}/snapshots"
export LD_LIBRARY_PATH="${project_folder}/CPP-ML-Interface/extern/phydll/build/lib:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_INTRA_OP_THREADS=24
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_DL_COUNT=1
export OMPI_MCA_pmix=pmix3x
export OMPI_MCA_ess=pmi
dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
[[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
[[ -x "${dl_client}" ]] || { echo "PhyDLL C++ client not found." >&2; exit 1; }

if [[ "${client_kind}" == "cpp" ]]; then
    dl_command="${dl_client}"
elif [[ "${build_variant}" == "scorep" ]]; then
    dl_command="python3 -m scorep --keep-files --instrumenter-type=dummy --noinstrumenter --mpp=none ${project_folder}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
else
    dl_command="python3 ${project_folder}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
fi

cd "${run_dir}"
srun --label --mpi=pmix --het-group=0 --ntasks=24 --cpus-per-task=1 --cpu-bind=cores \
    "${maia_build_dir}/bin/maia" ./properties.toml : \
    --het-group=1 --ntasks=1 --cpus-per-task=24 --cpu-bind=cores \
    bash -lc "${dl_command}"
