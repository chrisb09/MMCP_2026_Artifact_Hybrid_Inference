#!/usr/bin/env bash
# Packed surrogate-het benchmark group: 96 (CPU-side) + 96 (GPU-side) solver
# ranks / 4 GPUs. Runs the full 12-cell provider/layout/comm matrix for one
# (model, rep) sequentially inside one heterogeneous allocation.
#
# Required env: GRP_MODEL, GRP_REP
#SBATCH --job-name=cmi-group-het96p96c4g
#SBATCH --account=rwth0792
#SBATCH --time=01:15:00
#SBATCH --exclusive
#SBATCH --output=logs/cmi-group-het96p96c4g_%j.out
#SBATCH --error=logs/cmi-group-het96p96c4g_%j.err

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

set -uo pipefail

root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
root="$(realpath "${root}")"
: "${GRP_MODEL:?GRP_MODEL must be set}"
: "${GRP_REP:?GRP_REP must be set}"

CMI_AIX_DIR="${root}/CPP-ML-Interface/extern/AIxeleratorService"
export PROJECT_DIR="${root}"
# shellcheck source=slurm/cmi_bench_group_cells.sh
source "${root}/slurm/cmi_bench_group_cells.sh"

mkdir -p "${root}/logs/group_cells" "${root}/results"
summary="${root}/results/group_96p96c4g_${GRP_MODEL}_r${GRP_REP}_cells.csv"
: > "${summary}"

printf 'GROUP_START placement=96p96c4g model=%s rep=%s het_size=%s\n' \
    "${GRP_MODEL}" "${GRP_REP}" "${SLURM_HET_SIZE:-}"

fails=0
cmi_group_run_het "${GRP_MODEL}" "96p96c4g" 96 96 4 "${GRP_REP}" "${root}" "${summary}" || fails=$?

printf 'GROUP_DONE placement=96p96c4g model=%s rep=%s cells_failed=%s summary=%s\n' \
    "${GRP_MODEL}" "${GRP_REP}" "${fails}" "${summary}"

(( fails == 0 )) && exit 0
exit 2
