#!/usr/bin/env bash
# Packed single-node benchmark group: runs the full 12-cell provider/layout/comm
# matrix for one (placement, model, rep) sequentially inside one allocation.
#
# Placement resources (--ntasks/--gres) are supplied by the submitter via the
# sbatch command line; this file's values are the 24c1g defaults.
#
# Required env:
#   GRP_PLACEMENT, GRP_MODEL, GRP_REP, NP_SOLVER, GRP_GPUS
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --job-name=cmi-group-single
#SBATCH --output=logs/cmi-group-single_%j.out
#SBATCH --error=logs/cmi-group-single_%j.err

set -uo pipefail

root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
root="$(realpath "${root}")"
: "${GRP_PLACEMENT:?GRP_PLACEMENT must be set}"
: "${GRP_MODEL:?GRP_MODEL must be set}"
: "${GRP_REP:?GRP_REP must be set}"
: "${NP_SOLVER:?NP_SOLVER must be set}"
: "${GRP_GPUS:?GRP_GPUS must be set}"

CMI_AIX_DIR="${root}/CPP-ML-Interface/extern/AIxeleratorService"
export PROJECT_DIR="${root}"
# shellcheck source=slurm/cmi_bench_group_cells.sh
source "${root}/slurm/cmi_bench_group_cells.sh"

mkdir -p "${root}/logs/group_cells" "${root}/results"
# Override for retries to keep the original summary intact.
summary="${GRP_SUMMARY_FILE:-${root}/results/group_${GRP_PLACEMENT}_${GRP_MODEL}_r${GRP_REP}_cells.csv}"
: > "${summary}"

printf 'GROUP_START placement=%s model=%s rep=%s np_solver=%s gpus=%s\n' \
    "${GRP_PLACEMENT}" "${GRP_MODEL}" "${GRP_REP}" "${NP_SOLVER}" "${GRP_GPUS}"

fails=0
cmi_group_run_single "${GRP_MODEL}" "${GRP_PLACEMENT}" "${NP_SOLVER}" "${GRP_GPUS}" \
    "${GRP_REP}" "${root}" "${summary}" || fails=$?

printf 'GROUP_DONE placement=%s model=%s rep=%s cells_failed=%s summary=%s\n' \
    "${GRP_PLACEMENT}" "${GRP_MODEL}" "${GRP_REP}" "${fails}" "${summary}"

(( fails == 0 )) && exit 0
exit 2
