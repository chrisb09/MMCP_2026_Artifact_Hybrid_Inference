#!/usr/bin/env bash
# Rank-dispatching wrapper for the GPU-side het component that must host BOTH
# solver ranks and PhyDLL DL-client ranks. Slurm cannot reference one het group
# twice in a single srun MPMD, and --multi-prog is not allowed across multiple
# het groups, so this wrapper dispatches each task by its local rank.
#
# Usage: cmi_bench_dispatch.sh N_SOLVER SOLVER_SCRIPT DL_SCRIPT
set -uo pipefail

local_id="${SLURM_LOCALID:-0}"
n_solver="${1:?N_SOLVER required}"
solver_script="${2:?SOLVER_SCRIPT required}"
dl_script="${3:?DL_SCRIPT required}"

if [[ "${local_id}" -lt "${n_solver}" ]]; then
    exec "${solver_script}"
else
    exec "${dl_script}"
fi
