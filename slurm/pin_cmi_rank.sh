#!/usr/bin/env bash
# Rank wrapper for MAIA coupling runs.
#
# Slurm now binds solver ranks (--cpu-bind=cores). This wrapper handles only what
# Slurm cannot:
#   1. per-rank GPU hiding on the surrogate CPU-side het group,
#   2. an opt-in NUMA-aware physical-core pin (CMI_NUMA_PIN=1),
#   3. launching the per-node memory sampler once per node (local rank 0).
#
# Usage: pin_cmi_rank.sh <command> [args...]
set -u

# 1. GPU hiding. Slurm rewrites CUDA_VISIBLE_DEVICES per task after --export,
# so this must happen here, inside the rank.
if [[ "${HET_SURROGATE_HIDE_GPU:-0}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="NoDevFiles"
fi

local_rank="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"

# 3. One resource sampler per node (covers both het nodes).
if [[ "${local_rank}" == "0" && -n "${CMI_SAMPLER_SCRIPT:-}" && -x "${CMI_SAMPLER_SCRIPT}" && -n "${CMI_SAMPLER_DIR:-}" ]]; then
    ( "${CMI_SAMPLER_SCRIPT}" "${CMI_SAMPLER_DIR}" "${CMI_SAMPLER_INTERVAL:-1}" >/dev/null 2>&1 & )
fi

# 2. Opt-in NUMA-aware pinning (default: rely on Slurm --cpu-bind=cores).
if [[ "${CMI_NUMA_PIN:-0}" == "1" && -n "${CMI_NUMA_PIN_SCRIPT:-}" ]]; then
    pin_args=(python3 "${CMI_NUMA_PIN_SCRIPT}")
    [[ -n "${CMI_NUMA_PIN_BUDGET:-}" ]] && pin_args+=(--budget "${CMI_NUMA_PIN_BUDGET}")
    pin_args+=(-- "$@")
    exec "${pin_args[@]}"
fi

exec "$@"
