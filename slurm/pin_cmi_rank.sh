#!/usr/bin/env bash
# Pin the calling MPI rank to a core from the Slurm allocation's allowed set.
# Ranks are assigned round-robin so oversubscribed DL ranks share solver cores.
# Affinity is derived from the cgroup cpuset, not hardcoded core numbers.
set -u
rank="${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-${PMI_RANK:-0}}}"
allowed="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')"
[[ -n "${allowed}" ]] || allowed="0"

cpus=()
IFS=',' read -ra parts <<< "${allowed}"
for part in "${parts[@]}"; do
    if [[ "${part}" == *-* ]]; then
        lo="${part%-*}"
        hi="${part#*-}"
        for ((c = lo; c <= hi; ++c)); do
            cpus+=("${c}")
        done
    else
        cpus+=("${part}")
    fi
done
count="${#cpus[@]}"
target="${cpus[$((rank % count))]}"
taskset -p -c "${target}" $$ >/dev/null 2>&1 || true
if [[ "${CMI_PIN_VERBOSE:-0}" == "1" ]]; then
    printf '[CMI_PIN] rank=%s target_core=%s allowed=%s\n' "${rank}" "${target}" "${allowed}" >&2
fi
exec "$@"
