#!/usr/bin/env bash
# Per-node resource sampler for MAIA coupling runs.
#
# Runs once per node (launched by the rank wrapper on local rank 0) and writes
# running maxima into the run directory so every het node is covered:
#   mem_<host>.txt   : node RAM peak in MiB (MemTotal - MemAvailable)
#   gpu_<host>.txt   : one "<index> <peak_mib>" line per visible GPU
#
# Usage: cmi_node_sampler.sh OUT_DIR [INTERVAL_SECONDS]
set -u

out_dir="${1:?output dir required}"
interval="${2:-1}"
host="$(hostname -s 2>/dev/null || hostname)"
mem_file="${out_dir}/mem_${host}.txt"
gpu_file="${out_dir}/gpu_${host}.txt"

mem_peak=0
declare -A gpu_peak=()
printf '0\n' > "${mem_file}"
: > "${gpu_file}"

trap 'exit 0' TERM INT
while true; do
    used="$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{if(t!="") print t-a}' /proc/meminfo 2>/dev/null)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used > mem_peak )); then
        mem_peak="${used}"
        printf '%s\n' "${mem_peak}" > "${mem_file}"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS=',' read -r idx val; do
            idx="${idx//[!0-9]/}"
            val="${val//[!0-9]/}"
            [[ -n "${idx}" && -n "${val}" ]] || continue
            if (( val > ${gpu_peak[$idx]:-0} )); then
                gpu_peak[$idx]="${val}"
            fi
        done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
        if (( ${#gpu_peak[@]} > 0 )); then
            : > "${gpu_file}"
            for idx in $(printf '%s\n' "${!gpu_peak[@]}" | sort -n); do
                printf '%s %s\n' "${idx}" "${gpu_peak[$idx]}" >> "${gpu_file}"
            done
        fi
    fi
    sleep "${interval}"
done
