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
proc_peak=0
declare -A gpu_peak=()
printf '0\n' > "${mem_file}"
: > "${gpu_file}"
proc_file="${out_dir}/gpuproc_${host}.txt"
printf '0\n' > "${proc_file}"

# Baseline: GPU memory present before our workload allocates anything. This
# distinguishes our own growth from a co-tenant/leftover allocation.
base_file="${out_dir}/gpubase_${host}.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/, */ /' > "${base_file}" || true
fi

trap 'exit 0' TERM INT
while true; do
    # /proc/meminfo reports kB; convert to MiB to match the nvidia-smi units.
    used="$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{if(t!="") print int((t-a)/1024)}' /proc/meminfo 2>/dev/null)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used > mem_peak )); then
        mem_peak="${used}"
        printf '%s\n' "${mem_peak}" > "${mem_file}"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if [[ "${SMARTSIM_DB_LAYOUT:-shared}" == per-ml-node && "${HET_SURROGATE_HIDE_GPU:-0}" != 1 ]]; then
            # Keep UUIDs and process identity together for post-run isolation proof.
            stamp="$(date +%s)"
            while IFS= read -r app; do
                [[ -n "${app}" ]] || continue
                pid="${app%%,*}"; pid="${pid//[[:space:]]/}"
                cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
                printf '%s,%s,%s\n' "${stamp}" "${app}" "${cmd//,/;}" >> "${out_dir}/smartsim_gpu_processes.csv"
            done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null)
        fi
        # Number of processes holding a CUDA context on the GPUs (detects
        # one-context-per-rank rather than one GPU controller).
        nproc="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
        if [[ "${nproc}" =~ ^[0-9]+$ ]] && (( nproc > proc_peak )); then
            proc_peak="${nproc}"
            printf '%s\n' "${proc_peak}" > "${proc_file}"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader 2>/dev/null > "${out_dir}/gpuapps_${host}.txt" || true
        fi
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
