#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Only --submit calls sbatch; no account override is supplied.
mode="${1:---dry-run}"
[[ "${mode}" == --dry-run || "${mode}" == --submit ]] || { printf 'Usage: bash %s [--dry-run|--submit]\n' "$0" >&2; exit 2; }
project_dir="$(realpath "$(dirname "$0")/..")"
cases=(phydll-blocking phydll-readiness aix-collective smartsim)
[[ "${INCLUDE_AIX_SYNC_OFF:-0}" != 1 ]] || cases+=(aix-sync-off)
dependency="${SMOKE_DEPENDENCY:-}"
for smoke_case in "${cases[@]}"; do
    tasks=24
    [[ "${smoke_case}" != phydll-* && "${smoke_case}" != smartsim ]] || tasks=48
    if [[ "${smoke_case}" == phydll-* ]]; then
        np_dl="${NP_DL:-24}"
        [[ "${np_dl}" =~ ^[1-9][0-9]*$ ]] || { printf 'NP_DL must be a positive integer.\n' >&2; exit 2; }
        tasks=$((24 + np_dl))
    fi
    cmd=(sbatch --parsable --chdir="${project_dir}" --export=ALL --ntasks="${tasks}"
        --job-name="cmi-${smoke_case}")
    [[ -z "${dependency}" ]] || cmd+=(--dependency="${dependency}")
    cmd+=("${project_dir}/slurm/run_cmi_smoke_devel.sh" "${smoke_case}")
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${mode}" == --submit ]]; then
        job_id="$("${cmd[@]}")"
        job_id="${job_id%%;*}"
        printf 'Submitted %s: %s\n' "${smoke_case}" "${job_id}"
        # Serialize devel allocations, but allow later cases after a smoke failure.
        dependency="afterany:${job_id}"
        [[ -z "${SMOKE_DEPENDENCY:-}" ]] || dependency="${SMOKE_DEPENDENCY},${dependency}"
    fi
done
