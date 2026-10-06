#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --job-name=cmi-smoke-suite
#SBATCH --output=logs/cmi-smoke-suite_%j.out
#SBATCH --error=logs/cmi-smoke-suite_%j.err

set -uo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-500}"
export DEBUG_EXPORT=1
status=0
for smoke_case in ${CMI_SMOKE_CASES:-phydll-blocking phydll-readiness aix-collective smartsim aix-sync-off}; do
    case_timeout="${CASE_TIMEOUT_SECONDS}"
    [[ "${smoke_case}" != smartsim ]] || case_timeout="${SMARTSIM_CASE_TIMEOUT_SECONDS:-900}"
    if CASE_TIMEOUT_SECONDS="${case_timeout}" bash "${root}/slurm/run_cmi_smoke_devel.sh" "${smoke_case}"; then
        printf 'CASE_EXIT_PASS %s\n' "${smoke_case}"
    else
        result=$?
        printf 'CASE_EXIT_FAIL %s status=%s\n' "${smoke_case}" "${result}"
        status=1
    fi
done
exit "${status}"
