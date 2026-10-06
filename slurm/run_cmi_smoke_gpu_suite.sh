#!/usr/bin/env bash
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --job-name=cmi-gpu-suite
#SBATCH --output=logs/cmi-gpu-suite_%j.out
#SBATCH --error=logs/cmi-gpu-suite_%j.err

set -uo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the artifact repository}"
export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-700}"
export DEBUG_EXPORT="${DEBUG_EXPORT:-1}"
status=0
for smoke_case in ${CMI_SMOKE_CASES:-aix-collective phydll-blocking phydll-readiness aix-p2p1 aix-p2pfull smartsim}; do
    case_timeout="${CASE_TIMEOUT_SECONDS}"
    [[ "${smoke_case}" != smartsim ]] || case_timeout="${SMARTSIM_CASE_TIMEOUT_SECONDS:-900}"
    if CASE_TIMEOUT_SECONDS="${case_timeout}" bash "${root}/slurm/run_cmi_smoke_gpu.sh" "${smoke_case}"; then
        printf 'CASE_EXIT_PASS %s\n' "${smoke_case}"
    else
        result=$?
        printf 'CASE_EXIT_FAIL %s status=%s\n' "${smoke_case}" "${result}"
        status=1
    fi
done
exit "${status}"
