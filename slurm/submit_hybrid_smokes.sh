#!/usr/bin/env bash
set -euo pipefail

# Submit two independent 20-step jobs at a time. The second pair waits for the
# first pair, respecting the two-active-job limit without serializing all tests.
submit() {
    sbatch --export=ALL,RUN_STEPS=20 "$@" | awk '{print $NF}'
}

aix_job="$(submit slurm/run_hybrid_aix.sh)"
smartsim_job="$(submit slurm/run_hybrid_smartsim.sh)"
dependency="afterany:${aix_job}:${smartsim_job}"
cpp_job="$(sbatch --dependency="${dependency}" --export=ALL,RUN_STEPS=20,PHYDLL_CLIENT=cpp slurm/run_hybrid_phydll.sh | awk '{print $NF}')"
python_job="$(sbatch --dependency="${dependency}" --export=ALL,RUN_STEPS=20,PHYDLL_CLIENT=python slurm/run_hybrid_phydll.sh | awk '{print $NF}')"

printf 'AIX=%s SmartSim=%s PhyDLL-C++=%s PhyDLL-Python=%s\n' "${aix_job}" "${smartsim_job}" "${cpp_job}" "${python_job}"
