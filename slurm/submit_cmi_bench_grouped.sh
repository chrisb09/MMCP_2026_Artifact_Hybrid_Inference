#!/usr/bin/env bash
# Submit packed MAIA [TBL] benchmark groups (Option A):
#   4 placements x 2 models x REPS reps = 24 jobs, each running the full
#   12-cell provider/layout/comm matrix sequentially.
#
# Override via env to run a subset, e.g.:
#   PLACEMENTS=24c1g MODELS=transformer REPS=1 slurm/submit_cmi_bench_grouped.sh
set -euo pipefail
cd "$(dirname "$0")/.."
root="$(pwd)"
mkdir -p logs results

REPS="${REPS:-1 2 3}"
MODELS="${MODELS:-transformer mlp}"
PLACEMENTS="${PLACEMENTS:-24c1g 24p24c1g 96c4g 96p96c4g}"
TIME_LIMIT="${TIME_LIMIT:-01:15:00}"
dependency_args=()
if [[ -n "${DEPENDENCY:-}" ]]; then
    dependency_args=(--dependency="${DEPENDENCY}")
fi
jobids="${JOBIDS_FILE:-results/cmi_bench_grouped_jobids.txt}"

record() { printf '%-40s %s\n' "$1" "$2"; printf '%s,%s\n' "$1" "$2" >> "${jobids}"; }

for model in ${MODELS}; do
    for placement in ${PLACEMENTS}; do
        for rep in ${REPS}; do
            cell="${GROUP_PREFIX:-grp}-${placement}-${model}-r${rep}"
            common_out="${root}/logs/cmi-${cell}_%j.out"
            common_err="${root}/logs/cmi-${cell}_%j.err"
            case "${placement}" in
                24c1g|96c4g)
                    [[ "${placement}" == 24c1g ]] && { ntasks=24; gpus=1; } || { ntasks=96; gpus=4; }
                    jid="$(sbatch --parsable "${dependency_args[@]}" \
                        --partition=c23g --account=rwth0792 --gres="gpu:${gpus}" --exclusive --mem=0 \
                        --nodes=1 --ntasks="${ntasks}" --cpus-per-task=1 --time="${TIME_LIMIT}" \
                        --job-name="cmi-${cell}" --output="${common_out}" --error="${common_err}" \
                        --export="ALL,GRP_PLACEMENT=${placement},GRP_MODEL=${model},GRP_REP=${rep},GRP_GPUS=${gpus},NP_SOLVER=${ntasks}" \
                        "${root}/slurm/bench_group_single.sbatch")"
                    ;;
                24p24c1g|96p96c4g)
                    jid="$(sbatch --parsable "${dependency_args[@]}" \
                        --job-name="cmi-${cell}" --output="${common_out}" --error="${common_err}" \
                        --time="${TIME_LIMIT}" \
                        --export="ALL,GRP_PLACEMENT=${placement},GRP_MODEL=${model},GRP_REP=${rep}" \
                        "${root}/slurm/bench_group_het_${placement}.sbatch")"
                    ;;
                *) printf 'Unknown placement: %s\n' "${placement}" >&2; exit 2 ;;
            esac
            record "${cell}" "${jid}"
        done
    done
done

echo "Job IDs in ${jobids}"
