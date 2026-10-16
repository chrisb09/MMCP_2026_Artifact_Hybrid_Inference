#!/usr/bin/env bash
# Retry only the audited quota-incident cells, preserving original summaries.
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
manifest=results/cmi_bench_quota_retry_cells.csv
jobids=results/cmi_bench_quota_retry_jobids.txt
if [[ "${DRY_RUN:-0}" != 1 && -s "$jobids" ]]; then
    printf 'Refusing duplicate submission: %s already contains jobs.\n' "$jobids" >&2
    exit 1
fi
declare -A groups=() seen=()
count=0
while IFS=, read -r cell original reason; do
    [[ "$cell" != cell ]] || continue
    [[ "$cell" =~ ^(aix|phydll|smartsim)-(24c1g|96c4g|24p24c1g|96p96c4g)-.+-(transformer|mlp)-r([123])$ ]] || exit 2
    placement="${BASH_REMATCH[2]}" model="${BASH_REMATCH[3]}" rep="${BASH_REMATCH[4]}"
    [[ -z "${seen[$cell]:-}" && "$reason" == quota_* ]] || exit 2
    seen[$cell]=1
    key="${placement}_${model}_${rep}"
    groups[$key]="${groups[$key]:-}${groups[$key]:+ }${cell}"
    count=$((count + 1))
done < "$manifest"
[[ "$count" == 94 && "${#groups[@]}" == 12 ]] || exit 2
for model in transformer mlp; do
    for placement in 24c1g 24p24c1g 96c4g 96p96c4g; do
        for rep in 1 2 3; do
            key="${placement}_${model}_${rep}"
            cells="${groups[$key]:-}"
            [[ -n "$cells" ]] || continue
            read -r -a selected <<< "$cells"
            limit=01:15:00
            if (( ${#selected[@]} == 1 )); then
                limit=00:15:00
            elif (( ${#selected[@]} <= 4 )); then
                limit=00:30:00
            fi
            printf 'QUOTA_RETRY placement=%s model=%s rep=%s cells=%s\n' "$placement" "$model" "$rep" "$cells"
            [[ "${DRY_RUN:-0}" != 1 ]] || continue
            # Clear ambient dependencies and force the audited workload schedule.
            DEPENDENCY= PLACEMENTS="$placement" MODELS="$model" REPS="$rep" \
                GROUP_PREFIX=quota-retry-grp JOBIDS_FILE="$jobids" \
                CMI_GROUP_CELLS="$cells" \
                GRP_SUMMARY_FILE="${root}/results/quota_retry_group_${placement}_${model}_r${rep}_cells.csv" \
                RUN_STEPS=200 SOLUTION_INTERVAL=10000 CASE_TIMEOUT_SECONDS=300 READY_TIMEOUT_SECONDS=120 \
                TIME_LIMIT="$limit" \
                KEEP_BENCH_HDF5=0 AIX_LAYOUT_LIB= SMARTSIM_INTERFACE=ib0 SMARTSIM_DB_NODES=1 SMARTSIM_TPQ=1 \
                bash slurm/submit_cmi_bench_grouped.sh
        done
    done
done
