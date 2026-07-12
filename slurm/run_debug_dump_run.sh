#!/bin/zsh
# Submit the debug-dump run job.
# Run from the project root directory.

set -euo pipefail

mkdir -p logs

job_ids=$(squeue -u $(whoami) -h -o "%A" | tr '\n' ':' | sed 's/:$//')
echo "Current pending/running job IDs: ${job_ids:-none}"

if [[ -n "$job_ids" ]]; then
    JOB_OUTPUT=$(sbatch --dependency=afterany:$job_ids slurm/debug_dump_run.sh)
else
    JOB_OUTPUT=$(sbatch slurm/debug_dump_run.sh)
fi

JOB_ID=$(echo $JOB_OUTPUT | awk '{print $NF}')
if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    echo "$JOB_OUTPUT"
    exit 1
fi

echo "Debug dump job submitted: $JOB_ID"
echo "Monitor with: squeue -u $(whoami)"
echo "Output log:   logs/output_debug_${JOB_ID}.txt"
echo "Dumps will appear in: debug_dumps/ (copied from /tmp/flow_debug after run)"
