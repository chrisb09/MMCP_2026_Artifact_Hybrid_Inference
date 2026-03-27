#!/bin/zsh

rm m_log
rm forces.0.dat
rm Residual

# Get running/pending job IDs for current user
job_ids=$(squeue -u $(whoami) -h -o "%A" | tr '\n' ':' | sed 's/:$//')

echo "Current job IDs: $job_ids"
echo "Submitting main job..."

if [[ -n "$job_ids" ]]; then
    JOB_OUTPUT=$(sbatch --dependency=afterany:$job_ids slurm/new_example_job_hybrid_phydll_c23mm_c23g_x1.sh M "1.00")
else
    JOB_OUTPUT=$(sbatch slurm/new_example_job_hybrid_phydll_c23mm_c23g_x1.sh M "1.00")
fi

JOB_ID=$(echo $JOB_OUTPUT | awk '{print $NF}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit main job"
    echo "$JOB_OUTPUT"
    exit 1
fi

echo "Main job submitted: $JOB_ID"

# Submit archive job with dependency (job description auto-detected from custom_logs)
echo "Submitting archive job (depends on: $JOB_ID)..."
ARCHIVE_OUTPUT=$(sbatch --dependency=afterok:$JOB_ID slurm/archive_job.sh $JOB_ID)
ARCHIVE_JOB_ID=$(echo $ARCHIVE_OUTPUT | awk '{print $NF}')

echo "Archive job submitted: $ARCHIVE_JOB_ID"
echo ""
echo "Monitor with: squeue -u $(whoami)"