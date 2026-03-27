#!/bin/zsh

rm m_log
rm forces.0.dat
rm Residual

echo "Submitting main job..."
JOB_OUTPUT=$(sbatch slurm/new_example_job_c23mm_parallel.sh M "1.00")
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
