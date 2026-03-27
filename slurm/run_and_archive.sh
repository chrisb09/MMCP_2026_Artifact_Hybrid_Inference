#!/bin/zsh

# Helper script to submit a job and automatically chain an archive job
# Usage:
#   ./slurm/run_and_archive.sh <slurm_script> [additional args...]
# Example:
#   ./slurm/run_and_archive.sh slurm/new_example_job_devel.sh M "1.00"
#
# Note: Job description is auto-detected from custom_logs files

if [ $# -lt 1 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <slurm_script> [additional args...]"
    echo ""
    echo "Example:"
    echo "  $0 slurm/new_example_job_devel.sh M \"1.00\""
    exit 1
fi

SLURM_SCRIPT=$1
shift 1
EXTRA_ARGS="$@"

# Check if slurm script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Submit main job
echo "Submitting main job: $SLURM_SCRIPT $EXTRA_ARGS"
JOB_OUTPUT=$(sbatch $SLURM_SCRIPT $EXTRA_ARGS)
JOB_ID=$(echo $JOB_OUTPUT | awk '{print $NF}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    echo "$JOB_OUTPUT"
    exit 1
fi

echo "Main job submitted: $JOB_ID"

# Submit archive job with dependency (job description auto-detected)
echo "Submitting archive job (depends on: $JOB_ID)"
ARCHIVE_OUTPUT=$(sbatch --dependency=afterok:$JOB_ID slurm/archive_job.sh $JOB_ID)
ARCHIVE_JOB_ID=$(echo $ARCHIVE_OUTPUT | awk '{print $NF}')

echo "Archive job submitted: $ARCHIVE_JOB_ID"
echo ""
echo "========================================="
echo "Jobs submitted successfully!"
echo "  Main job ID:    $JOB_ID"
echo "  Archive job ID: $ARCHIVE_JOB_ID"
echo "========================================="
echo ""
echo "Monitor with:"
echo "  squeue -u $(whoami)"
echo "  tail -f logs/output.$JOB_ID.txt"
