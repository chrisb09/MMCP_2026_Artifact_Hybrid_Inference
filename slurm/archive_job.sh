#!/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name="archive-job"
#SBATCH --output=logs/archive.%J.txt
#SBATCH --error=logs/archive.%J.err

# Usage:
#   sbatch --dependency=afterok:JOBID slurm/archive_job.sh JOBID
# Example:
#   sbatch --dependency=afterok:64991271 slurm/archive_job.sh 64991271
#
# The script will auto-detect the job description from custom_logs files
# based on the pattern: {NODES}x{TASKS_PER_NODE}={NTASKS}_{PARTITION}_{NODELIST}_{JOB_ID}

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

# Check if project folder exists
if [ ! -d "$project_folder" ]; then
    echo "Project folder '$project_folder' not found!"
    exit 1
fi

cd "$project_folder" || exit 1

# Get job ID from argument
JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Error: JOB_ID not provided!"
    echo "Usage: sbatch --dependency=afterok:JOBID slurm/archive_job.sh JOBID"
    exit 1
fi

# Auto-detect JOB_DESC from custom_logs files
# Look for files matching pattern *_${JOB_ID}.log or *_${JOB_ID}.csv
JOB_DESC=""
if [ -f "custom_logs/"*"_${JOB_ID}.log" ]; then
    # Extract the job description from the filename
    LOG_FILE=$(ls custom_logs/*_${JOB_ID}.log 2>/dev/null | head -n 1)
    if [ -n "$LOG_FILE" ]; then
        # Get basename and remove the _JOBID.log suffix
        BASENAME=$(basename "$LOG_FILE")
        JOB_DESC="${BASENAME%_${JOB_ID}.log}"
    fi
fi

if [ -z "$JOB_DESC" ]; then
    echo "Warning: Could not auto-detect job description from custom_logs/*_${JOB_ID}.log"
    echo "Using fallback job description"
    JOB_DESC="job"
fi

# Create directory name
ARCHIVE_DIR="prev_execs/${JOB_DESC}_${JOB_ID}"

echo "========================================="
echo "Archiving job: $JOB_ID"
echo "Description: $JOB_DESC"
echo "Archive directory: $ARCHIVE_DIR"
echo "========================================="

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Files to copy
FILES_TO_COPY=(
    "custom_logs/${JOB_DESC}_${JOB_ID}.csv"
    "custom_logs/${JOB_DESC}_${JOB_ID}.log"
    "logs/error.${JOB_ID}.txt"
    "logs/output.${JOB_ID}.txt"
    "forces.0.dat"
    "m_log"
    "Residual"
)

# Copy each file if it exists
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$file" ]; then
        echo "Copying: $file"
        cp "$file" "$ARCHIVE_DIR/"
    else
        echo "Warning: File not found: $file"
    fi
done

echo "========================================="
echo "Archival complete!"
echo "Files stored in: $ARCHIVE_DIR"
echo "========================================="

# List contents of archive directory
echo ""
echo "Archive contents:"
ls -lh "$ARCHIVE_DIR"
