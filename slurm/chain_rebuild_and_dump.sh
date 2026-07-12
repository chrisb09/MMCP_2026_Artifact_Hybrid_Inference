#!/usr/bin/zsh
# Submit: rebuild -> debug dump run (chained)
set -euo pipefail

REBUILD_OUTPUT=$(sbatch --parsable slurm_rebuild_debug.sh)
echo "Rebuild job: ${REBUILD_OUTPUT}"

mkdir -p logs
RUN_OUTPUT=$(sbatch --parsable --dependency=afterok:${REBUILD_OUTPUT} slurm/debug_dump_devel.sh)
echo "Debug dump run job: ${RUN_OUTPUT}"
echo "Monitor: squeue -u \$(whoami)"
echo "Dumps will be in: debug_dumps/"
