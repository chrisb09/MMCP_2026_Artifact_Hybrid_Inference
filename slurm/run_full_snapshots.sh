#!/bin/zsh

rm -f m_log
rm -f forces.*.dat
rm -f Residual

# Enable field snapshot capture for verification/debugging
export MAIA_SNAPSHOT_DIR="/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/debug_dumps"

echo "Submitting main job..."
echo "Field snapshots will be written to: ${MAIA_SNAPSHOT_DIR}"
sbatch --export=ALL slurm/new_example_job_devel_24.sh M "1.00"
