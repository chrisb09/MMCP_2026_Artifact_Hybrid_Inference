#!/bin/zsh

#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name="hello-world-devel-1g"
#SBATCH --output=logs/test_output.%J.txt
#SBATCH --error=logs/test_error.%J.txt

username=$(whoami)
project_folder="/hpcwork/${username}/MMCP_2026_Artifact_Hybrid_Inference"

echo "Hello, World! This is a test job running in the devel partition."
echo "User: $username"
echo "Project folder: $project_folder"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"

sleep 10

echo "Job completed successfully."