#!/bin/bash
#SBATCH --job-name=test_c_vs_py
#SBATCH --time=00:05:00
#SBATCH --account=p0025821
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=2

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1

#SBATCH --output=logs/test_c_vs_py_%J.txt

module load foss/2023b
module load CUDA/12.4.0
module load Python/3.11.5

# Activate venv with mpi4py
source /hpcwork/ro092286/analysis/bin/activate

cd /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/phydll-tom

# Export PMIx settings
export OMPI_MCA_pmix=pmix3x
export OMPI_MCA_ess=pmi

echo "=== TEST: C binary on CPU partition, Python on GPU partition ==="
echo ""

# Test: C on het-group 0, Python on het-group 1
srun --mpi=pmix \
    --het-group=0 ./test_c_wrapper.sh \
    : --het-group=1 ./test_py_wrapper.sh

echo ""
echo "=== TEST COMPLETE ==="
