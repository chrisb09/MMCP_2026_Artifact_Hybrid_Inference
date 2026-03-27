#!/bin/zsh

############################
# Test hetjob MPI world
############################
#SBATCH --account=p0025821
#SBATCH --time=00:02:00
#SBATCH --job-name=test-mpi-hetjob
#SBATCH --output=logs/test_mpi_%J.txt
#SBATCH --error=logs/test_mpi_err_%J.txt

############################
# Component 0: CPU (c23mm)
############################
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=4

#SBATCH hetjob

############################
# Component 1: GPU (c23g)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2

echo "=== Testing hetjob MPI_COMM_WORLD ==="
echo "CPU nodes: $(scontrol show hostname $SLURM_JOB_NODELIST_HET_GROUP_0)"
echo "GPU nodes: $(scontrol show hostname $SLURM_JOB_NODELIST_HET_GROUP_1)"

mkdir -p logs
chmod +x phydll-tom/test_wrapper.sh

# Force OpenMPI to use external pmix, skip pmi/pmi2
export OMPI_MCA_pmix=pmix3x
export OMPI_MCA_ess=pmi

echo ""
echo "=== Test 1: With --het-group flags (current approach) ==="
srun --mpi=pmix --het-group=0 ./phydll-tom/test_wrapper.sh : --het-group=1 ./phydll-tom/test_wrapper.sh

echo ""
echo "=== Test 2: With SEPARATE output files (like PhyDLL job) ==="
srun --output=logs/test_mpi_cpu_%J.txt --mpi=pmix --het-group=0 ./phydll-tom/test_wrapper.sh : --output=logs/test_mpi_gpu_%J.txt --het-group=1 ./phydll-tom/test_wrapper.sh

echo ""
echo "=== Test 3: Without --het-group flags (pure MPMD) ==="
srun --mpi=pmix --ntasks=4 ./phydll-tom/test_wrapper.sh : --ntasks=2 ./phydll-tom/test_wrapper.sh

echo ""
echo "=== Test 4: MPI_Comm_split like PhyDLL (physical vs DL) ==="
chmod +x phydll-tom/test_phy_wrapper.sh phydll-tom/test_dl_wrapper.sh
srun --mpi=pmix --het-group=0 ./phydll-tom/test_phy_wrapper.sh : --het-group=1 ./phydll-tom/test_dl_wrapper.sh
