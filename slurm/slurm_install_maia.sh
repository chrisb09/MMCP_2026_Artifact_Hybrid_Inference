#!/usr/bin/env zsh
#SBATCH --job-name=install_maia_test
#SBATCH --output=logs/install_maia_test.%j.out
#SBATCH --error=logs/install_maia_test.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=0:50:00
#SBATCH --partition=devel

bash install-MAIA.sh