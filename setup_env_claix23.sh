#!/usr/local_rwth/bin/zsh

module purge

module load foss/2023b
module load FFTW.MPI/3.3.10
module load PnetCDF/1.12.3
module load HDF5/1.14.3
module load Python/3.11.5
module load CMake/3.29.3
module load imkl/2024.2.0

module load CUDA/12.4.0
module load cuDNN/8.9.7.29-CUDA-12.4.0
module load Score-P/8.4-CUDA-12.4.0


#module load foss/2024a
#module load FFTW.MPI/3.3.10
#module load PnetCDF/1.14.0
#module load HDF5/1.14.5
#module load Python/3.12.3
#module load CMake/4.0.2
#module load imkl/2024.2.0

#module load CUDA/12.6.3
#module load cuDNN/9.7.0.66-CUDA-12.6.3

#module load Score-P/9.0-CUDA-12.6.3

# Clean LD_LIBRARY_PATH from conflicting GCC 11.3 / OpenMPI 4.1.4 entries
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '11.3.0' | grep -v '2022a' | tr '\n' ':' | sed 's/:$//')
fi