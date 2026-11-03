# Hybrid Inference Optimization for AI-Enhanced Turbulent Boundary Layer Simulation on Heterogeneous Systems - Supplemental Material

This is supplemental material for the paper "Hybrid Inference Optimization for AI-Enhanced Turbulent Boundary Layer Simulation on Heterogeneous Systems" submitted to the MMCP 2026 workshop.


## Repository Structure
- [CPP-ML-Interface](CPP-ML-Interface/): Source code of the ML-Module partially ported to C++ from [Fortran](https://github.com/RWTH-HPC/Fortran-ML-Interface). It also includes the [AIxeleratorService](https://github.com/RWTH-HPC/AIxeleratorService/tree/MMCP_2026) library for hybrid inference as a submodule.
- [maia](maia/): Source code of the finite volume solver from the multi-physics simulation framework [m-AIA](https://git.rwth-aachen.de/aia/m-AIA/m-AIA/-/tree/v2024.1?ref_type=tags) coupled with the TBL-Transformer model.
- [input](input/): contains the required input files for the solver (e.g. grid, restart file, Transformer model)

## Installation

### Software Environment
The code was tested on the CLAIX-2023 cluster at RWTH Aachen University.
On CLAIX-2023 you can setup the correct environment via
```
source setup_env_claix23.sh
```
which will load some [modules](setup_env_claix23.sh) corresponding to the following toolchain:
- GNU compilers 13.2.0
- FFTW 3.3.10 (MPI parallel version)
- PnetCDF 1.12.3
- HDF5 1.14.3
- Python 3.11.5
- CMake 3.29.3
- Intel oneAPI Math Kernel Library 2024.2.0
- CUDA 12.4.0
- cuDNN 8.9.7.29
- Score-P 8.4

On other systems you may need to create a similar software environment yourself.
Please report any issue on other systems!

### Dependencies
The m-AIA solver code depends on the [CPP-ML-Interface](CPP-ML-Interface/) library, so this should be built first by using the provided [install script](CPP-ML-Interface/install-scorep.sh).
```
cd CPP-ML-Interface
./install-scorep.sh
```
This script should work on any Linux-based system and will automatically install the following dependencies into `CPP-ML-Interface/extern/`:
- a Python virtual environment
- [Physics Deep Learning coupLer (PhyDLL)](https://gitlab.com/cerfacs/phydll/)
- prebuilt libtorch 2.6.0
- [AIxeleratorService](https://github.com/RWTH-HPC/AIxeleratorService/tree/MMCP_2026)
- [HighFive](https://bluebrain.github.io/HighFive/)

and finally the compiled CPP-ML-Interface library will be found in `CPP-ML-Interface/BUILD-SCOREP`.
Further information can be found in the [README](CPP-ML-Interface/README.md).

### m-AIA Solver
To build the coupled m-AIA solver use the provided [install script](install-MAIA.sh).
```
./install-MAIA.sh
```
Further information regarding m-AIA's build process can be found in the [README](maia/README.md).



## Reproducing Results

To reproduce the results shown in the paper, we provide an automated benchmark suite. Ensure that you have successfully built the solver (see **Installation**) before proceeding.

### 1. Setup Benchmark Suite
First, run the setup script with your SLURM account. This script performs the following actions:
- Downloads the required large input files (grid, restart files) into the `input/` directory.
- Sets up a Python virtual environment (`.venv`) with all necessary plotting dependencies.
- Generates the complete directory structure for all experiments in `data/`.

```bash
# Replace <ACCOUNT> with your actual Slurm account (e.g., rwth1234)
./create_benchmark_suite.sh <ACCOUNT>
```

### 2. Run Experiments

Once the suite is set up, navigate to the data directory and execute the experiment runner. This will submit the jobs to the cluster.

```bash
cd data
./run_all_experiments.sh
```

### 3. Generate Plots

After the experiments have finished, you can generate the plots used in the paper by running:

```bash
./run_all_plotting.sh
```