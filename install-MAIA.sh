#!/usr/local_rwth/bin/zsh

source ./setup_env_claix23.sh

source ./CPP-ML-Interface/extern/python/venv/bin/activate

export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=1 --nocompiler --user --mpp=mpi --io=none --memory=none --thread=none --nocuda"
export SCOREP_ENABLE_CUDA=0
#export SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP"
pushd maia

# If $1 is set, use it as the number of processors.
if [[ -n "$1" ]]; then
    # Check if $1 is a valid positive integer
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
        echo "Error: NPROC must be a positive integer."
        exit 1
    fi
    NPROC=$1
    echo "Using user-defined NPROC: $NPROC"
elif [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
    NPROC=$SLURM_CPUS_PER_TASK
    echo "Using SLURM-defined NPROC: $NPROC"
else
    NPROC=$(nproc)
    echo "Using default NPROC: $NPROC"
fi

./configure.py 1 2 --enable-instrumentation scorep --instrument mpi --instrument user && make -j$NPROC