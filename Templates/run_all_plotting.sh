#!/bin/bash
# Loop through all directories in the current folder (excluding . and ..)

source $VENV_ACTIVATE
for dir in */; do
    # Skip if not a directory
    [ -d "$dir" ] || continue

    # Define subfolders
    benchmarkdir="${dir}Experiments"
    outputdir="${dir}plots"

    # Check that the expected subdirectories exist
    if [ -d "$benchmarkdir" ] && [ -d "$outputdir" ]; then
        echo "Running plotting.py for $dir"
        python plotting.py --benchmarkdir "$benchmarkdir" --output "$outputdir"
    else
        echo "Skipping $dir (missing Experiments or plots folder)"
    fi
done
