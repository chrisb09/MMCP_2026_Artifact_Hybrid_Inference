#!/usr/local_rwth/bin/zsh

# Define the sets of X and Y values
cpu_nodes=(1 2 4 8)
gpu_nodes=(1 4)

# Iterate over all combinations
for x in "${cpu_nodes[@]}"; do
    for y in "${gpu_nodes[@]}"; do
        folder="${x}CPUNodes${y}GPU"

        # Check if the folder exists
        if [ -d "$folder" ]; then
            echo "Entering folder: $folder"
            cd "$folder" || exit 1

            # Execute test.sh if it exists and is executable
            if [ -x "./run_step.sh" ]; then
                ./run_step.sh
            else
                echo "Warning: run_step.sh not found or not executable in $folder"
            fi

            # Go back to previous directory
            cd ..
        else
            echo "Folder not found: $folder"
        fi
    done
done