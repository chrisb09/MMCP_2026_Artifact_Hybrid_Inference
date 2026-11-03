#!/usr/local_rwth/bin/zsh

ROOT_DIR=$(pwd)

# Use find with -print0 to properly handle folder names with spaces.
# We search for leaf directories (with no subdirectories).
find . -mindepth 2 -type d -print0 | while IFS= read -r -d '' dir; do
    # Check if the directory is a leaf (no subdirectories inside)
    if [ -z "$(find "$dir" -mindepth 1 -type d -print -quit 2>/dev/null)" ]; then
        # Remove any leading "./" from the directory path.
        leaf=$(echo "$dir" | cut -d'/' -f3)  #${dir#./}

        # If the directory is not empty, skip processing it.
        if [ -n "$(ls -A "$dir")" ]; then
            echo "Skipping folder: $leaf because it is not empty."
            continue
        fi 

        echo "Processing folder: $leaf"
        
        pushd "$dir" > /dev/null || continue
        
        fraction=$(echo "$leaf" | sed -E 's/.*(A[0-9,]+).*/\1/' | sed 's/,/./g')
        echo $fraction
        if [[ "$fraction" == "1.00" || "$fraction" == "1" ]]; then
            echo "CPU Job"
            sbatch ${ROOT_DIR}/cpu.job M "$leaf"
        else
            echo "HYBRID Job"
            sbatch ${ROOT_DIR}/Hybrid.job M "$leaf"
        fi  
        #sbatch /hpcwork/rwth1859/HybridInference/8CPUNodes4GPU/Hybrid_2cpu.job M "$leaf"


        popd > /dev/null
    fi
done