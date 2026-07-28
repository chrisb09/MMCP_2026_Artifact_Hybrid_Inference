#!/usr/bin/zsh
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=direct-smartsim-libtorch
#SBATCH --output=logs/direct_smartsim_libtorch_%J.out
#SBATCH --error=logs/direct_smartsim_libtorch_%J.err

set -euxo pipefail
project_folder="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${project_folder}"
source setup_env_claix23.sh

torch_root="${project_folder}/CPP-ML-Interface/extern/python/smartsim_cuda-12/lib/python3.9/site-packages/torch"
torch_include="${torch_root}/include"
torch_api_include="${torch_include}/torch/csrc/api/include"
torch_lib="${torch_root}/lib"
runtime_libs="${project_folder}/CPP-ML-Interface/extern/python/smartsim_cuda-12/runtime_libs"
output_dir="${project_folder}/analysis/direct_torch_runs"

g++ -std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"${torch_include}" -I"${torch_api_include}" \
    analysis/direct_libtorch_inference.cpp \
    -L"${torch_lib}" -Wl,-rpath,"${torch_lib}" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
    -o "${output_dir}/direct_libtorch_smartsim"

export LD_LIBRARY_PATH="${runtime_libs}:${torch_lib}:${LD_LIBRARY_PATH:-}"
"${output_dir}/direct_libtorch_smartsim" \
    input/transformer_inference_scripted_fw2.pt \
    scratch/current_aix_cpu_2314057/debug/cmi/current_rank_0_inference_1_normalized_input.bin \
    "${output_dir}/cpp_smartsim_b3456.bin" 3456
