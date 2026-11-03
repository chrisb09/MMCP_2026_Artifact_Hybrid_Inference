#!/usr/bin/env bash
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
set -euo pipefail

project="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ "${1:-}" != --inside ]]; then
    exec srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --export=ALL bash "$project/slurm/redisai_batching_pilot.sbatch" --inside
fi

# RedisAI loads its Torch backend lazily in the server process. Activating the
# Python environment alone does not expose CUDA's NVRTC shared library to it.
source "$project/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
[[ -r "$cuda_lib/libnvrtc.so.12" ]] || { printf 'Missing CUDA NVRTC in %s\n' "$cuda_lib" >&2; exit 1; }
export LD_LIBRARY_PATH="$cuda_lib:${LD_LIBRARY_PATH:-}"
env_file="${SMARTSIM_PYTHON_ENV:-/hpcwork/$(whoami)/smartsim/python/smartsim_cuda-12/bin/activate}"
[[ -r "$env_file" ]] || { printf 'Missing environment: %s\n' "$env_file" >&2; exit 1; }
source "$env_file"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* && "$CUDA_VISIBLE_DEVICES" != -1 ]] || { printf 'Expected one allocated CUDA_VISIBLE_DEVICES token\n' >&2; exit 1; }
mkdir -p "$project/results/redisai_batching_pilot"
run_dir="$(mktemp -d "$project/results/redisai_batching_pilot/${SLURM_JOB_ID}.XXXXXX")"
cd "$run_dir"
exec > >(tee -a pilot.log) 2>&1
printf 'job=%s host=%s CUDA_VISIBLE_DEVICES=%s run_dir=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES" "$run_dir"
git -C "$project" rev-parse HEAD
git -C "$project" status --short
sha256sum "$project/scripts/redisai_batching_pilot.py" "$project/slurm/redisai_batching_pilot.sbatch" "$project/CPP-ML-Interface/dl_clients/smartsim_controller.py"
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader
export SR_CMD_TIMEOUT=30 SR_SOCKET_TIMEOUT=30000 SR_MODEL_TIMEOUT=30000
export SR_LOG_LEVEL=INFO SR_LOG_FILE="$run_dir/smartredis.log"
controller_pid=''
cleanup() {
    touch .done
    if [[ -n "$controller_pid" ]]; then
        for ((i=0; i<20; i++)); do
            kill -0 "$controller_pid" 2>/dev/null || break
            sleep 1
        done
        kill "$controller_pid" 2>/dev/null || true
        wait "$controller_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
timeout --signal=TERM --kill-after=5 240s python3 "$project/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
    --launcher local --interface lo --db-nodes 1 --auto-port --use-gpu \
    --intra-op-threads 2 --inter-op-threads 1 --threads-per-queue 1 --timeout-s 60 \
    --endpoint-file .endpoint --done-file .done --exp-dir "$run_dir/experiment" > controller.log 2>&1 &
controller_pid=$!
for ((i=0; i<60; i++)); do
    [[ ! -s .endpoint ]] || break
    kill -0 "$controller_pid" 2>/dev/null || { printf 'Controller exited; see controller.log\n' >&2; exit 1; }
    sleep 1
done
[[ -s .endpoint ]] || { printf 'Database readiness timed out\n' >&2; exit 1; }
IFS= read -r SSDB < .endpoint
export SSDB
timeout --signal=TERM --kill-after=5 120s python3 -u "$project/scripts/redisai_batching_pilot.py"
