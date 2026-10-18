#!/usr/bin/env bash
# Sourced by both runners after copying the run-local provider configuration.
smartsim_layout_args=()
smartsim_cpu_cores="${np_solver:-${gpu_tasks:-0}}"
if [[ "${provider}" == SMARTSIM ]]; then
    export SMARTSIM_DB_LAYOUT="${SMARTSIM_DB_LAYOUT:-shared}"
    case "${SMARTSIM_DB_LAYOUT}" in
        shared) ;;
        per-ml-node)
            smartsim_cpu_cores=0
            export SMARTSIM_DB_NODES=4 SMARTSIM_DB_GPU_COUNT=4 MAIA_SMARTSIM_NUM_GPUS=1
            smartsim_layout_args=(--db-layout per-ml-node)
            python3 - <<'PY'
from pathlib import Path
import re
p = Path('config_smartsim.toml')
text = p.read_text()
start = text.index('[library]')
end = text.find('\n[', start + 1)
if end == -1:
    end = len(text)
section = text[start:end]
for key, value in dict(db_layout='"per-ml-node"', nodes='4', num_gpus='1', first_gpu='0').items():
    section = re.sub(r'^\s*' + key + r'\s*=.*\n?', '', section, flags=re.M)
    section += f'\n{key} = {value}\n'
p.write_text(text[:start] + section + text[end:])
PY
            # Never infer physical IDs from the allocation's GPU count. The het
            # batch shell belongs to group 0, so probe group 1 explicitly.
            if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
                probe=(srun --het-group=1 --nodes=1 --ntasks=1 --cpus-per-task=1 --overlap --exact --mem=0)
                gpu_origin=het-group-1-step
            elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
                probe=()
                gpu_origin=inherited-allocation
            else
                probe=(srun --nodes=1 --ntasks=1 --cpus-per-task=1 --overlap --exact --mem=0)
                gpu_origin=single-node-step
            fi
            "${probe[@]}" /bin/bash -c 'printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits' > smartsim_gpu_identity.log
            SMARTSIM_DB_GPU_IDS="$(python3 - <<'PY'
from pathlib import Path
lines = Path('smartsim_gpu_identity.log').read_text().splitlines()
mask = lines[0].removeprefix('CUDA_VISIBLE_DEVICES=')
identities = dict(line.replace(' ', '').split(',', 1) for line in lines[1:] if ',' in line)
tokens = mask.split(',')
if not mask:
    # UUIDs from a GPU-side allocation step are safe even if Slurm omits CVD.
    tokens = list(identities.values())
assert len(tokens) == len(set(tokens)) == 4, f'Expected four allocated GPU tokens: {tokens}'
uuids = [identities.get(t, t if t in identities.values() else '') for t in tokens]
assert all(uuids) and len(set(uuids)) == 4, f'Cannot resolve allocated tokens: {tokens}'
print(','.join(tokens))
PY
            )" || exit 2
            export SMARTSIM_DB_GPU_IDS
            printf 'SMARTSIM_DB_GPU_ORIGIN=%s\nSMARTSIM_DB_GPU_IDS=%s\n' "${gpu_origin}" "${SMARTSIM_DB_GPU_IDS}" > smartsim_gpu_probe.info
            ;;
        *) printf 'Unknown SMARTSIM_DB_LAYOUT: %s\n' "${SMARTSIM_DB_LAYOUT}" >&2; exit 2 ;;
    esac
fi

cmi_smartsim_metadata() {
    [[ "${provider}" == SMARTSIM ]] || return 0
    printf 'JOB_ID=%s\nSMARTSIM_DB_LAYOUT=%s\nSMARTSIM_DB_NODES=%s\nSMARTSIM_DB_GPU_COUNT=%s\n' \
        "${SLURM_JOB_ID}" "${SMARTSIM_DB_LAYOUT}" "${SMARTSIM_DB_NODES:-1}" "${SMARTSIM_DB_GPU_COUNT:-unset}"
    printf 'SMARTSIM_TPQ=%s\nSMARTSIM_INTRA_OP_THREADS=4\nSMARTSIM_INTER_OP_THREADS=1\nSMARTSIM_CPU_CORES=%s\n' "${SMARTSIM_TPQ:-${1}}" "${smartsim_cpu_cores}"
    printf 'SMARTSIM_INTERFACE=%s\nSMARTSIM_DB_STEP_MEM=0\nCASE_TIMEOUT_SECONDS=%s\n' "${2}" "${timeout_s}"
    printf 'SMARTSIM_LAUNCHER=%s\nSMARTSIM_DB_AUTO_PORT=1\nSMARTSIM_DB_USE_GPU=1\nSMARTSIM_READY_TIMEOUT_SECONDS=%s\n' "${3}" "${ready_s}"
    printf 'SMARTSIM_CMD_TIMEOUT=%s\nSMARTSIM_SOCKET_TIMEOUT=%s000\nSMARTSIM_MODEL_TIMEOUT=%s000\n' "${4}" "${4}" "${4}"
    python3 - <<'PY'
import json, tomllib
with open('config_smartsim.toml', 'rb') as stream:
    print('SMARTSIM_LIBRARY_CONFIG=' + json.dumps(tomllib.load(stream)['library'], sort_keys=True))
PY
    printf 'CAMPAIGN_MANIFEST=%s\n' "${CMI_PERGPU_MANIFEST:-}"
    [[ ! -f smartsim_gpu_probe.info ]] || /bin/cat smartsim_gpu_probe.info
    if [[ "${SMARTSIM_DB_LAYOUT}" == per-ml-node ]]; then
        if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
            scontrol show job "${SLURM_JOB_ID}+0" > slurm_allocation.log
            scontrol show job "${SLURM_JOB_ID}+1" >> slurm_allocation.log
        else
            scontrol show job "${SLURM_JOB_ID}" > slurm_allocation.log
        fi
        git -C "${project_dir}" rev-parse HEAD | xargs -r printf 'GIT_REVISION=%s\n'
        sha256sum "${maia}" "${model_path}" properties.toml config_smartsim.toml \
            "${project_dir}/CPP-ML-Interface/dl_clients/smartsim_controller.py" > run.sha256
    fi
}

cmi_smartsim_status() {
    [[ -n "${CMI_PERGPU_STATUS_DIR:-}" ]] || return 0
    printf 'cell,jobid,status,run_dir\n%s,%s,%s,%s\n' "${CELL_NAME}" "${SLURM_JOB_ID}" "$1" "${run_dir}" \
        > "${CMI_PERGPU_STATUS_DIR}/${CELL_NAME}_${SLURM_JOB_ID}.csv"
}
