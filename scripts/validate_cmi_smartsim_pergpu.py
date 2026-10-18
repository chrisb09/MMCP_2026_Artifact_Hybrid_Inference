#!/usr/bin/env python3
"""Fail-closed validation of an explicitly designated per-GPU supplement.

Reads only manifest-selected status files and run paths. Never selects a latest
run, queries Slurm, submits jobs, or modifies canonical consolidation outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path

STEPS = (15, 44, 73, 102, 131, 160, 189)
FIELDS = ('cubes_per_rank', 'samples_per_rank', 'cubes_aggregate',
          'samples_aggregate', 'workload_solver_ranks')
MODELS = dict(transformer='transformer_inference_scripted_fw2.pt', mlp='mmcp_test_mlp_m5_cpu.pt')
WORKLOAD = re.compile(r'ML_WORKLOAD_LOG cubes_per_rank=(\d+) samples_per_rank=(\d+) '
                      r'cubes_aggregate=(\d+) samples_aggregate=(\d+).*?solver_ranks=(\d+)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def kv(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def check_hashes(path, base):
    hashes = {}
    for line in path.read_text().splitlines():
        digest, name = line.split('  ', 1)
        source = Path(name)
        source = source if source.is_absolute() else base / source
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, f'hash mismatch: {source}')
        hashes[source.name] = digest
    require(bool(hashes), 'empty artifact hashes')
    return hashes


def validate_run(row, manifest, baseline):
    cell, jobid = row['cell'], row['jobid']
    placement, model, rep = row['placement'], row['model'], row['rep']
    require(placement in ('96c4g', '96p96c4g') and model in MODELS and rep in ('1', '2', '3'), 'invalid cell dimensions')
    require(cell == f'smartsim-{placement}-pergpu-{model}-r{rep}', 'invalid per-GPU cell name')
    require(row['dry_run'] == '0' and jobid.isdigit(), 'dry-run or invalid job ID')
    tpq = 4 if (model, placement) == ('transformer', '96c4g') else 1
    # The original pilot manifest predates the explicit timeout column.
    case_timeout = row.get('case_timeout_seconds', '300')
    require(case_timeout in ('300', '600'), 'unsupported manifest case timeout')
    require(row['tpq'] == str(tpq), 'manifest TPQ mismatch')
    status_dir = manifest.parent / 'status'
    require(Path(row['status_dir']).resolve() == status_dir.resolve(), 'foreign status directory')
    statuses = rows(status_dir / f'{cell}_{jobid}.csv')
    require(len(statuses) == 1, 'expected exactly one designated status')
    status = statuses[0]
    require((status['cell'], status['jobid'], status['status']) == (cell, jobid, '0'), 'cell/job status mismatch')
    directory = Path(status['run_dir']).resolve()
    info = kv(directory / 'run.info')
    expected = dict(CELL=cell, REP=rep, JOB_ID=jobid, RUN_DIR=str(directory),
                    CAMPAIGN_MANIFEST=str(manifest), GIT_REVISION=row['revision'],
                    SOLVER_EXIT_STATUS='0', EXIT_STATUS='0', RUN_STEPS='200',
                    SOLUTION_INTERVAL='10000', ML_CUBE_OVERLAP='0' if model == 'transformer' else '4',
                    SMARTSIM_DB_LAYOUT='per-ml-node', SMARTSIM_DB_NODES='4',
                    SMARTSIM_DB_GPU_COUNT='4', MAIA_SMARTSIM_NUM_GPUS='1',
                    SMARTSIM_TPQ=str(tpq), SMARTSIM_INTRA_OP_THREADS='4',
                    SMARTSIM_INTER_OP_THREADS='1', SMARTSIM_CPU_CORES='0',
                     SMARTSIM_DB_STEP_MEM='0', CASE_TIMEOUT_SECONDS=case_timeout)
    if placement == '96c4g':
        expected.update(CASE='smartsim', SOLVER_RANKS='96', DL_RANKS='0', SMARTSIM_INTERFACE='lo')
    else:
        expected.update(PROVIDER='SMARTSIM', HET_CPU_TASKS='96', HET_GPU_TASKS='96', HET_GPUS='4', SMARTSIM_INTERFACE='ib0')
    for key, value in expected.items():
        require(info.get(key) == value, f'{key}: {info.get(key)!r} != {value!r}')
    require(Path(info['MODEL']).name == MODELS[model], 'model mismatch')
    require(math.isfinite(float(info['BENCHMARK_SOLVER_WALL_SECONDS'])) and float(info['BENCHMARK_SOLVER_WALL_SECONDS']) > 0, 'invalid wall time')
    config = tomllib.loads((directory / 'config_smartsim.toml').read_text())['library']
    for key, value in dict(db_layout='per-ml-node', nodes=4, num_gpus=1, first_gpu=0).items():
        require(config.get(key) == value, f'run-local config {key} mismatch')
    artifact_hashes = check_hashes(directory / 'run.sha256', directory)
    require({'maia', MODELS[model], 'properties.toml', 'config_smartsim.toml', 'smartsim_controller.py'} <= artifact_hashes.keys(), 'missing required artifact hashes')
    # The controller may be edited by another agent before submission, but the
    # submitted snapshot and the actual run must agree exactly.
    source_hashes = dict(line.split('  ', 1)[::-1] for line in (manifest.parent / 'source.sha256').read_text().splitlines())
    controller_hashes = [value for name, value in source_hashes.items() if name.endswith('/smartsim_controller.py')]
    require(controller_hashes == [artifact_hashes.get('smartsim_controller.py')], 'controller differs from submitted source')

    allocation = (directory / 'slurm_allocation.log').read_text()
    components = re.split(r'(?m)(?=^JobId=)', allocation)[1:]
    require(len(components) == (1 if placement == '96c4g' else 2), 'wrong allocation component count')
    for index, component in enumerate(components):
        if placement == '96c4g':
            require(re.search(r'\bJobId=' + jobid + r'\b', component), 'allocation job ID mismatch')
            require(re.search(r'MinMemoryNode=0(?:\s|$)', component), 'allocation memory mismatch')
        else:
            require(re.search(r'\bHetJobId=' + jobid + r'\b', component) and
                    re.search(r'\bHetJobOffset=' + str(index) + r'\b', component), 'heterogeneous allocation job/component mismatch')
            require(re.search(r'MinMemoryCPU=5G\b', component), 'allocation memory mismatch')
        require(re.search(r'NumNodes=1\b', component) and re.search(r'NumTasks=96\b', component), 'allocation ranks/nodes mismatch')
        require('TimeLimit=00:20:00' in component, 'allocation time mismatch')
        require(re.search(r'(?:OverSubscribe=EXCLUSIVE|Exclusive=NODE|Shared=0)\b', component), 'allocation not exclusive')
        require(re.search(r'(?:gres/gpu=4\b|TresPerNode=gres/gpu(?::[^\s:]+)?:4\b)', component), 'allocation GPU count mismatch')

    solver = (directory / 'solver.log').read_text()
    calls = re.findall(r'ML_STEP_TIMING step=(\d+) ml_ms=(\S+)', solver)
    require(tuple(int(step) for step, _ in calls) == STEPS and solver.count('ML_STEP_TIMING step=') == 7, 'expected seven inference steps')
    require(all(math.isfinite(float(ms)) and float(ms) > 0 for _, ms in calls), 'invalid inference timings')
    workloads = set(WORKLOAD.findall(solver))
    require(len(workloads) == 1, 'missing/conflicting workload')
    workload = next(iter(workloads))
    require(all(int(v) > 0 for v in workload), 'invalid workload')
    require(int(workload[-1]) == (96 if placement == '96c4g' else 192), 'workload rank count mismatch')
    shared = [r for r in baseline if r['cell'] == cell.replace('-pergpu-', '-shared-')]
    require(len(shared) == 1 and workload == tuple(shared[0][key] for key in FIELDS), 'workload differs from canonical shared cell')
    endpoints = (directory / '.ssdb_endpoint').read_text().strip().split(',')
    require(len(endpoints) == len(set(endpoints)) == 4, 'expected four distinct endpoints')
    leaders = re.findall(r'Provider initialized on rank (\d+) \[db_layout=per-ml-node, db_index=(\d+),.*?is_leader=true, SSDB=([^\]]+)\]', solver)
    # logging.hpp suppresses every nonzero MPI rank, including the other DB
    # leaders. Four live Redis/GPU processes below provide separate runtime proof.
    require(len(leaders) == 1 and leaders[0][:2] == ('0', '0'), 'missing/conflicting rank-zero provider DB leader')
    for _, index, address in leaders:
        require(address == endpoints[int(index)], 'provider endpoint mismatch')
    counts = re.findall(r'Number of GPUs:\s*(\d+)', solver)
    require(not counts or set(counts) == {'1'}, 'model GPU-count differs from one per DB')
    controller = (directory / 'controller.log').read_text()
    maps = [json.loads(line.split('SMARTSIM_DB_GPU_MAP ', 1)[1]) for line in controller.splitlines() if 'SMARTSIM_DB_GPU_MAP ' in line]
    require(len(maps) == 4 and {m['shard'] for m in maps} == set(range(4)), 'expected four controller GPU mappings')
    masks = info['SMARTSIM_DB_GPU_IDS'].split(',')
    require(len(masks) == len(set(masks)) == 4, 'invalid allocated GPU mask')
    require(info.get('SMARTSIM_DB_GPU_ORIGIN') in (('het-group-1-step',) if placement == '96p96c4g' else ('inherited-allocation', 'single-node-step')), 'invalid GPU probe origin')
    identities = dict(line.replace(' ', '').split(',', 1) for line in (directory / 'smartsim_gpu_identity.log').read_text().splitlines()[1:] if ',' in line)
    mapped = {}
    for mapping in maps:
        i = mapping['shard']
        require(mapping['mask'] == masks[i] and mapping['logical_device'] == 0, 'controller mask/logical-device mismatch')
        require(str(mapping['port']) == endpoints[i].rsplit(':', 1)[1], 'controller port mismatch')
        uuid = identities.get(masks[i], masks[i] if masks[i] in identities.values() else '')
        require(uuid.startswith('GPU-'), 'unresolved physical GPU UUID')
        mapped[uuid] = str(mapping['port'])
    require(len(mapped) == 4, 'GPU mappings alias the same physical GPU')
    cpu_host = info.get('BATCH_HOST', '')
    ml_host = info.get('HET_GPU_NODELIST', '') if placement == '96p96c4g' else cpu_host
    require(cpu_host and ml_host and 'unknown' not in (cpu_host, ml_host), 'missing resource hosts')
    resource_peaks = {}
    for host in {cpu_host, ml_host}:
        key = f'NODE_RAM_PEAK_MIB_{host}'
        peak = float(info.get(key, 'nan'))
        require(math.isfinite(peak) and peak > 0, f'missing/invalid resource peak {key}')
        resource_peaks[key] = peak
    for index, uuid in identities.items():
        if uuid not in mapped:
            continue
        key = f'GPU_VRAM_PEAK_MIB_{ml_host}_gpu{index}'
        peak = float(info.get(key, 'nan'))
        baseline_memory = float(info.get(f'GPU_VRAM_BASE_MIB_{ml_host}_gpu{index}', 'nan'))
        require(math.isfinite(peak) and peak > 0 and math.isfinite(baseline_memory) and baseline_memory >= 0,
                f'missing/invalid resource peak or baseline {key}')
        resource_peaks[key] = peak
    # A single contemporaneous sample must link four Redis processes to four
    # UUIDs and the four mapped endpoint ports. A union over time is insufficient.
    samples = {}
    for values in csv.reader((directory / 'smartsim_gpu_processes.csv').read_text().splitlines()):
        require(len(values) == 5, 'malformed GPU process sample')
        stamp, pid, uuid, memory, command = (v.strip() for v in values)
        if uuid in mapped and 'redis' in command.lower() and re.search(r'(?<!\d)' + mapped[uuid] + r'(?!\d)', command) and float(memory) > 0:
            samples.setdefault(stamp, {}).setdefault(uuid, set()).add(pid)
    require(any(set(sample) == set(mapped) and all(len(pids) == 1 for pids in sample.values()) and
                len(set.union(*sample.values())) == 4 for sample in samples.values()), 'no simultaneous four-process/four-GPU Redis proof')
    return dict(cell=cell, jobid=jobid, run_dir=str(directory), inferences=7, measured=5,
                workload=dict(zip(FIELDS, map(int, workload))), hashes=artifact_hashes, resource_peaks_mib=resource_peaks,
                provider_leader_log_scope='MPI rank zero only; other leaders suppressed by logging.hpp',
                inferred_leader_ranks=[i * int(workload[-1]) // 4 for i in range(4)])


def validate(manifest, baseline_path, expected_cells=None):
    manifest = manifest.resolve()
    entries = rows(manifest)
    require(bool(entries) and len({r['cell'] for r in entries}) == len(entries), 'empty/duplicate manifest cells')
    require(len({r['jobid'] for r in entries}) == len(entries), 'duplicate job IDs')
    if expected_cells is not None:
        require(len(entries) == expected_cells, f'expected {expected_cells} cells')
    baseline = rows(baseline_path)
    require(len(baseline) == 288 and len({r['cell'] for r in baseline}) == 288, 'expected preserved canonical 288-run baseline')
    accepted, errors = [], []
    for row in entries:
        try:
            accepted.append(validate_run(row, manifest, baseline))
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            errors.append(dict(cell=row.get('cell'), error=str(exc)))
    return dict(manifest=str(manifest), baseline=str(baseline_path.resolve()),
                baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                accepted=accepted, errors=errors, valid=not errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).resolve().parents[1] / 'results/cmi_bench_consolidated/runs.csv')
    parser.add_argument('--expected-cells', type=int, default=12, help='Use 2 for the Transformer r1 pilots')
    parser.add_argument('--output', type=Path, help='Optional dedicated JSON report; stdout always reports results')
    args = parser.parse_args()
    try:
        report = validate(args.manifest, args.baseline, args.expected_cells)
    except (ValueError, OSError, KeyError) as exc:
        report = dict(valid=False, errors=[dict(error=str(exc))])
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        require(args.output.resolve() != args.baseline.resolve() and 'cmi_bench_consolidated' not in args.output.parts, 'refusing canonical output path')
        args.output.write_text(text)
    print(text, end='')
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
