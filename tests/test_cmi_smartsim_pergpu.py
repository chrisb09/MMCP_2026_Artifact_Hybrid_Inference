"""Offline campaign contracts. No test invokes sbatch or requires Slurm/CUDA."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pergpu', ROOT / 'scripts/validate_cmi_smartsim_pergpu.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def write_csv(path, entries):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


class CampaignTests(unittest.TestCase):
    def test_shell_syntax(self):
        for name in ('run_cmi_smoke_gpu.sh', 'cmi_bench_het_engine.sh', 'cmi_node_sampler.sh',
                     'cmi_smartsim_pergpu.sh', 'submit_cmi_bench_smartsim_pergpu.sh',
                     'bench_smartsim_pergpu_single.sbatch', 'bench_smartsim_pergpu_het.sbatch'):
            subprocess.run(['bash', '-n', str(ROOT / 'slurm' / name)], check=True)

    def test_dry_run_matrix_and_pilots(self):
        for models, reps, count in [('transformer mlp', '3', 12), ('transformer', '1', 2)]:
            with tempfile.TemporaryDirectory() as temp:
                env = {**os.environ, 'DRY_RUN': '1', 'MODELS': models, 'REPS': reps,
                       'PLACEMENTS': '96c4g 96p96c4g', 'CMI_PERGPU_CAMPAIGN_ROOT': temp}
                result = subprocess.run(['bash', str(ROOT / 'slurm/submit_cmi_bench_smartsim_pergpu.sh')],
                                        env=env, text=True, capture_output=True, check=True)
                manifest = next(Path(temp).glob('*/manifest.csv'))
                entries = validator.rows(manifest)
                self.assertEqual(len(entries), count)
                self.assertEqual(len({r['cell'] for r in entries}), count)
                self.assertTrue(all(r['jobid'] == 'DRY_RUN' for r in entries))
                for row in entries:
                    self.assertEqual(row['tpq'], '4' if row['model'] == 'transformer' and row['placement'] == '96c4g' else '1')
                    self.assertEqual(row['case_timeout_seconds'], '600')
                self.assertEqual(sum(line.startswith('sbatch ') for line in result.stdout.splitlines()), count)
                with self.assertRaisesRegex(ValueError, 'dry-run'):
                    validator.validate_run(entries[0], manifest.resolve(), [])

    def test_local_config_and_gpu_probe(self):
        import tomllib
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = temp / 'config_smartsim.toml'
            config.write_bytes((ROOT / 'config_smartsim.toml').read_bytes())
            fake = temp / 'nvidia-smi'
            fake.write_text('#!/bin/bash\nprintf "2, GPU-a\\n4, GPU-b\\n6, GPU-c\\n7, GPU-d\\n"\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': str(temp) + ':' + os.environ['PATH'],
                   'SMARTSIM_DB_LAYOUT': 'per-ml-node', 'CUDA_VISIBLE_DEVICES': '2,4,6,7'}
            env.pop('SLURM_HET_SIZE', None)
            command = f'provider=SMARTSIM; source "{ROOT}/slurm/cmi_smartsim_pergpu.sh"; printf "%s\\n" "$SMARTSIM_DB_GPU_IDS" "$MAIA_SMARTSIM_NUM_GPUS" "${{smartsim_layout_args[*]}}"'
            result = subprocess.run(['bash', '-euc', command], cwd=temp, env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ['2,4,6,7', '1', '--db-layout per-ml-node'])
            library = tomllib.loads(config.read_text())['library']
            self.assertEqual([library[k] for k in ('db_layout', 'nodes', 'num_gpus', 'first_gpu')], ['per-ml-node', 4, 1, 0])
            env['CUDA_VISIBLE_DEVICES'] = '2,2,6,7'
            self.assertNotEqual(subprocess.run(['bash', '-euc', command], cwd=temp, env=env, capture_output=True).returncode, 0)
            # A heterogeneous controller must ignore the CPU-side inherited mask.
            srun = temp / 'srun'
            srun.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > probe_args\nwhile [[ "$1" != /bin/bash ]]; do shift; done\nexport CUDA_VISIBLE_DEVICES=2,4,6,7\nexec "$@"\n')
            srun.chmod(0o755)
            env['SLURM_HET_SIZE'] = '2'
            result = subprocess.run(['bash', '-euc', command], cwd=temp, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[0], '2,4,6,7')
            self.assertTrue((temp / 'probe_args').read_text().startswith('--het-group=1 --nodes=1 --ntasks=1 --cpus-per-task=1 --overlap --exact --mem=0'))
            # An allocation step with no CVD falls back to actual GPU UUIDs.
            env.pop('SLURM_HET_SIZE')
            env.pop('CUDA_VISIBLE_DEVICES')
            srun.write_text('#!/bin/bash\nwhile [[ "$1" != /bin/bash ]]; do shift; done\nunset CUDA_VISIBLE_DEVICES\nexec "$@"\n')
            result = subprocess.run(['bash', '-euc', command], cwd=temp, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines()[0], 'GPU-a,GPU-b,GPU-c,GPU-d')
            config.write_bytes((ROOT / 'config_smartsim.toml').read_bytes())
            before = config.read_bytes()
            env.pop('SMARTSIM_DB_LAYOUT')
            subprocess.run(['bash', '-euc', command], cwd=temp, env=env, capture_output=True, check=False)
            self.assertEqual(before, config.read_bytes())


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.run = self.base / 'run'
        self.run.mkdir()
        (self.base / 'status').mkdir()
        self.manifest = self.base / 'manifest.csv'
        self.row = dict(cell='smartsim-96c4g-pergpu-transformer-r1', jobid='12345', placement='96c4g',
                        model='transformer', rep='1', tpq='4', status_dir=str(self.base / 'status'), revision='revision', dry_run='0')
        write_csv(self.manifest, [self.row])
        write_csv(self.base / 'status' / f"{self.row['cell']}_12345.csv", [dict(cell=self.row['cell'], jobid='12345', status='0', run_dir=str(self.run))])
        info = dict(CELL=self.row['cell'], REP='1', JOB_ID='12345', RUN_DIR=str(self.run),
                    CAMPAIGN_MANIFEST=str(self.manifest), GIT_REVISION='revision', SOLVER_EXIT_STATUS='0', EXIT_STATUS='0',
                    RUN_STEPS='200', SOLUTION_INTERVAL='10000', ML_CUBE_OVERLAP='0', SMARTSIM_DB_LAYOUT='per-ml-node',
                    SMARTSIM_DB_NODES='4', SMARTSIM_DB_GPU_COUNT='4', MAIA_SMARTSIM_NUM_GPUS='1', SMARTSIM_TPQ='4',
                    SMARTSIM_INTRA_OP_THREADS='4', SMARTSIM_INTER_OP_THREADS='1', SMARTSIM_CPU_CORES='0', SMARTSIM_DB_STEP_MEM='0',
                    CASE_TIMEOUT_SECONDS='300', CASE='smartsim', SOLVER_RANKS='96', DL_RANKS='0', SMARTSIM_INTERFACE='lo',
                    MODEL='./input/transformer_inference_scripted_fw2.pt', BENCHMARK_SOLVER_WALL_SECONDS='100',
                     SMARTSIM_DB_GPU_IDS='2,4,6,7', SMARTSIM_DB_GPU_ORIGIN='inherited-allocation',
                     BATCH_HOST='host', HET_GPU_NODELIST='host', NODE_RAM_PEAK_MIB_host='1000')
        for index in (2, 4, 6, 7):
            info[f'GPU_VRAM_PEAK_MIB_host_gpu{index}'] = '500'
            info[f'GPU_VRAM_BASE_MIB_host_gpu{index}'] = '0'
        (self.run / 'run.info').write_text(''.join(f'{k}={v}\n' for k, v in info.items()))
        (self.run / 'config_smartsim.toml').write_text('[library]\ndb_layout="per-ml-node"\nnodes=4\nnum_gpus=1\nfirst_gpu=0\n')
        controller = self.base / 'smartsim_controller.py'
        controller.write_text('fixture controller')
        digest = hashlib.sha256(controller.read_bytes()).hexdigest()
        hashes = f'{digest}  {controller}\n'
        for name in ('maia', validator.MODELS['transformer'], 'properties.toml', 'config_smartsim.toml'):
            path = self.run / name
            if not path.exists():
                path.write_text('fixture artifact')
            hashes += f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n'
        (self.run / 'run.sha256').write_text(hashes)
        (self.base / 'source.sha256').write_text(f'{digest}  {controller}\n')
        (self.run / 'slurm_allocation.log').write_text('JobId=12345 NumNodes=1 NumTasks=96 TimeLimit=00:20:00 OverSubscribe=EXCLUSIVE AllocTRES=cpu=96,gres/gpu=4 MinMemoryNode=0\n')
        self.endpoints = [f'host:{6780+i}' for i in range(4)]
        (self.run / '.ssdb_endpoint').write_text(','.join(self.endpoints))
        log = ''.join(f'ML_STEP_TIMING step={step} ml_ms=12.3\n' for step in validator.STEPS)
        log += 'ML_WORKLOAD_LOG cubes_per_rank=1 samples_per_rank=3 cubes_aggregate=96 samples_aggregate=288 solver_ranks=96\n'
        for i, endpoint in enumerate(self.endpoints[:1]):
            log += f'SmartSim Coupling Provider initialized on rank {i*24} [db_layout=per-ml-node, db_index={i}, local_rank=0, block=[{i*24}..{i*24+24}), is_leader=true, SSDB={endpoint}]\nNumber of GPUs: 1\n'
        (self.run / 'solver.log').write_text(log)
        maps = [dict(shard=i, mask=mask, logical_device=0, port=6780+i) for i, mask in enumerate(('2','4','6','7'))]
        (self.run / 'controller.log').write_text(''.join('SMARTSIM_DB_GPU_MAP ' + json.dumps(m) + '\n' for m in maps))
        (self.run / 'smartsim_gpu_identity.log').write_text('CUDA_VISIBLE_DEVICES=2,4,6,7\n2, GPU-a\n4, GPU-b\n6, GPU-c\n7, GPU-d\n')
        (self.run / 'smartsim_gpu_processes.csv').write_text(''.join(f'100,{10+i},GPU-{u},500,redis-server *:{6780+i}\n' for i, u in enumerate('abcd')))
        self.baseline = [dict(cell=self.row['cell'].replace('-pergpu-', '-shared-'), **dict(zip(validator.FIELDS, ('1','3','96','288','96'))))]

    def validate(self):
        return validator.validate_run(self.row, self.manifest, self.baseline)

    def test_valid_run(self):
        self.assertEqual(self.validate()['inferences'], 7)

    def test_explicit_timeout_and_current_slurm_exclusive_format(self):
        self.row['case_timeout_seconds'] = '600'
        path = self.run / 'run.info'
        path.write_text(path.read_text().replace('CASE_TIMEOUT_SECONDS=300', 'CASE_TIMEOUT_SECONDS=600'))
        allocation = self.run / 'slurm_allocation.log'
        allocation.write_text(allocation.read_text().replace('OverSubscribe=EXCLUSIVE', 'OverSubscribe=NO Exclusive=NODE'))
        self.assertEqual(self.validate()['measured'], 5)
        self.row['case_timeout_seconds'] = '300'
        with self.assertRaisesRegex(ValueError, 'CASE_TIMEOUT_SECONDS'):
            self.validate()

    def test_valid_heterogeneous_run(self):
        old_cell = self.row['cell']
        self.row.update(cell=old_cell.replace('96c4g', '96p96c4g'), placement='96p96c4g', tpq='1')
        status = self.base / 'status' / f'{old_cell}_12345.csv'
        status.unlink()
        write_csv(self.base / 'status' / f"{self.row['cell']}_12345.csv", [dict(cell=self.row['cell'], jobid='12345', status='0', run_dir=str(self.run))])
        path = self.run / 'run.info'
        path.write_text(path.read_text().replace(old_cell, self.row['cell']).replace('SMARTSIM_TPQ=4', 'SMARTSIM_TPQ=1').replace('SMARTSIM_INTERFACE=lo', 'SMARTSIM_INTERFACE=ib0').replace('SMARTSIM_DB_GPU_ORIGIN=inherited-allocation', 'SMARTSIM_DB_GPU_ORIGIN=het-group-1-step') + 'PROVIDER=SMARTSIM\nHET_CPU_TASKS=96\nHET_GPU_TASKS=96\nHET_GPUS=4\n')
        component = ' NumNodes=1 NumTasks=96 TimeLimit=00:20:00 OverSubscribe=EXCLUSIVE AllocTRES=cpu=96,gres/gpu=4 MinMemoryCPU=5G\n'
        (self.run / 'slurm_allocation.log').write_text(''.join(f'JobId={12345+i} HetJobId=12345 HetJobOffset={i}' + component for i in range(2)))
        self.baseline[0]['cell'] = self.row['cell'].replace('-pergpu-', '-shared-')
        self.baseline[0].update(cubes_aggregate='192', samples_aggregate='576', workload_solver_ranks='192')
        solver = self.run / 'solver.log'
        solver.write_text(solver.read_text().replace('cubes_aggregate=96 samples_aggregate=288 solver_ranks=96', 'cubes_aggregate=192 samples_aggregate=576 solver_ranks=192'))
        self.assertEqual(self.validate()['inferences'], 7)

    def test_fail_closed_evidence(self):
        mutations = [
            ('run.info', 'JOB_ID=12345', 'JOB_ID=999', 'JOB_ID'),
            ('run.info', 'MAIA_SMARTSIM_NUM_GPUS=1', 'MAIA_SMARTSIM_NUM_GPUS=4', 'NUM_GPUS'),
            ('solver.log', 'step=189', 'step=190', 'seven'),
            ('solver.log', 'samples_aggregate=288', 'samples_aggregate=289', 'workload'),
            ('solver.log', 'Number of GPUs: 1', 'Number of GPUs: 4', 'GPU-count'),
            ('slurm_allocation.log', 'NumTasks=96', 'NumTasks=24', 'ranks'),
            ('run.info', 'GPU_VRAM_PEAK_MIB_host_gpu2=500', 'GPU_VRAM_PEAK_MIB_host_gpu2=0', 'resource'),
            ('controller.log', '"mask": "4"', '"mask": "2"', 'mask'),
            ('smartsim_gpu_processes.csv', 'GPU-d', 'GPU-c', 'simultaneous'),
            ('smartsim_gpu_processes.csv', '100,13', '101,13', 'simultaneous'),
        ]
        for name, old, new, error in mutations:
            with self.subTest(name=name, error=error):
                path = self.run / name
                original = path.read_text()
                self.assertIn(old, original)
                path.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, error):
                    self.validate()
                path.write_text(original)

    def test_foreign_status_and_missing_evidence(self):
        self.row['status_dir'] = str(self.base)
        with self.assertRaisesRegex(ValueError, 'foreign status'):
            self.validate()
        self.row['status_dir'] = str(self.base / 'status')
        (self.run / 'smartsim_gpu_processes.csv').unlink()
        with self.assertRaises(OSError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
