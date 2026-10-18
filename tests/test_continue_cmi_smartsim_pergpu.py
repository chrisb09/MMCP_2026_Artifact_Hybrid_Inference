"""Offline watchdog tests: all scheduler and submitter calls are mocked."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('continuation', ROOT / 'scripts/continue_cmi_smartsim_pergpu.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.manifest = self.directory / 'manifest.csv'
        self.rows = [dict(cell=f'smartsim-{p}-pergpu-transformer-r1', jobid=j,
                          placement=p, model='transformer', rep='1', dry_run='0')
                     for p, j in [('96c4g', '5109330'), ('96p96c4g', '5109331')]]
        self.write_manifest()
        self.argv = ['--manifest', str(self.manifest), '--once']
        self.completed = '5109330|COMPLETED|0:0\n5109331|COMPLETED|0:0\n'
        self.report = dict(valid=True, errors=[], accepted=[{'cell': r['cell']} for r in self.rows])

    def write_manifest(self):
        with self.manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def state(self):
        return json.loads((self.directory / watch.STATE_NAME).read_text())

    def result(self, stdout, code=0, stderr=''):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def test_completed_gates_and_command_snapshot(self):
        def execute(command, **kwargs):
            if command[0] == 'sacct':
                self.assertEqual(command, ['sacct', '-n', '-P', '-j', '5109330,5109331', '--format=JobIDRaw,State,ExitCode'])
                return self.result(self.completed + '5109331+0|FAILED|1:0\n5109330.batch|FAILED|1:0\n999|FAILED|1:0\n')
            batch = self.state()['batches'][-1]
            self.assertTrue(batch['started'])
            self.assertTrue(self.state()['launch_started'])
            self.assertEqual(command, ['bash', str(ROOT / 'slurm/submit_cmi_bench_smartsim_pergpu.sh')])
            env = kwargs['env']
            self.assertEqual((env['REPS'], env['PLACEMENTS'], env['DRY_RUN']), ('3', '96c4g 96p96c4g', '0'))
            self.assertEqual(env['FIRST_REP'], '2' if env['MODELS'] == 'transformer' else '1')
            return self.result(f"Campaign manifest: /fake/{env['MODELS']}/manifest.csv\n")
        with patch.object(watch.subprocess, 'run', side_effect=execute) as proc, patch.object(watch, 'validate_pilots', return_value=self.report) as validate:
            self.assertEqual(watch.main(self.argv), 0)
            validate.assert_called_once_with(self.manifest)
            self.assertEqual(proc.call_count, 3)
            self.assertEqual([b['expected_runs'] for b in self.state()['batches']], [4, 6])
            self.assertEqual(self.state()['status'], 'submitted')
            self.assertEqual(len(self.state()['submitted_manifest_paths']), 2)
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 3)

    def test_pending_once_and_missing_parent(self):
        for output in ('', '5109330|COMPLETED|0:0\n', '5109330|PENDING|0:0\n5109331|RUNNING|0:0\n', '5109330.batch|COMPLETED|0:0\n5109331+0|COMPLETED|0:0\n'):
            with patch.object(watch.subprocess, 'run', return_value=self.result(output)) as proc, patch.object(watch, 'validate_pilots') as validate:
                self.assertEqual(watch.main(self.argv), 0)
                self.assertEqual(self.state()['status'], 'pending')
                self.assertEqual(proc.call_count, 1)
                validate.assert_not_called()

    def test_failed_pilot_blocks_and_never_retries(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result('5109330|FAILED|1:0\n5109331|PENDING|0:0\n')) as proc, patch.object(watch, 'validate_pilots') as validate:
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(self.state()['status'], 'blocked')
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 1)
            validate.assert_not_called()

    def test_nonzero_completed_exit_blocks(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result(self.completed.replace('0:0', '0:9', 1))) as proc:
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 1)

    def test_validation_failure_blocks(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result(self.completed)) as proc, patch.object(watch, 'validate_pilots', return_value=dict(valid=False, errors=['bad'], accepted=[])):
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 1)
            self.assertFalse(self.state()['validation_report']['valid'])
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 1)

    def test_partial_submission_failure_is_not_retried(self):
        with patch.object(watch.subprocess, 'run', side_effect=[self.result(self.completed), self.result('partial stdout', 1)]) as proc, patch.object(watch, 'validate_pilots', return_value=self.report):
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(self.state()['batches'][0]['stdout'], 'partial stdout')
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 2)

    def test_lock_refuses_without_subprocess(self):
        (self.directory / watch.LOCK_NAME).write_text('existing owner')
        with patch.object(watch.subprocess, 'run') as proc:
            self.assertEqual(watch.main(self.argv), 1)
            proc.assert_not_called()
        self.assertEqual((self.directory / watch.LOCK_NAME).read_text(), 'existing owner')

    def test_submission_timeout_retains_started_flag(self):
        with patch.object(watch.subprocess, 'run', side_effect=[self.result(self.completed), subprocess.TimeoutExpired(['bash'], 5, output='partial')]) as proc, patch.object(watch, 'validate_pilots', return_value=self.report):
            self.assertEqual(watch.main(self.argv), 1)
            self.assertTrue(self.state()['batches'][0]['started'])
            self.assertEqual(self.state()['timeout_stdout'], 'partial')
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(proc.call_count, 2)

    def test_sacct_error_once_pending_then_completed_gates(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result(self.completed, 1, 'connection refused')) as proc, patch.object(watch, 'validate_pilots') as validate, patch.object(watch.time, 'sleep') as sleep:
            self.assertEqual(watch.main(self.argv), 0)
            state = self.state()
            self.assertEqual(state['status'], 'pending_retry')
            self.assertEqual(state['sacct_stderr'], 'connection refused')
            self.assertEqual(state['sacct_failure_count'], 1)
            self.assertFalse(state['launch_started'])
            self.assertEqual(state['batches'], [])
            self.assertEqual(proc.call_count, 1)
            validate.assert_not_called()
            sleep.assert_not_called()
        for output in ('', '5109330|COMPLETED|0:0\n'):
            with patch.object(watch.subprocess, 'run', return_value=self.result(output)) as proc, patch.object(watch, 'validate_pilots') as validate:
                self.assertEqual(watch.main(self.argv), 0)
                self.assertEqual(self.state()['status'], 'pending')
                self.assertFalse(self.state()['launch_started'])
                self.assertEqual(proc.call_count, 1)
                validate.assert_not_called()
        with patch.object(watch.subprocess, 'run', side_effect=[self.result(self.completed), self.result('Campaign manifest: /fake/transformer/manifest.csv\n'), self.result('Campaign manifest: /fake/mlp/manifest.csv\n')]) as proc, patch.object(watch, 'validate_pilots', return_value=self.report) as validate:
            self.assertEqual(watch.main(self.argv), 0)
            self.assertEqual(self.state()['status'], 'submitted')
            self.assertEqual(self.state()['total_runs_including_pilots'], 12)
            self.assertEqual(proc.call_count, 3)
            validate.assert_called_once_with(self.manifest)
        self.assertEqual(self.state()['deadline_epoch'], state['deadline_epoch'])
        self.assertEqual(self.state()['last_sacct_stderr'], 'connection refused')
        self.assertEqual(self.state()['sacct_failure_count'], 1)

    def test_sacct_retries_bounded_by_original_deadline(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result('', 1, 'connection refused')) as proc, patch.object(watch, 'validate_pilots') as validate, patch.object(watch.time, 'time', return_value=1000) as clock, patch.object(watch.time, 'sleep') as sleep:
            def advance(seconds):
                clock.return_value += seconds
            sleep.side_effect = advance
            self.assertEqual(watch.main(['--manifest', str(self.manifest), '--poll-seconds', '600', '--timeout-seconds', '90']), 1)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 30])
            self.assertEqual(proc.call_count, 2)
            self.assertEqual(self.state()['sacct_failure_count'], 2)
            self.assertEqual(self.state()['status'], 'timed_out')
            self.assertEqual(self.state()['deadline_epoch'], 1090)
            self.assertFalse(self.state()['launch_started'])
            validate.assert_not_called()

    def test_previous_sacct_block_resumes_only_before_launch(self):
        original = dict(manifest=str(self.manifest), jobids=['5109330', '5109331'],
                        deadline_epoch=1090, batches=[], launch_started=False,
                        status='blocked', error='sacct failed with exit 1')
        cases = [({}, True), ({'error': 'other sacct failed with exit 1'}, False),
                 ({'error': 'sacct failed with exit'}, False),
                 ({'error': 'pilot validation failed; no continuation submitted'}, False),
                 ({'launch_started': True}, False), ({'launch_started': None}, False),
                 ({'batches': [{'started': True}]}, False), ({'status': 'timed_out'}, False)]
        for changes, allowed in cases:
            with self.subTest(changes=changes):
                state = dict(original, **changes)
                watch.save(self.directory / watch.STATE_NAME, state)
                with patch.object(watch.time, 'time', return_value=1000), patch.object(watch.subprocess, 'run', return_value=self.result('')) as proc, patch.object(watch, 'validate_pilots') as validate:
                    self.assertEqual(watch.main(self.argv), 0 if allowed else 1)
                    self.assertEqual(proc.call_count, 1 if allowed else 0)
                    self.assertEqual(self.state()['deadline_epoch'], 1090)
                    self.assertEqual(self.state()['status'], 'pending' if allowed else state['status'])
                    validate.assert_not_called()

    def test_previous_sacct_block_cannot_extend_expired_deadline(self):
        watch.save(self.directory / watch.STATE_NAME,
                   dict(manifest=str(self.manifest), jobids=['5109330', '5109331'],
                        deadline_epoch=900, batches=[], launch_started=False,
                        status='blocked', error='sacct failed with exit 1'))
        with patch.object(watch.time, 'time', return_value=1000), patch.object(watch.subprocess, 'run') as proc:
            self.assertEqual(watch.main(self.argv), 1)
            self.assertEqual(self.state()['status'], 'timed_out')
            self.assertEqual(self.state()['deadline_epoch'], 900)
            proc.assert_not_called()

    def test_invalid_manifest_rejected(self):
        for key, value in [('cell', 'foreign'), ('dry_run', '1'), ('jobid', '5109331'), ('model', 'mlp')]:
            original = self.rows[0][key]
            self.rows[0][key] = value
            self.write_manifest()
            with patch.object(watch.subprocess, 'run') as proc:
                self.assertEqual(watch.main(self.argv), 1)
                proc.assert_not_called()
            self.rows[0][key] = original

    def test_timeout_and_poll(self):
        with patch.object(watch.subprocess, 'run', return_value=self.result('')) as proc, patch.object(watch.time, 'sleep') as sleep:
            def expire(_):
                state = self.state()
                state['deadline_epoch'] = 0
                # Simulate elapsed wall time without waiting.
                clock.return_value = 100000000000
            sleep.side_effect = expire
            with patch.object(watch.time, 'time', return_value=1000) as clock:
                self.assertEqual(watch.main(['--manifest', str(self.manifest)]), 1)
            sleep.assert_called_once_with(60)
            self.assertEqual(proc.call_count, 1)
            self.assertEqual(self.state()['status'], 'timed_out')

    def test_dynamic_validator_uses_current_sibling(self):
        with patch('importlib.util.spec_from_file_location', wraps=importlib.util.spec_from_file_location) as load:
            # An empty baseline need not validate; this exercises real dynamic loading.
            with patch.object(Path, 'open', side_effect=OSError('offline fixture')):
                with self.assertRaises(OSError):
                    watch.validate_pilots(self.manifest)
            self.assertEqual(Path(load.call_args.args[1]), ROOT / 'scripts/validate_cmi_smartsim_pergpu.py')

    def make_retry(self, **changes):
        directory = self.directory / 'retry'
        directory.mkdir(exist_ok=True)
        manifest = directory / 'manifest.csv'
        row = dict(self.rows[1], jobid='5109400', tpq='1', **changes)
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return manifest, row

    def test_retry_isolates_state_and_polls_effective_parent_ids(self):
        retry, _ = self.make_retry()
        original_state = self.directory / watch.STATE_NAME
        original_state.write_text('{"status": "blocked", "error": "failed original"}\n')
        original_lock = self.directory / watch.LOCK_NAME
        original_lock.write_text('original evidence')
        original_bytes = self.manifest.read_bytes(), original_state.read_bytes(), original_lock.read_bytes()
        argv = self.argv + ['--pilot-retry-manifest', str(retry)]

        def execute(command, **kwargs):
            if command[0] == 'sacct':
                self.assertEqual(command[4], '5109330,5109400')
                return self.result('5109330|COMPLETED|0:0\n5109331|FAILED|124:0\n'
                                   '5109400|COMPLETED|0:0\n5109400+0|COMPLETED|0:0\n'
                                   '5109400+1|COMPLETED|0:0\n5109400.batch|FAILED|1:0\n')
            env = kwargs['env']
            self.assertEqual(env['CASE_TIMEOUT_SECONDS'], '600')
            self.assertEqual(env['FIRST_REP'], '2' if env['MODELS'] == 'transformer' else '1')
            self.assertEqual(env['REPS'], '3')
            return self.result(f"Campaign manifest: /fake/{env['MODELS']}/manifest.csv\n")

        with patch.object(watch.subprocess, 'run', side_effect=execute) as proc, patch.object(watch, 'validate_pilots', return_value=self.report) as validate, patch.dict(watch.os.environ, CASE_TIMEOUT_SECONDS='300'):
            self.assertEqual(watch.main(argv), 0)
            validate.assert_called_once_with(self.manifest, retry)
            self.assertEqual(proc.call_count, 3)
            self.assertEqual(watch.main(argv), 1)
            self.assertEqual(proc.call_count, 3)
        state = json.loads((retry.parent / watch.STATE_NAME).read_text())
        self.assertEqual(state['original_manifest'], str(self.manifest))
        self.assertEqual(state['retry_manifest'], str(retry))
        self.assertEqual(state['original_jobids'], ['5109330', '5109331'])
        self.assertEqual(state['effective_jobids'], ['5109330', '5109400'])
        self.assertIn('Explicit --pilot-retry-manifest approval', state['retry_authorization'])
        self.assertEqual([b['model'] for b in state['batches']], ['transformer', 'mlp'])
        self.assertEqual([b['environment']['CASE_TIMEOUT_SECONDS'] for b in state['batches']], ['600', '600'])
        self.assertEqual(original_bytes, (self.manifest.read_bytes(), original_state.read_bytes(), original_lock.read_bytes()))
        self.assertFalse((retry.parent / watch.LOCK_NAME).exists())

    def test_reject_foreign_retry_without_scheduler_calls(self):
        retry, row = self.make_retry()
        cases = [('cell', self.rows[0]['cell']), ('placement', '96c4g'), ('model', 'mlp'),
                 ('rep', '2'), ('dry_run', '1'), ('tpq', '4'), ('jobid', '5109330'),
                 ('jobid', '5109331'), ('jobid', '5100000'), ('jobid', '5109400+0')]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                changed = dict(row, **{key: value})
                with retry.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(changed))
                    writer.writeheader()
                    writer.writerow(changed)
                with patch.object(watch.subprocess, 'run') as proc:
                    self.assertEqual(watch.main(self.argv + ['--pilot-retry-manifest', str(retry)]), 1)
                    proc.assert_not_called()
                self.assertFalse((retry.parent / watch.STATE_NAME).exists())
        with patch.object(watch.subprocess, 'run') as proc:
            self.assertEqual(watch.main(self.argv + ['--pilot-retry-manifest', str(self.manifest)]), 1)
            proc.assert_not_called()

    def test_retry_validation_preserves_each_manifest_provenance(self):
        retry, retry_row = self.make_retry()
        baseline_path = ROOT / 'results/cmi_bench_consolidated/runs.csv'
        baseline = [dict(cell=str(i)) for i in range(288)]
        validator = SimpleNamespace()
        validator.rows = lambda path: baseline if path == baseline_path else self.rows
        validator.require = lambda condition, message: None if condition else self.fail(message)
        calls = []

        def validate_run(row, manifest, actual_baseline):
            calls.append((row, manifest, actual_baseline))
            self.assertIs(actual_baseline, baseline)
            if row['placement'] == '96p96c4g':
                self.assertEqual(manifest, retry)
                self.assertEqual(row, retry_row)
            else:
                self.assertEqual(manifest, self.manifest)
            return dict(cell=row['cell'], jobid=row['jobid'])

        validator.validate_run = validate_run
        loader = SimpleNamespace(exec_module=lambda module: None)
        with patch.object(watch.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)), patch.object(watch.importlib.util, 'module_from_spec', return_value=validator):
            report = watch.validate_pilots(self.manifest, retry)
            self.assertTrue(report['valid'])
            self.assertEqual(len(report['accepted']), 2)
            self.assertEqual(report['errors'], [])
            self.assertEqual(len(calls), 2)
            validator.validate_run = lambda *args: (_ for _ in ()).throw(ValueError('provenance mismatch'))
            report = watch.validate_pilots(self.manifest, retry)
            self.assertFalse(report['valid'])
            self.assertEqual(report['accepted'], [])
            self.assertEqual([e['manifest'] for e in report['errors']], [str(self.manifest), str(retry)])

    def test_retry_requires_one_row_and_valid_original_manifest(self):
        retry, row = self.make_retry()
        with retry.open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=list(row)).writerow(row)
        with patch.object(watch.subprocess, 'run') as proc:
            self.assertEqual(watch.main(self.argv + ['--pilot-retry-manifest', str(retry)]), 1)
            proc.assert_not_called()
        self.make_retry()
        self.rows[0]['model'] = 'mlp'
        self.write_manifest()
        with patch.object(watch.subprocess, 'run') as proc:
            self.assertEqual(watch.main(self.argv + ['--pilot-retry-manifest', str(retry)]), 1)
            proc.assert_not_called()

    def test_retry_duplicate_exact_parent_fails_closed(self):
        retry, _ = self.make_retry()
        output = '5109330|COMPLETED|0:0\n5109400|COMPLETED|0:0\n5109400|COMPLETED|0:0\n'
        with patch.object(watch.subprocess, 'run', return_value=self.result(output)) as proc, patch.object(watch, 'validate_pilots') as validate:
            self.assertEqual(watch.main(self.argv + ['--pilot-retry-manifest', str(retry)]), 1)
            proc.assert_called_once()
            validate.assert_not_called()
        state = json.loads((retry.parent / watch.STATE_NAME).read_text())
        self.assertEqual(state['status'], 'blocked')
        self.assertFalse(state['launch_started'])


if __name__ == '__main__':
    unittest.main()
