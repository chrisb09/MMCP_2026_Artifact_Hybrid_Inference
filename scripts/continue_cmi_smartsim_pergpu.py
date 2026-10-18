#!/usr/bin/env python3
"""Watch the two designated Transformer pilots, then launch ten remaining runs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATE_NAME = 'continuation_state.json'
LOCK_NAME = 'continuation.lock'
PLACEMENTS = '96c4g 96p96c4g'
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    state['updated_at'] = stamp()
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def pilot_ids(manifest):
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        raise ValueError('expected exactly two pilot cells')
    expected = {f'smartsim-{p}-pergpu-transformer-r1' for p in PLACEMENTS.split()}
    if {r['cell'] for r in rows} != expected:
        raise ValueError('expected Transformer r1 pilots at both placements')
    for row in rows:
        p = row['placement']
        if (p not in PLACEMENTS.split() or row['model'] != 'transformer' or
                row['rep'] != '1' or row['cell'] != f'smartsim-{p}-pergpu-transformer-r1' or
                row['dry_run'] != '0' or not re.fullmatch(r'[1-9][0-9]*', row['jobid'])):
            raise ValueError('foreign or dry-run pilot manifest row')
    ids = [r['jobid'] for r in rows]
    if len(set(ids)) != 2:
        raise ValueError('duplicate pilot job IDs')
    return ids


def retry_pilot(manifest, retry_manifest, original_ids):
    if retry_manifest.parent == manifest.parent:
        raise ValueError('retry manifest must have a dedicated campaign directory')
    with retry_manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError('expected exactly one retry pilot cell')
    row = rows[0]
    if (row['cell'] != 'smartsim-96p96c4g-pergpu-transformer-r1' or
            row['placement'] != '96p96c4g' or row['model'] != 'transformer' or
            row['rep'] != '1' or row['dry_run'] != '0' or row['tpq'] != '1' or
            not re.fullmatch(r'[1-9][0-9]*', row['jobid'])):
        raise ValueError('foreign or dry-run retry pilot manifest row')
    # A replacement must be newly submitted, not an original or older campaign job.
    if int(row['jobid']) <= max(map(int, original_ids)):
        raise ValueError('retry job ID must be newer than both original pilots')
    return row


def validate_pilots(manifest, retry_manifest=None):
    # Load the current sibling source, independent of cwd and sys.path.
    spec = importlib.util.spec_from_file_location(
        'continuation_pilot_validator', Path(__file__).with_name('validate_cmi_smartsim_pergpu.py'))
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    baseline_path = ROOT / 'results/cmi_bench_consolidated/runs.csv'
    if retry_manifest is None:
        return validator.validate(manifest, baseline_path, expected_cells=2)
    original_ids = pilot_ids(manifest)
    retry = retry_pilot(manifest, retry_manifest, original_ids)
    selected = [(r, manifest) for r in validator.rows(manifest) if r['placement'] == '96c4g']
    selected.append((retry, retry_manifest))
    baseline = validator.rows(baseline_path)
    validator.require(len(baseline) == 288 and len({r['cell'] for r in baseline}) == 288,
                      'expected preserved canonical 288-run baseline')
    accepted, errors = [], []
    for row, source in selected:
        try:
            accepted.append(validator.validate_run(row, source, baseline))
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            errors.append(dict(cell=row.get('cell'), manifest=str(source), error=str(exc)))
    return dict(manifest=str(manifest), retry_manifest=str(retry_manifest),
                baseline=str(baseline_path.resolve()), accepted=accepted, errors=errors,
                valid=not errors and len(accepted) == 2)


def run(args):
    manifest = args.manifest.resolve()
    original_ids = pilot_ids(manifest)
    ids = original_ids.copy()
    retry_manifest = args.pilot_retry_manifest.resolve() if args.pilot_retry_manifest else None
    provenance = {}
    if retry_manifest is not None:
        retry = retry_pilot(manifest, retry_manifest, original_ids)
        with manifest.open(newline='') as stream:
            original_rows = list(csv.DictReader(stream))
        ids = [retry['jobid'] if r['placement'] == '96p96c4g' else r['jobid'] for r in original_rows]
        provenance = dict(original_manifest=str(manifest), retry_manifest=str(retry_manifest),
                          original_jobids=original_ids, effective_jobids=ids,
                          retry_authorization='Explicit --pilot-retry-manifest approval of failed heterogeneous pilot replacement; original failure reviewed by caller, not re-polled')
    state_directory = retry_manifest.parent if retry_manifest else manifest.parent
    state_path = state_directory / STATE_NAME
    lock_path = state_directory / LOCK_NAME
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(f'lock exists: {lock_path}; manual review required')
    state = None
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(dict(pid=os.getpid(), created_at=stamp()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (state.get('manifest') != str(manifest) or state.get('jobids') != ids or
                    any(state.get(k) != v for k, v in provenance.items())):
                state = None
                raise ValueError('continuation state does not match pilot manifest')
            if state.get('launch_started') or any(b.get('started') for b in state.get('batches', [])):
                state = None  # Do not overwrite evidence from an earlier launch.
                raise ValueError('launch already started; manual review required, never retry automatically')
            retryable_query = (state.get('status') == 'blocked' and
                               str(state.get('error', '')).startswith('sacct failed with exit ') and
                               state.get('launch_started') is False)
            if state.get('status') in ('blocked', 'timed_out') and not retryable_query:
                state = None
                raise ValueError('previous continuation blocked or timed out; manual review required')
        else:
            state = dict(manifest=str(manifest), jobids=ids, created_at=stamp(),
                         deadline_epoch=time.time() + args.timeout_seconds, batches=[], launch_started=False,
                         **provenance)
        state.update(pid=os.getpid(), invoked_at=stamp())
        # A subsequent invocation may shorten, but never extend, the original deadline.
        state['deadline_epoch'] = min(state['deadline_epoch'], time.time() + args.timeout_seconds)
        while True:
            remaining = state['deadline_epoch'] - time.time()
            if remaining <= 0:
                state.update(status='timed_out', error='pilot continuation deadline exceeded')
                save(state_path, state)
                return 1
            command = ['sacct', '-n', '-P', '-j', ','.join(ids), '--format=JobIDRaw,State,ExitCode']
            state.update(status='checking', sacct_command=command)
            save(state_path, state)
            result = subprocess.run(command, capture_output=True, text=True, check=False,
                                    timeout=min(60, remaining))
            state.update(sacct_stdout=result.stdout, sacct_stderr=result.stderr, checked_at=stamp())
            if result.returncode:
                state.update(status='pending_retry', error=f'sacct failed with exit {result.returncode}',
                             sacct_failure_count=state.get('sacct_failure_count', 0) + 1,
                             last_sacct_error=f'sacct failed with exit {result.returncode}',
                             last_sacct_stderr=result.stderr)
                save(state_path, state)
                if args.once:
                    return 0
                time.sleep(min(60, args.poll_seconds, max(0, state['deadline_epoch'] - time.time())))
                continue
            state.pop('error', None)
            jobs = {}
            for line in result.stdout.splitlines():
                fields = [v.strip() for v in line.split('|')]
                if not fields or fields[0] not in ids:
                    continue  # Ignore steps (.batch/.extern) and het components (+N).
                if len(fields) < 3 or fields[0] in jobs:
                    raise ValueError('malformed or duplicate exact parent sacct row')
                jobs[fields[0]] = dict(state=fields[1], exit_code=fields[2])
            state['pilot_states'] = jobs
            for job in jobs.values():
                if job['state'] == 'COMPLETED' and job['exit_code'] == '0:0':
                    continue
                if job['state'] not in ACTIVE:
                    raise ValueError(f'pilot not successful: {job}')
            if len(jobs) == 2 and all(j['state'] == 'COMPLETED' for j in jobs.values()):
                break
            state['status'] = 'pending'
            save(state_path, state)
            if args.once:
                return 0
            time.sleep(min(args.poll_seconds, max(0, state['deadline_epoch'] - time.time())))

        report = validate_pilots(manifest, retry_manifest) if retry_manifest else validate_pilots(manifest)
        state['validation_report'] = report
        save(state_path, state)
        if report.get('valid') is not True or report.get('errors') or len(report.get('accepted', [])) != 2:
            raise ValueError('pilot validation failed; no continuation submitted')
        for model, first_rep, count in [('transformer', '2', 4), ('mlp', '1', 6)]:
            remaining = state['deadline_epoch'] - time.time()
            if remaining <= 0:
                raise ValueError('deadline exceeded before submission; manual review required')
            env = dict(os.environ, MODELS=model, FIRST_REP=first_rep, REPS='3',
                        PLACEMENTS=PLACEMENTS, DRY_RUN='0', CASE_TIMEOUT_SECONDS='600',
                       CMI_PERGPU_CAMPAIGN_ROOT=str(manifest.parent.parent))
            command = ['bash', str(ROOT / 'slurm/submit_cmi_bench_smartsim_pergpu.sh')]
            batch = dict(model=model, expected_runs=count, command=command,
                          environment={k: env[k] for k in ('MODELS', 'FIRST_REP', 'REPS', 'PLACEMENTS', 'DRY_RUN', 'CASE_TIMEOUT_SECONDS', 'CMI_PERGPU_CAMPAIGN_ROOT')},
                         started=True, started_at=stamp())
            state['batches'].append(batch)
            state.update(launch_started=True, status='submitting')
            # This durable write must precede every potentially partial submission.
            save(state_path, state)
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, check=False, timeout=remaining)
            batch.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                         finished_at=stamp(), submitted_manifest_paths=re.findall(
                             r'^Campaign manifest: (.+)$', result.stdout, re.MULTILINE))
            save(state_path, state)
            if result.returncode or len(batch['submitted_manifest_paths']) != 1:
                raise ValueError(f'{model} submission failed or ambiguous; manual review required')
        state.update(status='submitted', finished_at=stamp(), total_runs_including_pilots=12,
                     submitted_manifest_paths=[p for b in state['batches'] for p in b['submitted_manifest_paths']])
        save(state_path, state)
        return 0
    except Exception as exc:
        if state is not None:
            state.update(status='blocked', error=str(exc))
            if isinstance(exc, subprocess.TimeoutExpired):
                state['timeout_stdout'] = str(exc.stdout or '')
                state['timeout_stderr'] = str(exc.stderr or '')
            save(state_path, state)
        raise
    finally:
        lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot-retry-manifest', type=Path,
                        help='Explicit approval to replace the failed heterogeneous r1 pilot with a new one-row campaign; state/lock live in its parent, original failure is not re-polled')
    parser.add_argument('--once', action='store_true', help='Exit successfully if pending; submit if gates pass')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--timeout-seconds', type=float, default=86400)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive and finite')
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 86400:
        parser.error('--timeout-seconds must be positive and at most 86400 (24h)')
    try:
        return run(args)
    except Exception as exc:
        print(f'Continuation refused/blocked: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
