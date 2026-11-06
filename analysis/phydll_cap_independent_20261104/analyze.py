#!/usr/bin/env python3
"""Reproduce the PhyDLL Transformer cap comparison from local artifacts."""
import csv
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_phydll_cap50000_reruns as validator

STEPS = (15, 44, 73, 102, 131, 160, 189)
T = 4.302652729911275


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    campaign = ROOT / 'results/phydll_cap50000_reruns/campaign_20261028T180740Z_n5AUNF'
    manifest = read_csv(campaign / 'manifest.csv')
    originals = read_csv(ROOT / 'results/cmi_bench_consolidated/runs.csv')
    canonical = {r['cell']: r for r in originals}
    assert len(canonical) == len(originals) and len(manifest) == 24
    assert len({r['cell'] for r in manifest}) == 24
    observations, checks = [], []
    for entry in manifest:
        assert entry['model'] == 'transformer' and entry['cap'] == '50000'
        checks.append(validator.validate(entry, canonical))
        old = canonical[entry['cell'].replace('-cap50000', '')]
        assert old['provider'] == 'phydll' and old['model'] == 'transformer'
        assert all(old[k] == entry[k] for k in ('rep', 'placement', 'wait'))
        for cap, run, jobid in ((2048, ROOT / old['run_dir'], old['jobid']),
                                (50000, validator.resolve_run(entry), entry['jobid'])):
            info = validator.info(run / 'run.info')
            assert info['CELL'] == (old['cell'] if cap == 2048 else entry['cell'])
            assert info['REP'] == entry['rep'] and info['MAIA_PHYDLL_BATCHSIZE'] == str(cap)
            assert info['MAIA_ML_STEP_TIMING_SYNC'] == info['MAIA_ML_STEP_TIMING_PRINT'] == '1'
            assert info['EXIT_STATUS'] == info['SOLVER_EXIT_STATUS'] == '0'
            text = (run / 'solver.log').read_text(errors='replace')
            matches = re.findall(r'ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)', text)
            assert tuple(int(s) for s, _ in matches) == STEPS
            values = [float(v) for _, v in matches[2:]]
            assert all(math.isfinite(v) and v > 0 for v in values)
            if cap == 2048:
                assert old['n_measured'] == '5' and values == [float(v) for v in old['measured_ms'].split(';')]
            wall = float(info['BENCHMARK_SOLVER_WALL_SECONDS'])
            assert math.isfinite(wall) and wall > 0
            observations.append(dict(placement=entry['placement'], wait=entry['wait'], rep=entry['rep'], cap=cap,
                                     jobid=jobid, run_dir=str(run.relative_to(ROOT)),
                                     measured_ms=';'.join(map(str, values)), mean_ms=statistics.mean(values), wall_seconds=wall))
    summary = []
    for placement in sorted({r['placement'] for r in observations}):
        for wait in ('default', 'readiness'):
            row = dict(placement=placement, wait=wait, n=3)
            for metric in ('mean_ms', 'wall_seconds'):
                means = []
                for cap in (2048, 50000):
                    group = [r for r in observations if r['placement'] == placement and r['wait'] == wait and r['cap'] == cap]
                    assert len(group) == 3 and {r['rep'] for r in group} == {'1', '2', '3'}
                    assert len({r['jobid'] for r in group}) == 3
                    vals = [r[metric] for r in group]
                    mean = statistics.mean(vals)
                    half = T * statistics.stdev(vals) / math.sqrt(3)
                    prefix = f'{metric}_cap{cap}'
                    row.update({prefix: mean, prefix + '_ci_low': mean-half, prefix + '_ci_high': mean+half})
                    means.append(mean)
                row[metric + '_change_pct'] = 100 * (means[1] / means[0] - 1)
            summary.append(row)
    assert len(summary) == 8 and len(observations) == 48
    # Inspect all individual replay records, not only the six block summaries.
    correctness = []
    for block in range(6):
        log = (ROOT / f'logs/phydll_cap_check_5300338_{block}.log').read_text()
        assert 'Checked: 8  Failures: 0' in log and 'OVERALL: PASS' in log
        for kind, pattern in [('caps', r'cap2048 vs cap50000'), ('pilot_ref', r'cap50000 vs pilot_ref')]:
            matches = re.findall(r'\[rank=(\d+) inf=(\d+)\] ' + pattern + r': (\w+)\s+max_abs=(\S+)\s+rmse=(\S+)\s+rel=(\S+)', log)
            assert len(matches) == 8
            for rank, inf, status, max_abs, rmse, rel in matches:
                assert int(rank) in range(4*block, 4*block+4) and int(inf) in (1, 2)
                correctness.append(dict(block=block, rank=int(rank), inference=int(inf), comparison=kind,
                                        status=status, max_abs=float(max_abs), rmse=float(rmse), relative=float(rel)))
    for kind in ('caps', 'pilot_ref'):
        rows = [r for r in correctness if r['comparison'] == kind]
        assert len({(r['rank'], r['inference']) for r in rows}) == 48
    write_csv('observations.csv', observations)
    write_csv('validation.csv', checks)
    write_csv('summary.csv', summary)
    write_csv('correctness_records.csv', correctness)
    print('placement wait | cap2048 mean [95% CI] | cap50000 mean [95% CI] | change %')
    for r in summary:
        def show(cap):
            p = f'mean_ms_cap{cap}'
            return f"{r[p]:.2f} [{r[p+'_ci_low']:.2f}, {r[p+'_ci_high']:.2f}]"
        print(f"{r['placement']} {r['wait']} | {show(2048)} | {show(50000)} | {r['mean_ms_change_pct']:+.2f}%")
    print('correctness statuses:', {kind: {s: sum(r['comparison'] == kind and r['status'] == s for r in correctness) for s in ('PASS', 'WARN', 'FAIL')} for kind in ('caps', 'pilot_ref')})
    print('pilot max abs:', max(r['max_abs'] for r in correctness if r['comparison'] == 'pilot_ref'))


if __name__ == '__main__':
    main()
