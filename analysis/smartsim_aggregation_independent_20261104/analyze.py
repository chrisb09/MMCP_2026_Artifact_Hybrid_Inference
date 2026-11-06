#!/usr/bin/env python3
"""Reproduce the matched-workload SmartSim aggregation comparison (SciPy required)."""
import csv
import math
from pathlib import Path
import re
import statistics
import sys
import tomllib

from scipy.stats import t, ttest_ind

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import validate_smartsim_aggregation_reruns as validator

CAMPAIGNS = [
    ROOT / 'results/smartsim_aggregation_reruns/campaign_20261103T181527Z_7tpvSF/manifest.csv',
    ROOT / 'results/smartsim_aggregation_reruns/campaign_20261103T181527Z_VGwZo0/manifest.csv',
]
STEPS = (15, 44, 73, 102, 131, 160, 189)
MEASURED = STEPS[2:]
PLACEMENTS = ('24c1g', '24p24c1g', '96c4g', '96p96c4g')


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(name, data):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def timings(path):
    found = re.findall(r'ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)', path.read_text(errors='replace'))
    assert tuple(int(s) for s, _ in found) == STEPS, path
    values = [float(v) for _, v in found]
    assert all(math.isfinite(v) and v > 0 for v in values), path
    return dict(zip(STEPS, values))


def ci(values):
    m = statistics.mean(values)
    half = t.ppf(.975, len(values)-1) * statistics.stdev(values) / math.sqrt(len(values))
    return m, m-half, m+half


def main():
    canonical_rows = rows(ROOT / 'results/cmi_bench_consolidated/runs.csv')
    canonical = {r['cell']: r for r in canonical_rows}
    assert len(canonical) == len(canonical_rows)
    inference = rows(ROOT / 'results/cmi_bench_consolidated/inferences.csv')
    manifests = [r for path in CAMPAIGNS for r in rows(path)]
    expected = {(m, p, str(r)) for m in ('mlp', 'transformer') for p in PLACEMENTS for r in range(1, 4)}
    assert len(manifests) == 24 and {(r['model'], r['placement'], r['rep']) for r in manifests} == expected
    assert len({r['jobid'] for r in manifests}) == 24
    raw = []
    checks = []
    for row in manifests:
        m, p, rep = row['model'], row['placement'], row['rep']
        assert row['dry_run'] == '0' and row['cell'] == f'smartsim-{p}-agg-{m}-r{rep}'
        assert (row['batch_size'], row['min_batch_size'], row['timeout_ms']) == (('45000', '45000', '1') if m == 'mlp' else ('7000', '7000', '5'))
        validator.validate(row, canonical, ROOT / 'scratch/smartsim_aggregation_reruns')
        run = validator.find_run(ROOT / 'scratch/smartsim_aggregation_reruns', row['cell'], row['jobid'])
        ref = canonical[row['cell'].replace('-agg-', '-shared-')]
        assert (ref['provider'], ref['layout'], ref['placement'], ref['model'], ref['rep']) == ('smartsim', 'shared', p, m, rep)
        assert ref['exit_status'] == ref['solver_exit_status'] == '0'
        refdir = ROOT / ref['run_dir']
        assert (refdir / 'solver.log').is_file() and (refdir / 'properties.toml').is_file() and (run / 'properties.toml').is_file()
        assert validator.workload((run / 'solver.log').read_text(errors='replace')) == validator.workload((refdir / 'solver.log').read_text(errors='replace'))
        assert tomllib.loads((run / 'properties.toml').read_text()) == tomllib.loads((refdir / 'properties.toml').read_text())
        config = tomllib.loads((run / 'config_smartsim.toml').read_text())
        baseline_config = tomllib.loads((refdir / 'config_smartsim.toml').read_text())
        for key in ('batch_size', 'min_batch_size', 'min_batch_timeout'):
            assert baseline_config['library'].pop(key) == 0
            config['library'].pop(key)
        assert config == baseline_config
        run_info = validator.info(run / 'run.info')
        base_info = validator.info(refdir / 'run.info')
        for key in ('MODEL', 'RUN_STEPS', 'SOLUTION_INTERVAL', 'ML_CUBE_OVERLAP',
                    'MAIA_SMARTSIM_NUM_GPUS', 'MAIA_ML_STEP_TIMING_SYNC', 'DEBUG_EXPORT'):
            assert key in run_info and run_info[key] == base_info.get(key), (row['cell'], key)
        actual = timings(run / 'solver.log')
        base = timings(refdir / 'solver.log')
        recorded = [r for r in inference if r['cell'] == ref['cell']]
        assert len(recorded) == 7 and all(r['jobid'] == ref['jobid'] and float(r['latency_ms']) == base[int(r['step'])] for r in recorded)
        for condition, cell, job, directory, times in [('baseline', ref['cell'], ref['jobid'], refdir, base), ('aggregation', row['cell'], row['jobid'], run, actual)]:
            raw.append(dict(model=m, placement=p, rep=rep, condition=condition, cell=cell, jobid=job, run_dir=str(directory.relative_to(ROOT)), **{f'step_{s}_ms': times[s] for s in MEASURED}, mean_ms=statistics.mean(times[s] for s in MEASURED)))
        checks.append(dict(cell=row['cell'], jobid=row['jobid'], baseline_cell=ref['cell'], baseline_jobid=ref['jobid'], status='ok', workload_and_properties='exact', batch_size=row['batch_size'], min_batch_size=row['min_batch_size'], timeout_ms=row['timeout_ms']))

    summary = []
    for m in ('mlp', 'transformer'):
        for p in PLACEMENTS:
            b = [r['mean_ms'] for r in raw if r['model'] == m and r['placement'] == p and r['condition'] == 'baseline']
            a = [r['mean_ms'] for r in raw if r['model'] == m and r['placement'] == p and r['condition'] == 'aggregation']
            assert len(b) == len(a) == 3
            bm, bl, bh = ci(b)
            am, al, ah = ci(a)
            vb, va = statistics.variance(b)/3, statistics.variance(a)/3
            se = math.sqrt(vb+va)
            df = (vb+va)**2 / (vb**2/2 + va**2/2)
            diff = am-bm
            half = t.ppf(.975, df)*se
            result = ttest_ind(a, b, equal_var=False)
            summary.append(dict(model=m, placement=p, baseline_mean_ms=bm, baseline_ci_low_ms=bl, baseline_ci_high_ms=bh, aggregation_mean_ms=am, aggregation_ci_low_ms=al, aggregation_ci_high_ms=ah, difference_ms=diff, difference_ci_low_ms=diff-half, difference_ci_high_ms=diff+half, change_percent=100*diff/bm, speedup=bm/am, welch_df=df, welch_p_two_sided=result.pvalue))
    write('runs.csv', raw)
    write('validation.csv', checks)
    write('summary.csv', summary)
    print(f'Validated {len(checks)} aggregation runs and {len(checks)} canonical runs; wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
