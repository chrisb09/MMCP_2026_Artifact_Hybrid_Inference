#!/usr/bin/env python3
"""Reproduce cap comparison after running the repository validator (see README)."""
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


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def ci(values):
    assert len(values) == 3
    mean = statistics.mean(values)
    half = 4.302652729911275 * statistics.stdev(values) / math.sqrt(3)
    return mean, mean - half, mean + half


def main():
    campaign = ROOT / 'results/phydll_cap50000_reruns/campaign_20261028T180740Z_n5AUNF'
    manifest = rows(campaign / 'manifest.csv')
    canonical = {r['cell']: r for r in rows(ROOT / 'results/cmi_bench_consolidated/runs.csv')}
    checks = []
    observations = []
    for entry in manifest:
        check = validator.validate(entry, canonical)
        checks.append(check)
        old = canonical[entry['cell'].replace('-cap50000', '')]
        assert old['provider'] == 'phydll' and old['model'] == 'transformer'
        assert old['rep'] == entry['rep'] and old['placement'] == entry['placement'] and old['wait'] == entry['wait']
        run = validator.resolve_run(entry)
        timings = dict((int(a), float(b)) for a, b in re.findall(
            r'ML_STEP_TIMING\s+step=(\d+)\s+ml_ms=(\S+)', (run / 'solver.log').read_text(errors='replace')))
        old_timings = [float(x) for x in old['measured_ms'].split(';')]
        assert len(old_timings) == 5 and old['n_measured'] == '5'
        for cap, jobid, run_dir, measured, wall in (
            (2048, old['jobid'], old['run_dir'], old_timings, float(old['wall_seconds'])),
            (50000, entry['jobid'], str(run.relative_to(ROOT)), [timings[s] for s in (73, 102, 131, 160, 189)], float(check['wall_seconds'])),
        ):
            observations.append(dict(placement=entry['placement'], wait=entry['wait'], rep=entry['rep'], cap=cap,
                                     jobid=jobid, run_dir=run_dir, measured_ms=';'.join(map(str, measured)),
                                     mean_ms=statistics.mean(measured), wall_seconds=wall))
    assert len(observations) == 48 and len({r['cell'] for r in manifest}) == 24
    summary = []
    for placement in sorted({r['placement'] for r in observations}):
        for wait in ('default', 'readiness'):
            groups = {}
            for cap in (2048, 50000):
                group = [r for r in observations if r['placement'] == placement and r['wait'] == wait and r['cap'] == cap]
                assert {r['rep'] for r in group} == {'1', '2', '3'} and len({r['jobid'] for r in group}) == 3
                groups[cap] = group
                for metric in ('mean_ms', 'wall_seconds'):
                    mean, lo, hi = ci([r[metric] for r in group])
                    summary.append(dict(placement=placement, wait=wait, cap=cap, metric=metric, n=3, mean=mean, ci95_low=lo, ci95_high=hi))
            a = statistics.mean(r['mean_ms'] for r in groups[2048])
            b = statistics.mean(r['mean_ms'] for r in groups[50000])
            print(f'{placement:10} {wait:9} cap2048={a:.3f} cap50000={b:.3f} change={100*(b/a-1):+.2f}%')
    for name, data in [('validation.csv', checks), ('observations.csv', observations), ('summary.csv', summary)]:
        with (HERE / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


if __name__ == '__main__':
    main()
