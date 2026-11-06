# Local batching comparisons

Reproduce from the repository root (Python 3.11+ and Matplotlib 3.10.7, as
pinned in the repository requirements, required):

```sh
python3 scripts/plot_local_batching_comparison.py
python3 scripts/plot_local_batching_comparison.py --pgf  # optional, requires XeLaTeX
```

The script reads existing data and logs only; it does not run validators, submit
jobs, or alter the thesis, other figures, or analysis inputs. It writes PNG and
PDF at 150 mm width, optionally PGF, and `plotted_statistics.csv` in this
directory. The PGF figure uses DejaVu Sans and requires that font and XeLaTeX
when compiled. `phydll_cap` compares caps 2,048 and 50,000 for Transformer,
separately for default and readiness waits. `smartsim_aggregation` compares
shared nonaggregated and aggregation for Transformer and MLP. Placement labels
are the campaign's original identifiers.

## Provenance and calculation

Inputs are `analysis/phydll_cap_comparison_local/{observations,validation,summary}.csv`
and `analysis/smartsim_aggregation_comparison_local/{runs,summary}.csv`, and
each referenced raw `scratch/.../solver.log`. The companion `analyze.py` files
describe their original selection and validation; SmartSim's README and
`inputs.sha256.json` document additional integrity checks and input hashes.
PhyDLL validation covers the cap-50000 reruns; its comparison analysis matches
these to canonical cap-2048 runs. The plot script checks all 96 logs against
the analysis values, complete step sequence, positive finite timings, full
configuration coverage, three distinct job IDs and repetition labels per
configuration, and agreement with the existing summary statistics. It checks
the 24 PhyDLL validation records are OK, but does not independently rerun all
campaign validation or check the SmartSim input hashes. Scratch is not an
immutable archive and must remain available.

The logs contain seven calls at steps 15, 44, 73, 102, 131, 160, 189. The first
two are warmup; **the same five measured steps 73, 102, 131, 160, 189** are
averaged for each run. Neither validator's four-call `measured_mean_ms` is used.
The plotted metric is MAIA `ML_STEP_TIMING ml_ms`, the maximum across solver
ranks for the synchronized ML coupling region, including packing, inference,
reconstruction and synchronization, not GPU execution alone. Units are seconds
in the figures and milliseconds in the statistics CSV.

Bars are means of three independent job-level five-call means; labels above the
error bars give the means in seconds. Error bars are
individual two-sided 95% t intervals, mean +/- 4.302652729911275 times the
sample standard deviation divided by sqrt(3). No lower bound is clamped. These
are not confidence intervals for differences or ratios; matching repetition
labels do not imply paired experiments. Three repetitions yield imprecise
intervals under a normality assumption, and overlap is not a significance test.

## Caveats

Historical baselines and reruns are separate campaigns, potentially affected by
node, software and time variation. The original validators check workload and
configuration, but do not establish identical binary/runtime environments or
numerical correctness of these production runs. SmartSim aggregation uses
Transformer batch/minimum 7000 and timeout 5 ms, MLP batch/minimum 45000 and
timeout 1 ms; actual RedisAI batch-size distributions are not established.
PhyDLL cap correctness was assessed separately by the repository's correctness
script. Source-level timing interpretation is not an independently verified
binary hash. No new experiments were performed.
