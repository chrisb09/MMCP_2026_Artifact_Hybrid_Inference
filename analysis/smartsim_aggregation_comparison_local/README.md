# SmartSim aggregation versus shared baseline

Run from anywhere with Python 3.11 or later:

```sh
python3.11 analysis/smartsim_aggregation_comparison_local/analyze.py
```

The script reads the two explicitly named production manifests, the canonical
`results/cmi_bench_consolidated/{runs.csv,inferences.csv,report.json}`, and the
selected raw scratch runs. It writes only `summary.csv`, `runs.csv`, and
`inputs.sha256.json` in this directory. The latter records SHA-256 hashes of all
input files read, including the raw properties and SmartSim configs. The input
locations must remain available; scratch is not an immutable archive. The script
also imports the repository's rerun validator. No jobs are submitted.

## Measurement and statistics

The metric is `ML_STEP_TIMING ml_ms`, in milliseconds, from MAIA's
`maia/src/FV/fvstructuredsolver.cpp` `solutionStep()`: the maximum elapsed time
across solver ranks for the full ML coupling region. The start follows the entry
solver-communicator barrier; the end follows the exit barrier. It includes
packing/copying, coupling/inference, reconstruction and synchronization, not
just GPU execution. All selected `run.info` files record timing sync and print
enabled. Source inspection describes the present code, not an independently
verified hash of the binaries used in each historical run.

Each run logs calls at steps 15, 44, 73, 102, 131, 160, 189. In agreement with
the canonical consolidation, calls 1 and 2 are warmup and calls 3 through 7
(steps 73, 102, 131, 160, 189) are averaged to yield one observation per run.
The rerun validator's `MEASURED` constant instead uses only steps 102 through
189. Its four-call output is not used here. Three independent job-level means
per configuration are summarized as mean +/- `t(0.975,2) * sample_sd / sqrt(3)`;
the confidence intervals are for each mean separately, not for their difference
or ratio. Percent change is `100 * (aggregation_mean / shared_mean - 1)`;
negative means lower elapsed time. Matching repetition labels do not imply paired
experiments. No significance claim follows from these individual intervals.

## Integrity and limitations

The script enforces the complete 2-model x 4-placement x 3-repetition manifest,
distinct job IDs, configured aggregation parameters, completion and workload
checks via `scripts/validate_smartsim_aggregation_reruns.py`. It separately
checks canonical identity and selected `run.info`/`solver.log` hashes against
the consolidation report, every canonical raw call against `inferences.csv`,
the canonical run mean, successful exits, timing flags, and the entire parsed
SmartSim configuration except the three batching fields. Baseline batching
fields must all be zero. Reruns use Transformer batch/minimum 7000 and timeout
5 ms, and MLP batch/minimum 45000 and timeout 1 ms. Workloads and parsed solver
properties match the corresponding baseline runs. These checks do not prove
identical model binaries or runtime environments, numerical correctness of
output fields, or actual RedisAI batch-size distributions. The validator notes
that numerical correctness was established separately in calibration, not by
these production timing runs.

Only three repetitions make the t intervals wide and dependent on a normality
assumption. The historical baseline and reruns are separate campaigns and may
be affected by node, software, and time-varying conditions. CI overlap alone is
not a test of a difference; the displayed percentage has no uncertainty
interval. Values in CSV and the printed table are rounded; calculations use
the unrounded logged measurements.
