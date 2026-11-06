# PhyDLL Transformer cap comparison

Reproduce locally from the repository root with Python 3.11 or newer:

```sh
python3.11 analysis/phydll_cap_independent_20261104/analyze.py
```

No jobs are submitted. Inputs are the 24 Transformer rows in
`results/phydll_cap50000_reruns/campaign_20261028T180740Z_n5AUNF/manifest.csv`,
the authoritative selections in `results/cmi_bench_consolidated/runs.csv`,
their original run directories, and all six
`logs/phydll_cap_check_5300338_{0..5}.log` files. These may be on local scratch
and are required for reproduction. The script imports the repository validator
and overwrites only this directory's CSVs. Assertions require normal Python (not
`python -O`). `validation.csv` is the validator output, `observations.csv` is
one row per run, `summary.csv` contains full-precision aggregates and wall-time
statistics, and `correctness_records.csv` contains the 96 logged comparisons.

The script checks unique canonical and rerun cells, the full 4 x 2 x 3 matrix,
distinct job IDs for repetitions within each cell, cell/rep/cap metadata,
successful solver and final exits, matching five canonical CSV timings against
original logs, seven ordered timing records, positive finite values, and enabled
synchronized timing. The imported validator checks rerun configurations,
workload against canonical fields and, where present, raw canonical workload and
parsed properties. Its four-call measured diagnostic uses steps 102, 131, 160,
189; it is not used as the comparison metric. Existence-based optional checks
in that validator are not proof of byte-identical executables or inputs.

## Results

Mean of three independent per-run means, each based on steps 73, 102, 131, 160,
189; steps 15 and 44 are warmup. Intervals are mean +/- t(0.975, 2) * sample
SD / sqrt(3), with t = 4.302652729911275. Percentage is the ratio of group
means, 100 * (cap50000 / cap2048 - 1), not an average of paired changes; no CI
for that ratio is claimed. Repetition labels do not imply matched random pairs.

| Placement | Wait | 2048 mean [95% CI], ms | 50000 mean [95% CI], ms | Change |
|---|---|---:|---:|---:|
| 24c1g | default | 8171.44 [7486.35, 8856.54] | 7733.81 [7698.27, 7769.36] | -5.36% |
| 24c1g | readiness | 8010.14 [7972.80, 8047.49] | 7748.29 [7733.21, 7763.37] | -3.27% |
| 24p24c1g | default | 8349.25 [7406.55, 9291.94] | 7710.89 [7678.47, 7743.32] | -7.65% |
| 24p24c1g | readiness | 8113.05 [7990.03, 8236.08] | 7695.98 [7631.61, 7760.34] | -5.14% |
| 96c4g | default | 3420.66 [1396.37, 5444.96] | 2611.17 [2191.54, 3030.81] | -23.66% |
| 96c4g | readiness | 2391.24 [2273.16, 2509.32] | 2320.28 [2285.47, 2355.09] | -2.97% |
| 96p96c4g | default | 2888.81 [2594.20, 3183.42] | 3678.20 [379.86, 6976.54] | +27.33% |
| 96p96c4g | readiness | 2526.68 [2491.36, 2562.00] | 2442.12 [2316.78, 2567.45] | -3.35% |

`ml_ms` is the maximum solver-rank duration of the MAIA ML step, including
solver field copies, coupling/communication and inference, reconstruction and
the exit barrier. The entry barrier and the subsequent reductions are excluded;
DL ranks do not join the solver barriers. It is not GPU kernel time. See
`maia/src/FV/fvstructuredsolver.cpp:8518-8619`. Wall time is separately in
`summary.csv`: integer-second launcher-to-exit time including the PhyDLL MPMD
processes, not allocation wall time. Source runners use `date +%s` around the
`timeout ... solver_cmd` command.

The campaigns were run separately and not randomized, so results can reflect
time, node and software differences as well as cap. Three runs are insufficient
for robust distributional inference; wide t intervals and outliers matter.
There is no CI or significance test for the change. Work varies across
placements: these are not fixed-work scaling comparisons. Heterogeneous labels
use c23g+c23g surrogate nodes, with 48 or 192 total solver ranks. `default`
means blocking, and `readiness` enables the solver readiness handshake.
Recorded campaign provenance includes source diffs and hashes, but does not
establish fully identical builds or a controlled experimental environment.

## Correctness Job 5300338

Slurm accounting reports COMPLETED, exit 0:0, elapsed 00:13:22; the job stdout
`logs/phydll_cap_check_5300338.out` reports six blocks each with 8 checks, zero
failures. The individual logs cover all ranks 0-23 and the first two exported
inferences of cap50000 blocking pilot job 5251529. The checker reloads the
TorchScript model on CPU and compares independent per-rank chunked forwards
at cap 2048 versus 50000: 48/48 PASS and logged maximum absolute difference
0. It also compares CPU cap50000 forwards to the pilot's exported provider
outputs: 48/48 PASS, largest logged max absolute difference 3.800e-7, using
atol=rtol=1e-4. Relative maxima near zero can be large and do not alone
determine allclose. All six logs identify model SHA-256
`bd2498fc594f73e4bad1088efa7522495e2289b85a853d2f3c7e081e41842147`.

The checker treats comparison with the pilot output as informational: missing
or mismatched reference outputs do not by themselves fail the job; here the
96 parsed comparison records establish that each comparison was present and
passed. CPU replay is rank-by-rank, whereas actual GPU aggregation/chunking
can differ. This does not establish cap equivalence on every GPU, placement,
wait mode, step, or whole solver trajectory. The logs warn that NumPy input
buffers are non-writable when passed to PyTorch; the checker assumes the model
does not write to them. Source and job artifacts: `scripts/check_phydll_cap_correctness.py`
and `slurm/check_phydll_cap_correctness.sbatch`.
