# MAIA GPU Coverage Audit

Audit of the selected 288-run snapshot and targeted searches of neighboring
SmartSim campaigns. Missing measurements and fields not exported by consolidation
are distinguished below. The canonical selection has not been changed.

## Selected Campaign

| Quantity | Available | Missing |
| --- | ---: | ---: |
| Successful selected runs | 288 | 0 |
| ML-call timings, including warmup | 2016 | 0 |
| Measured ML-call timings | 1440 | 0 |
| Expected whole-node RAM peaks | 432 | 0 |
| Expected inference-device VRAM peaks | 719 | 1 |
| Selected AIx runs with rank diagnostic CSVs | 216 | 0 |

The missing VRAM observation is
`aix-24c1g-default-coll-transformer-r2`, job `5009522`, host `n23g0007`,
device 0. Its selected raw `gpu_n23g0007.txt` file is empty and `run.info`
has no matching device peak key. A process-memory snapshot exists, but is neither
a device-wide peak nor a valid replacement. Plots retain n=2, not zero/imputation.

There are also 59 absent peaks among 360 non-expected solver-side GPU observations
on the heterogeneous surrogate nodes. Those GPUs are not configured inference
devices. This does not reduce the expected 719/720 inference-device coverage.

## Campaign Dimensions Not Measured

| Requested configuration | Located matching MAIA GPU data | Additional runs for three repetitions |
| --- | --- | ---: |
| SmartSim DB-per-GPU, 96c4g and 96p96c4g, both models | No | 12 |
| SmartSim c1 and c3, all four placements, both models | No | 48 |

The grouped matrix creates only shared-DB SmartSim cells. Both production runners
force `SMARTSIM_MPI_SEQUENTIAL_PUT=0`. The provider-specific submission script
explicitly calls per-GPU DB a follow-up. These are missing experiment dimensions,
not missing bars caused by the plotting scripts. Chaining coverage is documented
in `figures/ch7_smartsim_chaining_draft/coverage.csv`.

Nearby per-GPU/c1/c3 campaigns under `../smartsim/mini_app/results/retest_smartsim/`
and `../smartsim/mini_app/results/gpu_refresh/` use terrain WaterCNN/GiantMLP.
Nearby matching-model MAIA campaigns under `../smartsim/cpu_benchmark/` use CPU
execution. Neither is substituted for MAIA GPU observations.

## SmartSim Configuration Caveat

All 24 selected SmartSim controller logs report one shared DB, intra-op 4,
inter-op 1. Transformer runs on 24c1g and 96c4g use TPQ4; Transformer
heterogeneous runs and all MLP runs use TPQ1. These settings are not explicit
dimensions in the canonical summary. They are now documented in plotting notes
and readiness/chaining provenance. Future c1/c3 comparisons must match the full
configuration, or rerun the affected c0 baselines; changing only chaining would
otherwise conflate TPQ and chaining effects.

## Present Raw, Not Consolidated

- All 2016 timing records include rank-minimum and rank-maximum fields.
- Solver-reported RAM, GPU-process peak counts, binding evidence, and GPU identity
  logs are present in selected raw artifacts but are not normalized in the four
  canonical CSV tables.
- AIx collective stages and P2P events are present in rank CSVs and are consumed
  directly by the phase/prediction/timeline renderers with separate source hashes.
- SmartSim deployment/thread settings survive in all selected controller logs.
- The canonical `aix_gpu_batch_records` field is blank for every run because no
  selected AIx solver log contains the expected `AIX_GPU_MEM` marker. Collective
  `device_batches` and P2P sample-range diagnostics remain available separately.

Blank provider-specific configuration fields, node-RAM device IDs, node-RAM
baselines, and original-run retry reasons are not lost observations: they are
not applicable or were not collected by that schema. Whole-node used RAM is not
process/cgroup RSS; sampled device-wide VRAM is not a guaranteed instantaneous
allocation maximum.

## Evidence

- `results/cmi_bench_consolidated/{report.json,runs.csv,inferences.csv,resources.csv,summary.csv}`
- `scripts/consolidate_cmi_bench.py`
- `slurm/{cmi_bench_group_cells.sh,submit_cmi_bench_smartsim.sh,run_cmi_smoke_gpu.sh,cmi_bench_het_engine.sh}`
- Selected raw `run.info`, `solver.log`, `controller.log`, GPU sampler files,
  and AIx rank CSVs identified by the canonical provenance report.

The read-only audit reproduced canonical rows and checked 42 ledger/status-source
hashes and 576 selected run.info/solver.log hashes. Expected counts match the
declared matrix. This is not a claim that every raw event has a globally
synchronized timestamp or that all possible experiment variants were run.

## Supplemental Telemetry Repeat

Job `5106286` completed successfully and recovered a device-wide peak of 16969 MiB
(16.5712890625 GiB) for the missing-peak configuration. It is a separately
validated repeat, not a retrospective measurement of original job `5009522`.
The canonical coverage above remains 719/720; Chapter 6 capacity plots now include
the explicitly tagged repeat and label the affected bar `2 original + 1 repeat`.
Its height is unchanged because the recovered value equals the prior observed
maximum. Timings and all other resource observations are unchanged. See
`results/cmi_bench_telemetry_retry_validation.md` and its JSON companion for
configuration checks, source hashes, and the execution-time binary-hash limitation.
