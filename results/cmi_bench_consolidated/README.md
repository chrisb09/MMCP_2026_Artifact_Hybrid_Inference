# Consolidated GPU Campaign

Authoritative, deduplicated MAIA [TBL] Transformer/MLP GPU benchmark results.
Original runs and historical analysis outputs remain unchanged.

## Contents

| File | Rows | Meaning |
| --- | ---: | --- |
| `runs.csv` | 288 | One selected successful run per cell, metadata, provenance, workload, timing statistics |
| `inferences.csv` | 2,016 | All seven logged inference timings per selected run |
| `resources.csv` | 1,512 | Long-form node RAM and per-device VRAM peaks/baselines, including explicit missing observations |
| `summary.csv` | 96 | Timing aggregates across the three repetitions of each configuration |
| `report.json` | | Validation results, measurement coverage, retry lineage, and source/artifact SHA-256 hashes |

## Selection

The matrix is 4 placements x 2 models x 3 repetitions x 12 configurations:
9 AIX layout/communication combinations, 2 PhyDLL wait modes, and 1 shared-DB
SmartSim configuration. Heterogeneous placements use c23g+c23g surrogate nodes.

Selection follows the original grouped campaign ledger and explicit cell-level
SmartSim, quota, shutdown, and final retry manifests, in that order. A selected
retry replaces only its named cells. No pilot jobs, historical provider campaigns,
fastest-run selection, implicit latest-run selection, or fallback from an invalid
designated retry is allowed. Match both job ID and `run.info:CELL`; the directory
name alone does not identify single-node cells.

All 288 selected runs have successful solver and final exit statuses, matching
configuration/workload metadata, and exactly the expected seven timing records.
The selection contains 187 originals, 94 quota retries, 3 shutdown retries,
2 standalone SmartSim retries, and 2 final retries.

## Timing and Workload

Timings are milliseconds. Steps 15 and 44 are warmup; steps 73, 102, 131, 160,
and 189 are measured. `runs.csv` preserves logged timings without presentation
rounding. `summary.csv` averages the three per-repetition measured means and
reports their sample standard deviation. It does not treat 15 calls as 15
independent campaign repetitions. Solver wall time is a separate launch-to-exit
metric, not inference latency or whole-allocation time.

Workload aggregates differ across placements and are retained exactly as logged.
Do not assume equal aggregate work across placements or reconstruct aggregates
by multiplying a single rank's count by the number of ranks. Transformer uses
overlap 0; MLP uses overlap 4. Heterogeneous total solver ranks are 48 or 192,
not just the CPU-side count. Configured GPU counts come from the matrix, not
the potentially misleading `SLURM_JOB_GPUS` device list.

## Memory Coverage

Node RAM is sampled whole-node used memory, not process RSS or cgroup usage.
VRAM is sampled device-wide used memory. Values are MiB, with GiB = MiB / 1024.
Baselines and each device are preserved separately. These sampled peaks are not
guaranteed instantaneous allocation maxima.

- Expected node RAM observations: **432/432 valid**.
- Expected GPU VRAM observations: **719/720 valid**.
- Missing VRAM: `aix-24c1g-default-coll-transformer-r2`, job `5009522`.

The missing VRAM sample is explicitly marked `missing`, with blank peak values;
it is not zero and does not invalidate the successful timing run. Resource rows
with `expected=False` retain other observed devices or baselines, but do not
count toward configured-device coverage. Filter `expected=True` when computing
headline memory coverage. The validation report records the telemetry warning.

## Reproduction

From the repository root:

```bash
python3 scripts/consolidate_cmi_bench.py
python3 -m unittest discover -s scripts -p test_consolidate_cmi_bench.py
```

The consolidator requires the original text logs, run metadata, and campaign
ledgers. It does not require the deleted HDF5 outputs. It fails rather than
silently accepting missing runs, failed designated retries, duplicate identities,
incorrect timing schedules, or configuration mismatches. Missing memory telemetry
is a reported warning rather than an imputed value.
