# Telemetry Retry Validation

Status: validated supplemental device-wide capacity observation, not canonical replacement.

## Identity

| Field | Evidence |
| --- | --- |
| Cell | `aix-24c1g-default-coll-transformer-r2` |
| Original job | `5009522` |
| Supplemental job | `5106286` |
| Raw repeat | `scratch/cmi_smokes/aix-collective_gpu_5106286_1ln5Uv` |
| Raw original | `scratch/cmi_smokes/aix-collective_gpu_5009522_NZ2UZZ` |
| Scheduler accounting | Checked with Slurm sacct: COMPLETED, exit 0, elapsed 2:54, node n23g0013 |
| Group status | `results/telemetry_retry_group_24c1g_transformer_r2_cells.csv`: designated cell status 0 |
| Submission output | `logs/cmi-telemetry-retry-24c1g-transformer-r2_5106286.out`: `cells_failed=0` |
| Raw path receipt | `logs/group_cells/aix-24c1g-default-coll-transformer-r2.log`, appended repeat at lines 40-78 |

The explicit `results/cmi_bench_telemetry_retry_cells.csv` authorizes this one repeat only. No latest/fastest selection or canonical consolidator selection was invoked. The JSON companion pins SHA256 for both runs' `run.info` and `solver.log`, the repeat ledger and group status. Original raw hashes also match the canonical consolidation report.

## Configuration

Both designated runs pass `consolidate_cmi_bench.parse_selected_run`, and the repeat passes `extract_resources` with no resource warnings. Canonical cell identity, configuration and workload are compared directly rather than inferred from the directory name.

| Configuration | Original and repeat |
| --- | --- |
| Model | `./input/transformer_inference_scripted_fw2.pt` |
| Build executable path | `maia/build_gnu_production/bin/maia` under the workspace |
| AIx layout / communication / credits | default / collective / unset |
| Placement | 24 solver ranks, shared host, 1 expected GPU (device 0) |
| Overlap / batch size | 0 / 50000 |
| Run steps / solution interval | 200 / 10000 |
| Synchronized timing / timing print | 1 / 1 |
| Cubes per rank / aggregate | 1152 / 27648 |
| Samples per rank / aggregate | 3456 / 82944 |
| Workload solver ranks | 24 |
| Solver exit / overall exit | 0 / 0 |

Exactly seven inference steps occur: 15, 44, 73, 102, 131, 160, 189. The first two are warmup; the last five are measured. Repeat warmups are 20071.38 and 5748.75 ms; repeat steady calls are 5738.74, 5753.69, 5738.13, 5740.75 and 5748.53 ms (mean 5743.968 ms). These timings are validation evidence only and never enter any plotted latency statistic. Original rep2 mean remains 5746.674 ms.

The logged executable and model paths match. Execution-time executable/model content hashes, build revision and compiler flags are not recorded in these raw metadata; binary identity cannot be established retrospectively and is not guessed. The JSON hashes identify raw evidence, not execution-time binaries.

## Capacity Observation

Expected GPU 0 on shared host `n23g0013` has a raw baseline of **0 MiB** and raw device-wide peak of **16969 MiB = 16.5712890625 GiB**. Device-wide usage is not process allocation; the baseline is not subtracted. The four GPUs listed by Slurm do not imply four expected inference devices.

| Affected AIx Transformer 24c/1GPU bar | GiB |
| --- | --- |
| Original observed maximum (2 original repetitions) | 16.5712890625 |
| Supplemental repeat peak | 16.5712890625 |
| Refreshed maximum (2 original + 1 repeat) | 16.5712890625 |

The bar height does not change. Its annotation now states `2 original + 1 repeat`, not three original repetitions. `memory_values.csv` preserves blank original `rep2_gib` and `n_reps_observed=2`, while separately exposing `effective_rep2_gib`, supplemental job 5106286 and original missing job 5009522. Effective resource aggregation uses one row per expected device, so adding the audit repeat does not duplicate expected coverage.

`device_peaks.csv` retains all 180 original expected device rows, including the missing original row, and adds one `supplemental_repeat` row with source job, measurement job and original job provenance. There are 179 measured original device peaks plus one supplemental peak; the one historical missing original observation remains missing. Repeat RAM is not used.

## Verification

- `python -m unittest discover -s scripts -p test_plot_cmi_ch6.py`: **8 tests passed**, including mismatch rejection for identity, hashes, workload, device, scope and missing baseline.
- Latency inputs and outputs are identical with and without the supplement; `plot_values.csv` SHA256 before and after regeneration is `c549e9971af4cf01f1f37322fcb1b5639282d845a2c7f3492edeb4cbddae99b1`.
- Canonical runs, resources and report remain identical; every other memory row is unchanged with and without the supplement.
- SHA256 checks before and after regeneration confirm all six canonical snapshot files are unchanged. Runs: `7a149e18053301c8563294bc0b5d326fef7f0d4cdb35bb1384e2ba4b07b32adb`; resources: `8e3ddcd2317fc9fe156d8778aa432753c3458d793e23771f7506658721b07995`; report: `f1fdd90012d4cc3920edff951009688c53859ce511a195ebf1e177bbf5599f48`.
- Regenerated **20 PDF/PNG/PGF triplets** exclusively under `figures/ch6_gpu_draft`; export dimensions, text bounds and no-scatter contracts pass.
- No canonical data, phase/pipeline figures or their scripts were modified.

The plot generator loads only the fixed manifest if present, validates its designated job against the explicit ledger and raw evidence, then applies the supplemental peak before memory aggregation. Removing or explicitly disabling the supplement restores the original two-observation capacity summary; it does not select another run.
