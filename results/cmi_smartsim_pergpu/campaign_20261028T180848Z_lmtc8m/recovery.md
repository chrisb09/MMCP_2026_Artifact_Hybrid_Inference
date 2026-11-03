# Second heterogeneous pilot retry

Job `5297612` repeats the `smartsim-96p96c4g-pergpu-transformer-r1` pilot with
`CASE_TIMEOUT_SECONDS=600` after fixing the hetjob wrapper: `--exclusive` was
present only before `#SBATCH hetjob`, so component 1 of job `5112134` reported
`Exclusive=NO`. The wrapper now requests `--exclusive` for every component.

Provenance chain: `5109331` (300 s timeout) -> `5112134` (completed workload,
non-exclusive component 1) -> `5297612`. The machine-readable chain is
`results/cmi_smartsim_pergpu/pilot_retry_cells.csv`.

Continuation gate: validate successful single-node pilot `5109330` and retry
`5297612`, then submit the remaining ten cases.

```sh
python3.11 scripts/continue_cmi_smartsim_pergpu.py \
  --manifest results/cmi_smartsim_pergpu/campaign_20261017T211347Z_kal3PM/manifest.csv \
  --pilot-retry-manifest results/cmi_smartsim_pergpu/campaign_20261028T180848Z_lmtc8m/manifest.csv \
  --poll-seconds 60 --timeout-seconds 86400
```
