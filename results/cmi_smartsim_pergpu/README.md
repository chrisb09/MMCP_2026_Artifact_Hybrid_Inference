# Per-GPU Continuation

Recovery: original heterogeneous pilot 5109331 exhausted its 300-second command
budget near teardown after completing all seven inference calls. Retry 5112134
uses 600 seconds and is recorded in `pilot_retry_cells.csv`. See
`campaign_20261018T044023Z_qZ4xw3/recovery.md` for evidence and the explicit
`--pilot-retry-manifest` invocation. Original single-node pilot 5109330 passed
validation and is retained; the original failed manifest and blocked state are
not rewritten. Remaining cases require both effective pilots to pass.

The two Transformer r1 pilots are recorded in
`campaign_20261017T211347Z_kal3PM/manifest.csv` (jobs `5109330`, `5109331`).
The continuation script does nothing until explicitly invoked. It never cancels
or modifies jobs and queries only the manifest's two exact IDs via `sacct`.

Run from the repository root using Python 3.11:

```bash
python3.11 scripts/continue_cmi_smartsim_pergpu.py --manifest results/cmi_smartsim_pergpu/campaign_20261017T211347Z_kal3PM/manifest.csv
```

Default polling is every 60 seconds, bounded by a persisted 24-hour deadline.
Use `--poll-seconds 60 --timeout-seconds 86400` to set these explicitly; the
timeout cannot exceed 24 hours. An external launcher may background the same
command if desired. No watchdog has been launched by implementing these files.

For a single check:

```bash
python3.11 scripts/continue_cmi_smartsim_pergpu.py --manifest results/cmi_smartsim_pergpu/campaign_20261017T211347Z_kal3PM/manifest.csv --once
```

`--once` exits zero when pending, but **still submits if both gates pass**.
It is not a dry-run option. Missing parent accounting records remain pending;
job steps and heterogeneous component rows (`.` and `+`) cannot satisfy gates.
Both exact parent rows must be `COMPLETED` with `ExitCode=0:0`, followed by a
valid current-source validator report with exactly two accepted pilot cells.
Failed pilots, validation errors, accounting errors, or timeout exit nonzero
without automatic retries or cancelling any jobs.

After gating, the dedicated submitter is invoked twice with these overrides:

```bash
MODELS=transformer FIRST_REP=2 REPS=3 PLACEMENTS='96c4g 96p96c4g' DRY_RUN=0 bash slurm/submit_cmi_bench_smartsim_pergpu.sh
MODELS=mlp FIRST_REP=1 REPS=3 PLACEMENTS='96c4g 96p96c4g' DRY_RUN=0 bash slurm/submit_cmi_bench_smartsim_pergpu.sh
```

The script also pins `CMI_PERGPU_CAMPAIGN_ROOT` to the pilot campaign's parent.
These are four Transformer runs plus six MLP runs: twelve total including the
two pilots. Success means the ten additional runs were submitted, not completed
or data-validated. Validate the resulting manifests separately after completion.

`continuation_state.json` in the pilot campaign records PID, timestamps,
deadline, pilot states, validation report, commands/environment overrides,
submission stdout/stderr and returned manifest paths. `continuation.lock` is
created with `O_EXCL`; concurrent invocations are refused. State is atomically
replaced and fsynced before each submit subprocess starts. Any persisted
launch-started flag refuses a later invocation, even after success or a crash.
Submission failures may have launched some jobs: **never blindly retry**.
A stale lock, blocked state, or started batch requires manual review of state,
stdout, and submitted manifests before any deliberate recovery. Pending-only
`--once` checks may be invoked again without extending the original deadline.

Offline tests (mocked subprocesses, no Slurm submissions):

```bash
python3.11 -m unittest discover -s tests -p test_continue_cmi_smartsim_pergpu.py -v
```
