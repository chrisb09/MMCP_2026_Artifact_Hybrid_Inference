# Heterogeneous Pilot Recovery

This campaign repeats only `smartsim-96p96c4g-pergpu-transformer-r1` as job
`5112134`. It explicitly replaces failed pilot job `5109331` for the continuation
gate, not for canonical shared-DB consolidation. Original manifests, statuses,
and the blocked continuation state remain unchanged.

## Observed Failure

The original Slurm allocation ended with exit 124 after the solver command's
300-second outer timeout. Four standalone databases became ready and runtime
samples show four distinct Redis processes on four distinct GPU UUIDs.

Original artifacts:
`scratch/cmi_smartsim_pergpu/smartsim-96p96c4g-pergpu-transformer-r1_5109331_fGfdWn`.

- `solver.log` records all seven inference steps: 15, 44, 73, 102, 131, 160, 189.
- `solver.log:2146` records the final inference, 5949.57 ms.
- `solver.log:2148-2169` records the after-run memory summary and completed final
  solution output; `solver.log:2170` deinitializes the application log.
- `m_log` records an application wall time of about 177.4 seconds, completed
  memory cleanup, and log closure at 02:12:22 on the recorded local clock.
- The Slurm step is cancelled at about 02:12:25. About 119 seconds of its
  300-second budget elapsed before MAIA's 02:09:24 startup banner.

This establishes an exhausted outer command budget, not an initialization
deadlock, GPU-routing failure, or OOM. The exact pre-banner delay and the final
teardown operation are not identified. A persistent finalization issue is not
ruled out; the retry must finish and validate successfully.

## Change

Only the command timeout is increased from 300 to 600 seconds. The dedicated
submitter defaults future supplement runs to this bound and records it in each
manifest. Allocation wall time stays 20 minutes. Models, workload, DB/GPU mapping,
TPQ, intra/inter-op settings, batching, seven-call schedule and profiling settings
are unchanged. Measured ML-call timing excludes the outer launch/teardown budget.

The original single-node pilot `5109330` completed and now passes the validator,
including all seven calls, canonical shared-workload equality, resource coverage,
four endpoints and simultaneous four-process/four-GPU runtime proof. It is not rerun.

## Validator Corrections

Two offline assumptions were corrected against actual runtime evidence:

- Current Slurm reports `OverSubscribe=NO Exclusive=NODE`, which explicitly
  establishes node exclusivity and must not be rejected for lacking the older
  `OverSubscribe=EXCLUSIVE` spelling.
- `CPP-ML-Interface/include/logging.hpp:93-103` suppresses nonzero MPI ranks.
  Only DB leader rank zero is logged even when all four DB leaders operate.
  Validation checks that logged leader and endpoint, labels other leader ranks as
  inferred from the block-mapping algorithm, and retains the independent
  simultaneous four-Redis-process/four-GPU UUID proof. It does not claim four
  directly observed provider-leader log lines.

The original pilot manifest has no timeout column and retains its historical
300-second contract. New manifests explicitly require their recorded 600-second
timeout; the validator rejects a mismatch.

## Continuation

The replacement gate uses original single-node job `5109330` and heterogeneous
retry `5112134`. Each is validated against its own manifest and status directory.
New continuation state and lock are kept in this retry campaign, so the original
blocked state is preserved. Only after both accepted pilots are successful may
the remaining four Transformer and six MLP runs be submitted.

```sh
python3.11 scripts/continue_cmi_smartsim_pergpu.py \
  --manifest results/cmi_smartsim_pergpu/campaign_20261017T211347Z_kal3PM/manifest.csv \
  --pilot-retry-manifest results/cmi_smartsim_pergpu/campaign_20261018T044023Z_qZ4xw3/manifest.csv \
  --poll-seconds 60 --timeout-seconds 86400
```

This command can submit the already-authorized remaining ten cases after the
validation gate. It is not a dry run. Failed pilots, invalid telemetry and
ambiguous/partial submissions remain fail-closed.
