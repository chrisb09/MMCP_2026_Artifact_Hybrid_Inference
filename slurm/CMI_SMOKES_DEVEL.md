# CMI CPU Devel Smokes

Run from `/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference`.
These scripts neither rebuild MAIA/CMI nor alter the existing artifact scripts,
configs, or shared outputs.

`build_cmi_master_devel.sh` incrementally rebuilds the patched PhyDLL runtime,
CMI (including registry and CTest), and MAIA on devel without replacing the
existing MAIA build directory. `run_cmi_smoke_suite_devel.sh` runs all five cases
serially in one allocation to respect the two-submitted-job devel limit:

```bash
sbatch slurm/build_cmi_master_devel.sh
sbatch --dependency=afterok:<build-job-id> slurm/run_cmi_smoke_suite_devel.sh
```

The suite prints process-exit results; verify numerical exports separately before
claiming inference correctness or parity. `solver_readiness_wait` does not change
the MAIA ghost-cell policy: all cases retain its existing default.

## Current Validation

On 2026-10-04, CMI was moved to `master` at `2c29c0969`, with PhyDLL pinned
to `007f73b`. Shell syntax and `git diff --check` passed, and the lightweight
PhyDLL metadata decoder regression script reported `ALL PASS`.

Submitted on devel with the default account:

| Job | Purpose | Status at submission |
| --- | --- | --- |
| `4764749` | First CMI build | Compiled; 8/9 CTests passed; provider test failed at MPI_Init |
| `4765011` | First smoke suite | Cancelled after build failure; did not run |
| `4768269` | Corrected incremental build and tests | Passed: eight CTests, explicit MPI provider test, MAIA rebuild |
| `4768271` | Five-case CPU smoke suite | Failed: PhyDLL field count, AIX step OOM, SmartSim timeout |
| `4770709` | Corrected five-case CPU smoke suite | AIX sync-on/off and SmartSim passed; both PhyDLL cases timed out during teardown |
| `4772481` | PhyDLL-only retry with matched shutdown barrier | Passed: blocking 193 s, readiness 165 s, both clean exits |
| `4772530` | Compare completed binary exports | Passed: AIX sync-on/off and PhyDLL blocking/readiness each have 720 bitwise-equal files |

The initial test failed because a batch-shell child called MPI_Init without a
valid Slurm MPI step. The corrected script runs the eight non-MPI CTest entries
via CTest, then launches the provider integration test explicitly with
`srun --mpi=pmix`. PhyDLL is built without its bundled nested `crun` launcher,
with shell error propagation enabled; runtime validation uses the smoke suite.
Both scripts exclude OpenMPI's legacy `s1/s2` PMIx components and do not force
the ESS component. Retry jobs `4768098`/`4768123` were cancelled before running
so the final submitted build includes this launcher correction.

Build logs are `logs/cmi-master-build_4768269.{out,err}`; suite logs are
`logs/cmi-smoke-suite_4770709.{out,err}`. Runtime completion, two-inference
coverage, numerical parity, and the GPU 3x3 matrix remain unverified.

The failed smoke suite exposed harness configuration errors. Auto transport
derives two DL output fields from this model's 5:2 input/output sequence ratio,
so both PhyDLL clients now receive `PHYDLL_DL_FIELD_COUNT=2`. Slurm accounting
showed AIX's 24-rank steps limited to 72 GiB within the 144-GiB job. The retry
uses a 180-GiB per-node allocation and explicitly gives each solver step the
full allocation. SmartSim completed all tensor uploads, then timed out waiting
for inference. Its previous successful settings (8 intra-op threads, 8 queue
workers, and 600-second Redis timeouts) are restored; the suite allows 900 seconds
for that case. The logs do not establish a SmartSim source-code defect.

In job `4770709`, both PhyDLL cases completed two frames, wrote final output,
and terminated the coupling loop. They then hung because MAIA calls a world
barrier at shutdown while the DL counterpart is opt-in. The runner now sets
`PHYDLL_MPMD_SHUTDOWN_BARRIER=1` for PhyDLL, matching this existing protocol.
The retry runs only these two cases:

```bash
sbatch --time=00:20:00 \
  --export=ALL,CMI_SMOKE_CASES='phydll-blocking phydll-readiness' \
  slurm/run_cmi_smoke_suite_devel.sh
```

The previous successful process exits took 173 s (AIX sync-on), 215 s
(SmartSim), and 188 s (AIX sync-off). Export validation confirmed complete
24-rank/two-inference coverage and bitwise equality of all 720 binaries for
AIX sync-on/off. This does not replace the outstanding clean PhyDLL teardown
check, and the GPU layout matrix has not run.

Job `4772530` also confirmed complete 24-rank/two-inference coverage and bitwise
equality of all 720 binaries between the PhyDLL blocking/readiness runs from
`4770709`. Both comparisons finished successfully in 6 min 39 s. Readiness
therefore did not change these inference results; the remaining failure is clean
process teardown. Job `4772481` subsequently passed both cases with exit status
zero, completing in 6 min 6 s overall. All five CPU smoke cases now have clean
successful runs. The parity comparisons above used the completed exports from
`4770709`, before the teardown-only environment correction; exports from the
clean-exit retry have not been compared separately. The GPU 3x3 matrix remains
untested.

## Cases

| Case | Solver ranks | DL ranks / service | Configuration |
| --- | ---: | --- | --- |
| `phydll-blocking` | 24 | 24 C++ DL ranks | `solver_readiness_wait` omitted to test the default |
| `phydll-readiness` | 24 | 24 C++ DL ranks | `library.solver_readiness_wait = true` |
| `aix-collective` | 24 | Internal AIX service | `library.communication_mode = "collective"`, MAIA timing barriers on |
| `smartsim` | 24 | Local CPU Redis/SmartSim controller | Existing CPU environment, loopback, auto-selected port |
| `aix-sync-off` | 24 | Internal AIX service | Same collective configuration, MAIA timing barriers off |

The runner requests one CPU `devel` node with 48 tasks, one CPU per task,
180 GiB per node, and 20 minutes. There is no `--account` directive or submission
override: Slurm selects the user's default account. The helper reduces AIX jobs
to 24 allocated tasks; SmartSim retains 48 to leave capacity for its local
service. It does not request GPUs.

## Usage

Preview the four standard submissions without submitting anything:

```bash
bash slurm/submit_cmi_smokes_devel.sh --dry-run
INCLUDE_AIX_SYNC_OFF=1 bash slurm/submit_cmi_smokes_devel.sh --dry-run
```

When ready, explicitly submit the cases, optionally with numerical exports:

```bash
bash slurm/submit_cmi_smokes_devel.sh --submit
DEBUG_EXPORT=1 INCLUDE_AIX_SYNC_OFF=1 bash slurm/submit_cmi_smokes_devel.sh --submit
```

Submit one case directly, or invoke the runner with Bash inside an existing
single-node allocation with enough tasks:

```bash
sbatch slurm/run_cmi_smoke_devel.sh phydll-readiness
sbatch --ntasks=24 slurm/run_cmi_smoke_devel.sh aix-collective
sbatch --ntasks=24 slurm/run_cmi_smoke_devel.sh aix-sync-off
bash slurm/run_cmi_smoke_devel.sh phydll-blocking
```

Available environment overrides:

| Variable | Default / meaning |
| --- | --- |
| `PROJECT_DIR` | Slurm submission directory, otherwise current directory |
| `MAIA_BUILD_DIR` | `${PROJECT_DIR}/maia/build_gnu_production` (current Torch 2.4 build) |
| `RUN_STEPS` | `60`; values below `51` are rejected |
| `NP_DL` | `24`; PhyDLL DL MPI ranks, not field count; increase the allocation if submitting directly |
| `CASE_TIMEOUT_SECONDS` | `900`; solver timeout, then TERM and KILL after 20 seconds |
| `READY_TIMEOUT_SECONDS` | `120`; SmartSim endpoint readiness timeout |
| `SMARTSIM_PYTHON_ENV` | `/hpcwork/$(whoami)/smartsim/python/smartsim_cpu/bin/activate` |
| `SMOKE_SCRATCH_ROOT` | `${PROJECT_DIR}/scratch/cmi_smokes` |
| `DEBUG_EXPORT` | `0`; set `1` for per-rank CMI numerical dumps, capped at two inferences |
| `MAIA_ML_STEP_TIMING_SYNC` | `1`; `aix-sync-off` forces `0`; must be identical on all solver ranks |
| `INCLUDE_AIX_SYNC_OFF` | `0`; set `1` to add the fifth case in the helper |
| `SMOKE_DEPENDENCY` | Optional Slurm dependency, e.g. `afterok:<build-job-id>`; cases run serially |

## Isolation And Results

Every invocation creates a fresh directory:

```text
scratch/cmi_smokes/<case>_<job-id>_<unique-suffix>/
```

It contains a private `properties.toml`, the selected provider config, a private
restart copy, output directories, `solver.log`, and `run.info` with elapsed solver
wall time and exit status. PhyDLL also writes `phydll.conf`. SmartSim writes
`controller.log`, `smartredis.log`, and its experiment directory here. Optional
numerical dumps go to the same run's `debug_dumps/`. Slurm stdout/stderr are
`cmi_smoke_<job-id>.out` and `.err` in the submission directory.

No shared output is deleted. The grid and `input` directory are read-only in
normal use via symlinks; the restart is copied rather than linked to prevent
accidental write-through. Scratch and diagnostics remain available on failure.
Solver timeout status is propagated, including GNU timeout's status `124`.
SmartSim gets a separate lifetime guard and bounded shutdown on solver exit.

## Inspected Behavior And Caveats

- `maia/src/FV/fvstructuredsolver.cpp` selects `./config_phydll.toml`,
  `./config_aix.toml`, or `./config_smartsim.toml` by
  `CPP_ML_INTERFACE_PROVIDER_ENV`. It overrides provider model fields from
  the input `modelPath`, and PhyDLL/SmartSim device fields from
  `CPP_ML_INTERFACE_DEVICE`. The runner sets CPU explicitly, preserves
  `./input/transformer_inference_scripted_fw2.pt`, and checks it is readable.
- MAIA also overrides the behavior/application schedule from properties and
  hardcodes global restart offset 10. The runner uses interval 5, input length 5,
  step coefficient 12, forecast window 2, scaling 1, and input distance 1,
  matching the CPU artifact setup. The expected first two inference steps are
  15 and 51; the default stop at 60 covers both. A successful process exit alone
  is not proof of two inferences: inspect `solver.log` or use `DEBUG_EXPORT=1`.
- CMI source being synced does not establish that the existing executable and
  C++ DL client were rebuilt against it. Both sides must support the current
  readiness metadata handshake, and the generated config registry in the build
  must accept `solver_readiness_wait`. These scripts do not regenerate or build.
- `aix-sync-off` means `MAIA_ML_STEP_TIMING_SYNC=0`, not removal of required
  collective communication or GPU stream synchronization. MAIA barriers bracket
  the per-step MLCoupling timer on the solver-only communicator. The final
  barrier wait is included in that timer, as in mini_app. Compare per-rank dumps and
  `BENCHMARK_SOLVER_WALL_SECONDS`; the scripts do not assert numerical equality.
- OpenMP, MKL, OpenBLAS, NumExpr, and CMI intra/inter-op thread limits are one.
  SmartSim's controller uses the previously successful eight-worker/eight-thread
  configuration, with one inter-op thread.
  The existing module setup and CUDA CPU stubs are retained. Executables are
  launched directly after setup so `maia_runner.sh` cannot re-prepend a different
  MAIA build's libraries when `MAIA_BUILD_DIR` is overridden.
- SmartSim uses the existing local controller pattern. The spare allocated CPUs
  provide capacity, but local DB processes are not placed in a separate Slurm
  step or disjoint CPU affinity mask; this is a functional smoke, not a controlled
  performance benchmark. Auto-port selection reduces conflicts but is not an
  atomic reservation. On forced termination Slurm allocation cleanup remains
  the final safeguard for detached Redis children.
- Syntax checks and submission dry-runs were performed; MPI/provider runtime
  behavior and two-inference completion require the queued devel jobs to finish.
  No commits were created.
