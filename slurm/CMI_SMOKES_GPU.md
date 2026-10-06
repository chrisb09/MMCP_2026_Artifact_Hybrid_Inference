# CMI Single-GPU Devel Smokes (c23g)

Functional GPU validation for the artifact CMI on one non-exclusive `c23g`
node: 24 CPU cores, one Hopper GPU, account `rwth0792`. The solver stays on CPU;
AIX, PhyDLL, and SmartSim move inference to the GPU.

## Files

| File | Purpose |
| --- | --- |
| `run_cmi_smoke_gpu.sh` | Single case; usable directly via `sbatch` |
| `run_cmi_smoke_gpu_suite.sh` | Several cases serially in one allocation |
| `pin_cmi_rank.sh` | Round-robin core pinning from the cgroup cpuset |

## Resource Model

- `#SBATCH --account=rwth0792 --partition=c23g --gres=gpu:1`.
- One GPU caps memory at 122 GB, so the scripts request **120 GB** (5 GB/core).
- The job allocates **24 cores** but PhyDLL launches **24 solver + 1 DL rank**.
  The extra rank is intentionally oversubscribed via `srun --overcommit`, with
  explicit affinity from `pin_cmi_rank.sh` rather than Slurm's 1:1 binding.
- The DL rank shares a core with solver rank 0. The readiness-enabled PhyDLL
  case polls with a 100 us sleep so the colocated DL rank can make progress.
- `pin_cmi_rank.sh` reads `taskset -pc $$` (the allocation cpuset), so shared
  nodes with a partial core mask are safe; no hardcoded core numbers.
- Slurm's allocation-aware `CUDA_VISIBLE_DEVICES` is preserved; the script never
  clears it. `nvidia-smi -L` is captured to `gpu_list.log`.
- PhyDLL uses `srun --wait=0` to avoid the site's five-second first-rank-exit
  deadline killing the GPU DL client during teardown. The outer case timeout
  remains active, and nonzero rank exits still terminate the step.
- AIX runs capture execution diagnostics under `aix_gpu_diagnostics` in each
  run directory. Collective CSVs must show controller CUDA forward timings and
  device batches; P2P timelines must show CUDA-event-derived `torch_forward_*`
  events. A clean exit without that evidence is not verified GPU execution.
  `scripts/check_aix_gpu_evidence.py` now rejects a successful solver exit unless
  diagnostics identify at least two CUDA inference calls.

## Cases

| Case | Provider | Communication | GPU use |
| --- | --- | --- | --- |
| `aix-collective` | AIX | collective | AIX service on GPU |
| `phydll-blocking` | PhyDLL C++ | blocking receive | DL client on GPU |
| `phydll-readiness` | PhyDLL C++ | readiness handshake | DL client on GPU |
| `aix-p2p1` | AIX | pipelined, 1 credit | AIX service on GPU |
| `aix-p2pfull` | AIX | pipelined, full credits | AIX service on GPU |
| `smartsim` | SmartSim | RedisAI | DB/model on GPU (`--use-gpu`) |

`RUN_STEPS` defaults to 60, covering the two inferences at steps 15 and 51.
`DEBUG_EXPORT=1` writes per-rank two-inference exports for parity checks.

## Usage

```bash
sbatch slurm/run_cmi_smoke_gpu_suite.sh
# Subset, no rebuild:
sbatch --export=ALL,CMI_SMOKE_CASES='aix-collective phydll-readiness' \
  slurm/run_cmi_smoke_gpu_suite.sh
# Single case:
sbatch slurm/run_cmi_smoke_gpu.sh aix-p2p1
```

Overrides: `NP_SOLVER`, `NP_DL`, `RUN_STEPS`, `CASE_TIMEOUT_SECONDS`,
`READY_TIMEOUT_SECONDS`, `MAIA_BUILD_DIR`, `SMARTSIM_PYTHON_ENV`,
`MAIA_ML_STEP_TIMING_SYNC`.

## Verified PhyDLL Runs

Both cases completed two GPU inference frames with 24 solver ranks and one DL
rank oversubscribed onto 24 cores. GPU device selection is explicit in each
`solver.log`, and both Slurm steps exited with status 0.

| Case | Job | Solver wall time | Run directory under `scratch/cmi_smokes` |
| --- | --- | --- | --- |
| Readiness | `4783519` | 100 s | `phydll-readiness_gpu_4783519_fHvAXu` |
| Blocking (readiness omitted) | `4787119` | 171 s | `phydll-blocking_gpu_4787119_lxoGqi` |

The comparison passed for all **720 binary exports**, covering 24 ranks and two
inferences, including raw provider outputs and reconstructed fields:

```bash
python3 scripts/compare_smoke_exports.py \
  scratch/cmi_smokes/phydll-blocking_gpu_4787119_lxoGqi \
  scratch/cmi_smokes/phydll-readiness_gpu_4783519_fHvAXu
```

These separate shared-node functional runs are not a controlled performance
comparison. CPU/GPU numerical agreement remains unchecked. AIX collective job
`4778976` exited successfully, but its logs do not explicitly prove CUDA use.

## Remaining-Case Results

Job `4789284` completed with exit status 0 for every case:

| Case | Solver wall time | Run directory under `scratch/cmi_smokes` |
| --- | --- | --- |
| AIX P2P, 1 initial credit | 214 s | `aix-p2p1_gpu_4789284_87c1gf` |
| AIX P2P, 24 initial credits | 163 s | `aix-p2pfull_gpu_4789284_LDKqGa` |
| SmartSim, GPU configuration | 92 s | `smartsim_gpu_4789284_QJtCRd` |

The AIX cases have 720 bitwise-identical binary exports across 24 ranks and two
inferences. However, neither produced `aix_gpu_diagnostics`, and neither logged
GPU distribution/controller selection. These results do **not** validate CUDA
execution or the requested P2P credit modes.

Source comparison with the working `../smartsim/mini_app` confirmed a
communicator-boundary mismatch: MAIA passed
`static_cast<void*>(&ml_comm)` as `library.app_comm`, while the production
generated registry casts that pointer directly to `MPI_Comm` without
dereferencing it. AIX's distribution constructor treats a non-world handle as
zero available GPUs, selecting CPU inference and bypassing P2P. The mini-app
passes `static_cast<void*>(MPI_COMM_WORLD)`, matching the registry. MAIA now
passes `static_cast<void*>(ml_comm)` instead. A focused OpenMPI check reproduced
the old non-world handle and verified the corrected round trip.

Incremental MAIA rebuild job `4793380` and its dependent AIX collective/P2P
retry `4793381` were submitted. Runtime CUDA verification is still pending;
the retry requires GPU execution evidence rather than only a clean exit.

SmartSim completed inference and normal database shutdown. Its Redis log shows
the CUDA environment's TORCH backend loaded; the provider's GPU configuration
uses SmartRedis multi-GPU model APIs. No runtime device telemetry was captured.

## Caveats

- A single GPU validates the communication modes but **cannot** exercise
  multi-GPU affinity/NUMA distribution: with one controller, affine and NUMA
  grouping both collapse onto it. The full 3x3 layout matrix needs a multi-GPU
  allocation and per-layout AIX installs (`INSTALL`/`INSTALL-AFFINE`/`INSTALL-NUMA`),
  which the artifact tree does not currently provide.
- AIX background P2P overlap requires `MPI_THREAD_MULTIPLE`; MAIA currently
  requests `MPI_THREAD_FUNNELED`, so the controller may run synchronously. Check
  the log rather than assuming overlap.
- Clearing `CUDA_VISIBLE_DEVICES` would silently force CPU inference; verify GPU
  use in the solver and DL logs, not just a successful exit.
- The shared node may queue behind other `rwth0792` work.

## Upstream Integration (2026-10-06)

The artifact's submodules were fast-forwarded to the pushed upstream state:

| Submodule | From | To |
| --- | --- | --- |
| `CPP-ML-Interface` | `2c29c0969` | `22631669e` (docs/scripts + AIX pointer) |
| `CPP-ML-Interface/extern/AIxeleratorService` | `d952fe2` | `86968b6` (NUMA-domain workgroups, timeline staleness fix, inference-stage sync) |

The CMI delta is docs/scripts/`example.config.toml` plus the AIX pointer; it
changes no `include/` headers, so the generated registry is unaffected. The AIX
delta does not touch the `app_comm_ == MPI_COMM_WORLD` guard, so the MAIA
communicator fix is still required.

Both AIX prebuilts were rebuilt at `86968b6` with `build_aix.sh both`, reusing
the existing Score-P/PAPI stack (`SMARTSIM_SCOREP_ROOT`/`SMARTSIM_PAPI_ROOT`).

Linking note: the `scorep` prebuilt cannot be linked into the uninstrumented
MAIA build in this environment. Its `libscorep_measurement` references
`scorep_subsystems`/`scorep_number_of_subsystems`, which are undefined in both
the local and cvmfs Score-P stacks. MAIA (`WITH_SCOREP=OFF`) therefore links the
plain `INSTALL` prebuilt; the artifact's canonical `slurm_build_maia.sh` builds
AIX from source with `AIX_USE_PREBUILT=OFF`, avoiding the issue entirely.

The full six-case GPU smoke re-run against the integrated submodules is job
`4797929` (exit 0, ~11 min):

| Case | Runtime | Result |
| --- | ---: | --- |
| `aix-collective` | 109 s | pass, CUDA evidence |
| `phydll-blocking` | 130 s | pass |
| `phydll-readiness` | 93 s | pass |
| `aix-p2p1` | 72 s | pass, CUDA evidence |
| `aix-p2pfull` | 59 s | pass, CUDA evidence |
| `smartsim` | 133 s | pass |

Same-device parity is bitwise (720/720) for AIX P2P1 vs P2Pfull, PhyDLL blocking
vs readiness, and AIX vs SmartSim. AIX collective on `86968b6` is also bitwise
identical to the previous `d952fe2` run, so the upstream AIX change is
numerically neutral on the single-GPU smoke.
