# DEVELOPMENT.md

Working log and architectural plan for the `MMCP_2026_Artifact_Hybrid_Inference` repository. This document tracks in-flight work, decisions, and forward-looking plans for the project. Cross-cutting topics (PhyDLL integration, SmartSim multiplexing, build infrastructure) are summarized briefly and link to the dedicated insight notes under `~/insights/`.

---

## 1. Current State of the Working Tree

Branch: `debug/current-prepost` (active development branch).

### 1.1 Repository Layout

* The project integrates the MAIA CFD solver with an ML coupling interface.
* The C++ coupling library lives in a git submodule at `CPP-ML-Interface/`.
* The submodule at `CPP-ML-Interface` points to the artifact fork, branch `debug/current-prepost`, commit `5ac6160`. All upstream changes through the `uniform_chunks` PhyDLL transport layout have been integrated.
* The old coupling code (`CPP-ML-Interface-old/`) has been removed (Phase 8). Its snapshot_writer and logger utilities were moved to `maia/src/`.
* The new CPP-ML-Interface is a complete rewrite: TOML-driven configuration, templated `<In, Out>` base classes, fluent proxies for ordered/keyed APIs, application-led `ml_step(provider&, behavior&)` orchestration, and `MLCouplingBehaviorFlowExtrapolator` for MAIA timing.

### 1.2 Files of Interest

| Path | Purpose |
|------|---------|
| `maia/src/FV/fvstructuredsolver.cpp` | MAIA solver's main time loop. Contains the ML coupling calls and now the snapshot writer hooks. |
| `maia/src/FV/fvstructuredsolver.h` | Solver class declaration. Will need a `unique_ptr<MLCoupling<...>>` member. |
| `maia/src/globals/maia.cpp` | Hosts the global `m_mlCoupler` instance. To be removed. |
| `maia/src/globals/globalvariables.{h,cpp}` | Global state. The old `m_mlCoupler` global lives here. To be removed. |
| `CPP-ML-Interface/include/snapshot_writer.hpp` | New header-only HDF5 snapshot writer (rank 0, /tmp staged). |
| `CPP-ML-Interface-old/include/ml_coupling/maia/ml_coupling_maia.hpp` | Old `MLCouplingMaia` class. Has the cubing/un-cubing (`extract_cubes`) logic we must port. |
| `slurm/new_example_job_devel_24.sh` | 24-task/node SLURM job script. Adds post-job copy from `/tmp` to `MAIA_SNAPSHOT_DIR`. |
| `slurm/run_new_example_job_devel_24.sh` | Runner script. Sets `MAIA_SNAPSHOT_DIR=/hpcwork/thes2181/mmcp` before `sbatch`. |
| `slurm/slurm_install_maia.sh` | SLURM submission of the MAIA build (32 tasks, `devel` partition). |
| `input/properties_run_les_ref_medium.toml` | Solver properties (`timeSteps = 300`, domain 126×131×733). |
| `install-CPP-ML-Interface.sh` | Builds CPP-ML-Interface. Skips if `libmlCoupling.so` exists. |
| `install-MAIA.sh` | Builds MAIA. Dynamically detects processor count. |
| `transition_plan_cpp_ml.md` | Companion plan for the new interface transition. See Section 6. |

### 1.3 Job Configuration

* Job type: 24 tasks per node (replaced the misleadingly named `_48` variants — see Commit `9f1af1d`).
* Domain: 126 × 131 × 733 cells (`properties_run_les_ref_medium.toml`).
* `timeSteps = 300`, with coupling/inference on a regular interval (controlled via `mlInterval`, `mlInputLength`).
* Test job ID from first end-to-end run: **874293**. Reference snapshot at `/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5` (renamed from `snapshots_874293.h5`).

---

## 2. Commit History on `dev` (just before transition branch)

`dev` was clean before branching. Commits on `dev` that landed from this session (in order):

| Hash | Subject |
|------|---------|
| `43cacf2` | `gitignore: add rules for core dumps and scratch input TOML copies` |
| `9f1af1d` | `slurm: replace 48-task devel job scripts with 24-task variants` |
| `bd331d0` | `Add runtime HDF5 field snapshot capture for ML inference verification` (note: this is also on `dev` from a follow-up, not just on the transition branch — verify with `git log --oneline origin/dev` if needed) |

Notes:

* The submodule pointer was already correctly updated to `d0f9c3e` (`ef9a3a5`, an earlier commit) and further to `9794f57` by `bd331d0`. The `extern/` untracked content inside the submodule is expected (build artifacts populated by external install scripts).
* `.gitignore` now excludes: `core.*` (Node.js crash dumps) and `input/*copy*.toml` (scratch copies).

---

## 3. Snapshot Writer Feature (Reference Output Capture)

Goal: capture the C++ solver's U/V/W fields at every coupling/inference step and store them as HDF5, so that when we later switch to the new CPP-ML-Interface we have a golden reference to verify against.

### 3.1 Design Summary

* **Trigger:** environment variable `MAIA_SNAPSHOT_DIR`. If unset, all snapshot calls are no-ops. If set, writes are enabled.
* **Writer location:** rank 0 only (matches `logger.hpp` pattern; uses `SLURM_PROCID == "0"` to detect).
* **Staging:** writes to fast local SSD at `/tmp/maia_snapshots_${SLURM_JOB_ID}.h5` during the run.
* **Persistence:** the SLURM job script copies the file to `${MAIA_SNAPSHOT_DIR}/snapshots_${SLURM_JOB_ID}.h5` after `srun` completes.
* **Schema:** `/step_NNNN/{U,V,W,nCells,nOffsetCells,globalTimeStep,type}` with `type = "sent"` (before ML) or `"received"` (after ML). Flushes every 20 snapshots.
* **Capture timing:** both before sending to ML and after receiving from ML, at every coupling step.

### 3.2 Files Created/Modified

| File | Action |
|------|--------|
| `CPP-ML-Interface/include/snapshot_writer.hpp` | **New** (header-only, like `logger.hpp`). |
| `maia/src/FV/fvstructuredsolver.cpp` | **Modified**: include the new header, call `snapshot::init()` in the ctor, `snapshot::finalize()` in the dtor, and replace the compile-time `#ifdef OUTPUT_FIELDS` blocks with runtime-conditional `snapshot::write_step(...)` calls. |
| `slurm/new_example_job_devel_24.sh` | **Modified**: export `MAIA_SNAPSHOT_DIR`, copy from `/tmp` to persistent after run. |
| `slurm/run_new_example_job_devel_24.sh` | **Modified**: pre-set `MAIA_SNAPSHOT_DIR=/hpcwork/thes2181/mmcp` for debug runs. |

### 3.3 Build & Run Notes (Important Lessons)

* **Header install path:** the CPP-ML-Interface build copies all `*.hpp` files from `include/` into `BUILD-SCOREP/include/` via its `install()` directive. So a fresh CPP-ML-Interface build (or manually copying `snapshot_writer.hpp` into `BUILD-SCOREP/include/`) is needed for MAIA to find the header.
* **The install script (`install-CPP-ML-Interface.sh`) skips rebuilds** if `libmlCoupling.so` already exists. After adding a new header, either force a clean rebuild or just `cp` the header into `BUILD-SCOREP/include/` (since it is header-only, no library rebuild is strictly required).
* **MAIA build is a SLURM job.** Use `sbatch slurm/slurm_install_maia.sh` from a login node. The first build attempt (job `873931`) failed because the new header wasn't in the install include path; a manual copy + resubmit (job `874189`) succeeded.
* **Test job:** `sbatch ./slurm/run_new_example_job_devel_24.sh` (job `874293`). 12 MPI ranks over 1 node, `c23mm`. Snapshot file is ~370–560 MB on disk for 300 steps.
* **Memory/storage estimate:** ~24 MB per snapshot per rank. With ~65 coupling+inference steps and both `sent` + `received` (~130 total), output is roughly 3 GB — well within `/tmp` and `/hpcwork/thes2181/mmcp`.
* **Reference output:** `/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5` (was originally `snapshots_874293.h5`, renamed for clarity).
* **Open caveat:** rank 0 captures only its own partition (~67×69×126 cells for this 12-rank decomposition), not the full 126×131×733 domain. This is sufficient for verifying the *new* interface produces the same outputs (same decomposition ⇒ same rank-0 partition), but it is **not** a full-domain snapshot. If we ever need the full assembled domain we would have to switch to either parallel HDF5 (MPI-IO) or per-rank files plus a reassembly step.

### 3.4 Why We Did This

The verification picture is: switch to the new CPP-ML-Interface (which has its own multiple DL-side implementations) → run with the new interface → diff rank-0 U/V/W at coupling steps against the reference HDF5. If the diffs are within tolerance, the new interface is functionally equivalent for our purposes.

---

## 4. Module Test (`/hpcwork/ro092286/smartsim/module_test`)

Cross-cutting reference. The full historical log lives in `~/insights/module_test.md`. Key items relevant to the MAIA work:

* **SmartSim GPU indexing:** `num_gpus` is an upper bound (not a count). Use `CUDA_VISIBLE_DEVICES` + `first_gpu=0, num_gpus=1` for predictable mapping.
* **SmartSim model caching:** changing `model_path` without changing `model_name` can cause stale model execution. Always use a unique `model_name` per architecture variant.
* **Step timing & profiling:** the `TIMING_LOG` mechanism emits per-step start/end nanosecond timestamps and per-step inference durations to `<TIMING_LOG>_rank_<N>.csv`. Aggregated via `analyze_timings.py`. Step 0 (handshake + model load) is ~83% of total time on CPU inference at 72 clients; steady-state inference is ~0.88 s per step with very low std-dev.
* **Dynamic batch sizes:** `test_matrix.py` runs exhaustive batch sweeps (default `[1, 7]`) and dynamically appends `_split_flat` to the model name for SmartSim multi-input runs.

These patterns will be reused when porting MAIA's coupling to the new interface (we will likely want a similar step-timing CSV and a TOML-driven config analogous to the module_test's `config_*.toml` files).

---

## 5. New CPP-ML-Interface Architecture Summary

Source: `CPP-ML-Interface/` (artifact fork, `debug/current-prepost`, HEAD `5ac6160`).

### 5.1 What It Is

A complete rewrite of the coupling interface. Key departures from the old interface:

* **TOML-driven configuration.** The main app constructs a coupling by calling a factory with a `config.toml` path, instead of using compile-time flags. Providers (AIx, PhyDLL, SmartSim, dummy) are loaded at runtime.
* **Templated core:** `MLCoupling<In, Out>` instead of the old non-templated `MLCouplingMaia`. Both type parameters default to `double` and represent the input/output scalar types.
* **Fluent proxies:** `obj->ordered().set(a).set(b).inference()` style API. Uses short-lived proxy views that hold a reference to the parent. See `~/insights/cpp_api_design_patterns.md` for the design pattern.
* **Fluent API design gotchas:** when both `In` and `Out` are the same type, overloaded `set(In)` and `set(Out)` collide. Use semantically distinct names like `set()` and `set_target()`. See `cpp_api_design_patterns.md` for details.
* **Advanced `flex_*` methods on the base class** (not in a separate subclass). Default implementation **buffers** staged inputs and **falls back to `static_inference`**, with merge-by-concatenation. This means any provider gets multi-input support for free. Note: the older `MLCouplingProviderFlexible` subclass has been removed — custom providers should now inherit from `MLCouplingProvider` directly. See `CPP-ML-Interface/documentation/migration_guide.md`.
* **Resource-efficient fallback:** the flex fallback passes a pointer to the user's pre-allocated output buffer, so the subsequent `.get()` is zero-cost.
* **`MLCouplingApplication`:** the user inherits from `MLCouplingApplication<In, Out>` and overrides `preprocess()`, `postprocess()`, and `ml_step()`. This is where application-specific pre/post (like our cubing/un-cubing) lives.
* **Providers available:** `aixelerator`, `smartsim`, `phydll`, `dummy`. See `~/insights/phydll_integration.md` for PhyDLL-specific fixes.
* **Normalization layer:** MinMax and Standardization are first-class config items.
* **Behavior layer:** `default` and `periodic` behaviors (e.g. for time-series coupling).

### 5.2 Concrete Headers in the New Interface

```
include/
  ml_coupling.hpp                          # MLCoupling<In, Out> top-level class
  config.hpp / config_overrides.hpp        # TOML config + programmatic overrides
  coupling_type.hpp                        # In/Out type helpers
  data/
    ml_coupling_data.hpp                   # MLCouplingData tensor wrapper
    ml_coupling_data_type.hpp
    ml_coupling_memory_layout.hpp
  provider/
    ml_coupling_provider.hpp               # Base provider (MLCouplingLibrary base)
    ml_coupling_provider_aixelerator.hpp   # MLCouplingLibraryAixelerator
    ml_coupling_provider_smartsim.hpp      # MLCouplingLibrarySmartSim
    ml_coupling_provider_phydll.hpp        # MLCouplingLibraryPhydll (packed + uniform_chunks)
    ml_coupling_provider_dummy.hpp         # MLCouplingLibraryDummy
  application/
    ml_coupling_application.hpp            # Base application class
    ml_coupling_application_turbulence_closure.hpp
    ml_coupling_application_flow_extrapolator.hpp
  behavior/
    ml_coupling_behavior.hpp               # Base behavior
    ml_coupling_behavior_flow_extrapolator.hpp  # FlowExtrapolatorBehavior (HDF-safe scheduling)
  normalization/
    ml_coupling_normalization.hpp
    ml_coupling_minmax_normalization.hpp
  logging.hpp
  training_tracker.hpp
  tool.h
  c_api.h
```

### 5.3 Reference Examples

* `CPP-ML-Interface/test/` — unit tests for behavior scheduling (`test_behavior_flow_extrapolator.cpp`, 19 tests covering all scheduling paths including cumulative delta and HDF-avoidance).
* `CPP-ML-Interface/test/phydll_mpmd/` — MPMD regression harness for the PhyDLL transport layer (2 PHY ranks + 1 DL rank, all 4 permutations: 18to1/1to18 × packed/uniform_chunks).
* `/hpcwork/ro092286/smartsim/CPP-ML-Interface/` — upstream SmartSim worktree (source of `uniform_chunks` integration). Do not modify.

---

## 6. Transition Plan: MAIA → New CPP-ML-Interface — ✅ COMPLETED

**Full plan:** `transition_plan_cpp_ml.md` (v2, now removed). All 8 phases have been implemented and committed.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Submodule cutover to artifact fork `debug/current-prepost` | ✅ |
| 1 | Upstream: `ml_step(provider&, behavior&)`, `step()` returns `int`, removed `ml_step()` alias | ✅ |
| 2 | Upstream: affected consumers (mini_app/module_test — accepted breakage) | ✅ (doc'd) |
| 3 | Upstream: `MLCouplingBehaviorFlowExtrapolator` | ✅ |
| 4 | Upstream: `MLCouplingApplicationFlowExtrapolator` override of new `ml_step` | ✅ |
| 5 | MAIA: solver integration (remove globals, rewrite setup/solutionStep) | ✅ |
| 6 | Config: `config.toml`, CMakeLists.txt, ConfigOverrides from Context | ✅ |
| 7 | Verification: snapshot comparison | ✅ |
| 8 | Cleanup: remove `CPP-ML-Interface-old/`, push upstream, move utils to `maia/src/` | ✅ |

**Phase 7 notes:** The step-15 post-inference divergence (`max_abs ~0.1`) was investigated via `verify_inference.py` and `FLOW_DUMP_DEBUG` binary dumps. Root causes identified and fixed:
- Scheduling bug: `next_global` was not tracking cumulative `time_step_delta()` jumps — fixed via `effective_global_step_` accumulation in `ml_coupling_behavior_flow_extrapolator.hpp`.
- HDF5 group loading bug in `verify_inference.py` — fixed via attribute-index scan.
- Both fixes verified by the 19-test behavior unit suite (including Test 6: `test_cumulative_delta_regression`).

### 6.1 Verification Summary (Phase 7)

Completed via `FLOW_DUMP_DEBUG` binary dumps from a 20-step MAIA run and the standalone `verify_inference.py` script. The step-15 "received" divergence from the old-code reference was traced to the HDF-safe scheduling computation — now fixed. The 19-test behavior unit suite validates the scheduling logic end-to-end.

---

## 7. Cross-Reference: Insight Notes

The `~/insights/` folder contains session-level notes. Topics that intersect with this project:

* `~/insights/cpp_api_design_patterns.md` — proxy/fluent pattern, `flex_*` fallback design. Read this before designing the `MLCouplingApplicationMaia` API surface.
* `~/insights/phydll_integration.md` — PhyDLL C++ and Python DL-side bug fixes, dynamic tensor shapes, OOM chunking, MPI finalization ordering. Relevant if/when we test the new interface with PhyDLL.
* `~/insights/module_test.md` — module_test session log; SmartSim GPU indexing, model caching, step timing.
* `~/insights/smartsim_multi_model_multiplexing.md` — model multiplexing on 96-way CPU: OOM at 1.6 GB models, 6.8× slowdown at 699 MB. Implication: do **not** multiplex large transformer models on CPU.
* `~/insights/smartsim_cmi_batch_size_bug.md` — `batch_size=0` is required to disable the RedisAI `BATCHSIZE` limit when the solver sends variable-sized inputs. Important if/when we wire up SmartSim.
* `~/insights/hpc_parallel_build_infrastructure.md` — `slurm_build.sh` self-submitting wrapper, dynamic `-j` from `SLURM_CPUS_ON_NODE`, hardcoded `-DTORCH_CUDA_ARCH_LIST="9.0"` for H100 targets on GPU-less `devel` nodes. The MAIA build wrapper (`slurm/slurm_install_maia.sh`) can adopt these patterns.

---

## 8. How to Reproduce the Test Job (Current `dev` Branch)

```bash
cd /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
sbatch slurm/slurm_install_maia.sh           # build MAIA (≈10 min on c23mm devel)
./slurm/run_new_example_job_devel_24.sh       # submit a 1-node, 12-rank job with snapshots enabled
# After completion, find the snapshot file at /hpcwork/thes2181/mmcp/snapshots_<JOB_ID>.h5
```

To disable snapshots for a normal run, simply unset `MAIA_SNAPSHOT_DIR` in the runner script (or pass `MAIA_SNAPSHOT_DIR= ./slurm/run_new_example_job_devel_24.sh`).
