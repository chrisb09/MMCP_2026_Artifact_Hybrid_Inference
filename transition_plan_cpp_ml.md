# Transition Plan: Integrating the New CPP-ML-Interface (v2)

> Supersedes the original `transition_plan_cpp_ml.md`. Companion to `DEVELOPMENT.md` §6.

## 0. Goal and Verification Bar

Replace the old `CPP-ML-Interface-old` MAIA coupling (provider-specific subclasses `MLCouplingMaiaAix`/`PhyDLL`/`Ref`, compile-time `#ifdef` provider selection, global `m_mlCoupler`) with the new redesigned `CPP-ML-Interface` (branch `redesign/coupling-interface`, HEAD `55baf5b`): TOML-driven config, registry/factory, `MLCouplingApplicationFlowExtrapolator`, a new `MLCouplingBehaviorFlowExtrapolator`, and an application-led `ml_step` orchestration.

**Functional-equivalence bar:** reproduce the old rank-0 U/V/W fields at every coupling/inference step to floating-point tolerance, verified by diffing against the golden reference `/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5` (job 874293, 12 ranks, `properties_run_les_ref_medium.toml`, both `"sent"` and `"received"` groups).

**Non-goals (port):** SmartSim provider (out of scope, known flex-API issues), field-value normalization (old code has none), training.

## 1. Resolved Design Decisions

| # | Decision | Resolution |
|---|----------|------------|
| D0 | Submodule vs symlink | **Convert first.** Remove the `CPP-ML-Interface` symlink; fix `.gitmodules` URL `chrisb09/CPP-ML-Interface-MAIA.git` -> `chrisb09/CPP-ML-Interface.git`; checkout `redesign/coupling-interface` (`55baf5b`) as a real submodule. |
| A | Where app/behavior classes live | **Inside `CPP-ML-Interface/include/{application,behavior}/`** so the upstream `generate_registry.py` picks up `@registry_name` annotations. No MAIA-local merged registry. `MLCouplingApplicationFlowExtrapolator` already exists upstream. |
| B | Orchestration model | **Application-led, opt-in.** New virtual `ml_step(MLCouplingProvider<In,Out>&, MLCouplingBehavior&)` on `MLCouplingApplication`. `MLCoupling::step()` delegates to it. **Remove** the legacy `ml_step(){step()}` compat alias and the dead `ml_step(MLCouplingData<In>)` virtual. Migration guide documents the change. |
| C | Time-step orchestration | **`MLCouplingBehaviorFlowExtrapolator`** subclassing `MLCouplingBehavior` directly (not `Periodic`). Proactive next-cycle computation; reactive `should_*()` answers. |
| E3 | Normalization | **Omit `[normalization]`** for the like-for-like port (old code sends raw field values). |
| T1 | Scalar type | **Universal `MLCoupling<float,float>`.** Solver `double*` fields cast to `float` at the `MLCouplingData<float>` boundary. AIx native float; PhyDLL auto-casts through its internal `double` transmission buffer. |
| T2 | `step()` return | **`MLCoupling::step()` returns `int`** = step delta: 0 for no-op / coupling-only, N for inference. Solver branches on the return. |
| T3 | Behavior step tracking | Behavior tracks `logical_step_count` internally (from 0); `global_step = logical_step_count + global_step_offset`. `global_step_offset` is a constructor parameter with default 0, surfaced as a TOML key / `ConfigOverrides` value. No per-step solver->behavior callbacks. |
| T4 | What `ml_step` receives | `MLCouplingProvider<In,Out>&` and `MLCouplingBehavior&` only. |
| T5 | Stride | Derived internally from `round(input_step_distance * scaling_factor)`. Not a separate TOML key. Current test config: stride = 1. |
| T6 | HDF-avoidance | Private internal method (the range-check from old `:275`); when triggered, shifts `next_inference_step` forward. No `prohibit_inference` lambda hook (we don't inherit from `Periodic`). |
| T7 | `should_send_data()` | Computed on-the-fly from `next_inference_step` -- no `coupling_steps` vector. |

## 2. Architecture After Transition

```
Solver (FvStructuredSolver)
  -> owns std::unique_ptr<MLCoupling<float,float>> m_mlCoupler
       |- int step()   // returns step delta (0 or N)
       |     -> application->ml_step(provider&, behavior&)
       |           |- behavior.should_send_data()? -> prepare_input() (grow history)
       |           |- behavior.should_perform_inference()? -> provider.static_inference()
       |           -> return behavior.time_step_delta() or 0
       |- MLCouplingProvider<float,float>  (Aixelerator | Phydll | Dummy)
       |- MLCouplingApplicationFlowExtrapolator  (cubing + history + un-cubing)
       +- MLCouplingBehaviorFlowExtrapolator  (coupling/inference timing + HDF avoidance + scaling)
```

## 3. Phase-by-Phase Execution Plan

### Phase 0 -- Submodule cutover (blocking, first)

1. `rm CPP-ML-Interface` (the symlink); `git rm --cached CPP-ML-Interface`.
2. Edit `.gitmodules`: `url = git@github.com:chrisb09/CPP-ML-Interface.git`.
3. `git submodule add git@github.com:chrisb09/CPP-ML-Interface.git CPP-ML-Interface`; `cd CPP-ML-Interface && git checkout redesign/coupling-interface` (pin `55baf5b`).
4. Commit the new pointer; verify `git status` is clean.
5. Check `install-MAIA.sh`'s `source ./CPP-ML-Interface/extern/python/venv/bin/activate` path against the new repo layout; adjust if moved.
6. Keep `CPP-ML-Interface-old/` in-tree as reference until Phase 7 passes; delete in Phase 8.

### Phase 1 -- Upstream: application-led `ml_step`

**Repo:** `CPP-ML-Interface`, `redesign/coupling-interface`.

1. **`include/application/ml_coupling_application.hpp`:**
   - Add new virtual `virtual int ml_step(MLCouplingProvider<In,Out>& provider, MLCouplingBehavior& behavior)`.
   - **Base default implementation** (reproduces current `step()` semantics + returns the delta):
     ```cpp
     virtual int ml_step(MLCouplingProvider<In,Out>& provider, MLCouplingBehavior& behavior) {
         if (behavior.should_perform_inference()) {
             prepare_input();
             provider.static_inference(&input_data_after_preprocessing,
                                       &output_data_before_postprocessing);
             finalize_output();
             return behavior.time_step_delta();
         }
         return 0;
     }
     ```
   - **Remove** the old `virtual MLCouplingData<Out> ml_step(MLCouplingData<In>)` (dead code, never called by `step()`).
   - Make `coupling_step(MLCouplingData<In>)` non-pure (empty default) -- unused in the provider-driven path.
2. **`include/ml_coupling.hpp`:**
   - Change `void step()` -> **`int step()`**:
     ```cpp
     int step() {
         if (provider && application && behavior) {
             return application->ml_step(*provider, *behavior);
         }
         return 0;
     }
     ```
   - **Remove** the `void ml_step() { step(); }` backward-compat alias (line 311).
   - Make `provider` and `behavior` accessible (add `protected` accessors or befriend the application, or pass `*provider.get()`, `*behavior.get()`).
3. **`documentation/migration_guide.md`:** add section "Application-led orchestration" covering: new `ml_step(provider&,behavior&)` signature; removed `ml_step()` alias (callers rename to `step()`); `step()` now returns `int`; how to opt in by overriding `ml_step`. List known-affected consumers: `mini_app/solver_cpp`, `module_test/solver.cpp`.

### Phase 2 -- Upstream: affected consumers

Per the decision not to clutter CMI with legacy support, we do **not** retroactively port mini_app/module_test in this transition. They are listed in the migration guide as known-affected; fixing them is a separate follow-up.

### Phase 3 -- Upstream: `MLCouplingBehaviorFlowExtrapolator`

**New file:** `include/behavior/ml_coupling_behavior_flow_extrapolator.hpp`, `@registry_name: FlowExtrapolatorBehavior` (aliases `flow-extrapolator-behavior`, `maia-flow-extrapolator-behavior`).

**Subclasses `MLCouplingBehavior` directly** (not `MLCouplingBehaviorPeriodic`). The base is a clean abstract interface (3 pure virtuals, no state).

Constructor params (all TOML-configurable; `global_step_offset` defaults 0):
```
inference_interval, coupled_steps_before_inference,
step_increment_after_inference, hdf_output_interval, total_timesteps,
scaling_factor, forecast_window, input_step_distance,
inference_start_step, global_step_offset = 0
```

Internal state:
- `logical_step_count` (from 0, incremented on each `should_perform_inference()` call)
- `global_step = logical_step_count + global_step_offset`
- `next_inference_step` (init `inference_start_step`)
- `stride` = `round(input_step_distance * scaling_factor)` (computed once in constructor)

**`should_perform_inference()`** (reactive answer + proactive next-cycle computation):
```cpp
bool should_perform_inference() override {
    logical_step_count++;
    if (logical_step_count != next_inference_step)
        return false;

    // Inference fires NOW. Proactively compute next cycle.
    int increment = time_step_delta();
    int next_logical = logical_step_count + inference_interval;
    int next_global = next_logical + global_step_offset;

    if (next_logical + increment >= total_timesteps) {
        next_inference_step = total_timesteps + 1;  // no more coupling
    } else if (is_hdf_unsafe(next_global, increment)) {
        // HDF-avoidance shift (old :281)
        next_inference_step = next_logical
            + (hdf_output_interval - ((next_global - 1) % hdf_output_interval));
    } else {
        next_inference_step = next_logical;
    }
    return true;
}
```

**`is_hdf_unsafe()`** (private, ports old `:275` range check):
```cpp
bool is_hdf_unsafe(int next_global, int increment) const {
    int remainder = next_global % hdf_output_interval;
    return !(remainder > 0 && remainder < (hdf_output_interval - increment));
}
```

**`should_send_data()`** (reactive, on-the-fly from `next_inference_step`):
```cpp
bool should_send_data() override {
    int dist = next_inference_step - logical_step_count;
    return dist >= 0
        && dist < coupled_steps_before_inference * stride
        && dist % stride == 0;
}
```
No `coupling_steps` vector -- derived each call. When `next_inference_step` shifts (HDF), `should_send_data()` automatically reflects the new target.

**`time_step_delta()`** (ports old `:290`):
```cpp
int time_step_delta() override {
    return static_cast<int>(std::round(step_increment_after_inference
                                       * scaling_factor * forecast_window));
}
```

### Phase 4 -- Upstream: verify/fix `MLCouplingApplicationFlowExtrapolator`

Already exists (`include/application/ml_coupling_application_flow_extrapolator.hpp`). Verified matches old code on: cube extraction (ghost-inclusive `[z,y,x]`), `linspace`/`get_full_indices` with overlap, batch layout `[nFields*numCubes][seqLen][cubeSize]`, `batch_index=f*numCubes+c`, forecast-window-1 selection, weight-averaged un-cubing, active-region clearing.

Required changes:
1. **Override `ml_step(provider&, behavior&)`** to drive the iter/history logic:
   - If `behavior.should_send_data()` -> `prepare_input()` (grows `history_` by one sample).
   - If `behavior.should_perform_inference()` -> `provider.static_inference(...)` -> `finalize_output()` -> return `behavior.time_step_delta()`.
   - Else return 0.
2. **Remove** the dead `MLCouplingData<Out> ml_step(MLCouplingData<In>)` stub and the `coupling_step` stub.
3. **Scalar type:** instantiated as `<float,float>` (T1). `make_input_buffer`/`from_flat_copy` allocate `In` (=float) storage. The solver glue copies double->float when building `MLCouplingData<float>` wrappers.
4. **`inputStepDistance` spacing:** the application has no notion of it -- the behavior ensures the solver only enters the coupling path on correctly-spaced steps. Document this contract in the header.
5. **Partial-history duplication** (`resolve_history_index`): safe because `should_perform_inference()` won't fire until enough samples are gathered (`coupled_steps_before_inference` samples at stride spacing). Set `coupled_steps_before_inference = inputSeqLen` in config.

### Phase 5 -- MAIA: solver integration (`feature/new-cpp-ml-interface`)

1. **`maia/src/globals/globalvariables.{h,cpp}` + `maia/src/maia.cpp`:**
   - Remove the 3 `#ifdef` `m_mlCoupler` globals (`globalvariables.h:25,31,37`, `.cpp:21,26,31`).
   - Remove `m_mlCoupler = std::make_unique<MLCouplingMaia{PhyDLL,Aix,Ref}>()` + `init()` + `getComm()` + `finalize()` in `maia.cpp:132-142,291`.
   - Remove `#include "ml_coupling/maia/..."` headers.
2. **`maia/src/FV/fvstructuredsolver.h`:**
   - Replace `std::unique_ptr<MlCouplingStrategy> m_mlCoupler` (under `WITH_PHYDLL_DIRECT`, :941) with an **unconditional** `std::unique_ptr<MLCoupling<float,float>> m_mlCoupler`.
   - Remove `#include "ML/mlCouplingStrategy.h"` (:34); add `#include "ml_coupling.hpp"`.
3. **`maia/src/FV/fvstructuredsolver.cpp`:**
   - **Setup** (replace :307-402): build `MLCouplingData<float>` wrappers around `phyFields` (U/V/W `double*`): copy double->float into owned `float` buffers, wrap with `MLCouplingTensor<float>::from_flat_copy` and dims `[nCells[0],nCells[1],nCells[2]]` per field. Call `MLCoupling<float,float>::create_from_config("config.toml", std::move(input_data), std::move(output_data), overrides)`. Build `ConfigOverrides` using the `dotted` map from the existing `Context::getBasicProperty` values:
     - `["provider.model_path"]`, `["provider.app_comm"]` = `<void*>globalMaiaCommWorld()`, `["provider.enable_hybrid"]`, `["provider.host_fraction"]`
     - `["behavior.global_step_offset"]` = `m_restartTimeStep`
     - Other behavior/application params can be static in TOML or overridden.
   - **`solutionStep()`** (:8463-8565): replace the entire `#if defined(WITH_PHYDLL_DIRECT) || ... #endif` block with:
     ```cpp
     RECORD_TIMER_START(m_timers[Timers::MLCoupling]);
     snapshot::write_step(..., globalTimeStep, "sent");
     int delta = m_mlCoupler->step();
     if (delta > 0) {
         snapshot::write_step(..., globalTimeStep, "received");
         m_physicalTime += delta * m_timeStep * m_timeRef;
         m_time += delta * m_timeStep;
         globalTimeStep += delta;
         step = true;
     }
     RECORD_TIMER_STOP(m_timers[Timers::MLCoupling]);
     if (delta == 0) {
         rhs(); rhsBnd(); rungeKuttaStep(); setTimeStep(); lhsBnd();
     }
     ```
     This replaces the old `isCouplingStep`/`isInferenceStep` queries, `m_mlCoupler->inference(...)`/`ml_step()` calls, `getInferenceIncrement`/`setNextInferenceStep` calls, and all `#ifdef` wrappers. The skip-`rhs`-on-inference semantics are preserved via `if (delta == 0)`.
   - Remove all `WITH_PHYDLL_DIRECT`/`WITH_AIXSERVICE`/`WITH_PHYDLL`/`WITH_REFERENCE_MODEL` `#ifdef`s in this file.
4. **`maia/src/CMakeLists.txt:162-164`:** remove `add_definitions("-DWITH_AIXSERVICE")`, `WITH_ML_INTERFACE`; add `add_definitions("-DUSE_CPP_ML_INTERFACE")`; wire CMI include/lib path; pass `-DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_TORCH=ON` down.
5. **`install-MAIA.sh` / `slurm/slurm_install_maia.sh`:** adjust to the new CMI layout (venv path, provider flags, `-DTORCH_CUDA_ARCH_LIST="9.0"` for H100 if GPU).

### Phase 6 -- Config + build

Create `config.toml` (repo root or `input/`):
```toml
[general]
coupling_type = "STATIC"

[logging]
level = "info"

[provider]
class = "Aixelerator"        # or "Phydll"
model_path = "<modelPath>"
device = "CPU"               # or "GPU"

[behavior]
class = "FlowExtrapolatorBehavior"
inference_interval = <mlInterval>
coupled_steps_before_inference = <mlInputLength>
step_increment_after_inference = <mlStepCoefficient>
hdf_output_interval = <solutionInterval>
total_timesteps = <timeSteps>
scaling_factor = <mlScalingFactor>
forecast_window = <mlForecastWindow>
input_step_distance = <mlInputStepDistance>
inference_start_step = <mlStart>    # = mlInterval
global_step_offset = 0              # overridden at runtime with m_restartTimeStep

[application]
class = "MLCouplingApplicationFlowExtrapolator"
cube_dimension = <mlCubeD>
cube_overlap = <mlCubeOverlap>
input_sequence_length = <mlInputLength>
forecast_window = <mlForecastWindow>
n_ghost_layers = <m_noGhostLayers>
```
No `[normalization]` section. MPI comm, `model_path`, `global_step_offset` injected via `ConfigOverrides` at runtime.

### Phase 7 -- Verification

1. Build MAIA + new CMI (`slurm/slurm_install_maia.sh`).
2. Run the 12-rank job (`slurm/run_new_example_job_devel_24.sh`, same `properties_run_les_ref_medium.toml`, `MAIA_SNAPSHOT_DIR` set).
3. Produce `new_snapshots_rank0.h5` at `/hpcwork/thes2181/mmcp/`.
4. Diff vs `reference_snapshots_rank0.h5`: per-step U/V/W max-abs + L2 at rank 0, for every `/step_NNNN/{U,V,W}` group, both `type="sent"` and `type="received"`.
5. **Pass:** within float tolerance for the deterministic AIx provider.
6. **If divergence,** check in order: (a) double->float copy at the data boundary, (b) history spacing (does `should_send_data()` fire on the right steps?), (c) HDF-avoidance shift, (d) `scaling_factor` rounding, (e) `global_step_offset` correctness, (f) cube extraction indexing.
7. Reference trace for sanity: old force-coefficient file shows first coupling block at logical steps 11,12,13,14 (`mlStart=11`, `inputSeqLen=4`, stride=1), inference at 14, jump to ~39, etc. The new behavior should reproduce this step pattern. Current test config uses `mlInterval=5`, `mlInputLength=5`, `mlStepCoefficient=12`, `mlForecastWindow=2` -> increment=24, stride=1.
8. Record results in `DEVELOPMENT.md`.

### Phase 8 -- Cleanup

1. Remove `CPP-ML-Interface-old/` from the tree.
2. Remove any leftover `maia/src/ML/` strategy headers if present.
3. Push upstream `redesign/coupling-interface` changes (Phases 1, 3, 4); bump the MAIA submodule pointer.
4. Update `DEVELOPMENT.md` §6 and this plan to "completed."
5. (Follow-up, not blocking) port mini_app/module_test per the migration guide.

## 4. Risk Register

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `ml_step` redesign breaks mini_app/module_test | Acceptable per decision; documented in migration guide. Not a MAIA blocker. |
| R2 | Model/dtype mismatch | T1 uses `<float,float>` matching old AIX; PhyDLL auto-casts. Verify AIx `AIxeleratorService<float>` links. |
| R3 | History not grown on coupling-only steps | Application's `ml_step` calls `prepare_input()` whenever `should_send_data()`. |
| R4 | HDF-avoidance behavior divergence | Port `setNextInferenceStep` logic (restructured); diff step pattern against force-coefficient trace. |
| R5 | `ConfigOverrides` API misuse | Use the real `dotted`/`sections` maps; test with `MLCOUPLING_LOG_LEVEL=debug`. |
| R6 | Install scripts / venv path break after cutover | Phase 0.5 + Phase 5.5 check. |
| R7 | `global_step_offset` wrong on restart runs | Config-injected from `m_restartTimeStep`; verify with a restart test. |

## 5. File-Change Summary

**Upstream (`CPP-ML-Interface`, `redesign/coupling-interface`):**
- Edit `include/application/ml_coupling_application.hpp` (new `ml_step(provider&,behavior&)` + base default; remove old `ml_step(MLCouplingData<In>)`; `coupling_step` non-pure)
- Edit `include/ml_coupling.hpp` (`step()` -> `int`; remove `ml_step(){step()}` alias; expose provider/behavior)
- Edit `include/application/ml_coupling_application_flow_extrapolator.hpp` (override `ml_step(provider&,behavior&)`; remove dead stubs)
- New `include/behavior/ml_coupling_behavior_flow_extrapolator.hpp`
- Edit `documentation/migration_guide.md`

**MAIA repo (`feature/new-cpp-ml-interface`):**
- Edit `.gitmodules`, submodule pointer (Phase 0)
- Edit `maia/src/globals/globalvariables.{h,cpp}`, `maia/src/maia.cpp`
- Edit `maia/src/FV/fvstructuredsolver.{h,cpp}`
- Edit `maia/src/CMakeLists.txt`
- Edit `install-MAIA.sh`, `slurm/slurm_install_maia.sh`
- New `config.toml`
- Delete `CPP-ML-Interface-old/` (Phase 8)
