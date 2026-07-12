# CONTEXT.md — MMCP 2026 Artifact Hybrid Inference Debugging Session

**Last updated:** 2026-07-05 19:22 CEST  
**Conversation ID:** `c2157c33-8fcd-446a-b7f9-7bd3733d8afc`

---

## 1. Project Overview

This is a hybrid CFD+ML inference project ("MMCP 2026 Artifact").  
The solver is **m-AIA** (structured 3D finite-volume CFD) with a coupled **ML inference pipeline**
(new CPP-ML-Interface, "CMI") that periodically replaces CFD time steps with ML-predicted fields.

- **CFD solver**: `maia/src/FV/fvstructuredsolver.cpp` — `FvStructuredSolver3D`
- **ML coupling**: `CPP-ML-Interface/` — `MLCouplingApplicationFlowExtrapolator`
- **Behavior scheduler**: `MLCouplingBehaviorFlowExtrapolator`
- **Provider**: `Aixelerator` (runs TorchScript `.pt` model via LibTorch)
- **Model**: `input/transformer_inference_scripted_fw2.pt`

### What the ML pipeline does
Every `mlInterval` solver steps, ML inference replaces the current flow fields (U, V, W) with
model-predicted future fields. The model takes a **sequence of 5 consecutive send-steps** as input
(assembled into cubes), runs inference, and outputs a future state
(`forecast_window * step_increment` steps ahead = 24 steps).

---

## 2. Problems Being Debugged

### Primary: Step-15 Divergence
When the solver performs its **first ML inference at solver step 15**, the output fields written to
`reference_snapshots_rank0.h5` as `type=received` differ from what the new CMI produces.
The reference was written by the **old CMI** code. We need to determine **where** the pipeline
diverges: input assembly, model execution, or output reconstruction.

### Secondary: HDF-Safe Scheduling Bug (Step 5 in DEBUG_PLAN.md)
`MLCouplingBehaviorFlowExtrapolator::should_perform_inference()` computes:

    next_global = logical_step_count_ + inference_interval_ + global_step_offset_

This does NOT account for cumulative delta jumps (each inference jumps 24 solver steps). By the
3rd inference cycle (step ~73), `next_global` ≈ 30 instead of ~78, causing spurious
`is_hdf_unsafe()` triggers. Fix: accumulate `time_step_delta()` returns to track effective step.

---

## 3. Repository Layout (Key Files)

    MMCP_2026_Artifact_Hybrid_Inference/
    ├── CPP-ML-Interface/include/
    │   ├── application/ml_coupling_application_flow_extrapolator.hpp  ← EDITED (dump blocks)
    │   ├── behavior/ml_coupling_behavior_flow_extrapolator.hpp        ← BUG HERE (Step 5)
    │   └── ml_coupling.hpp                                            ← step() dispatcher
    ├── maia/src/FV/
    │   ├── fvstructuredsolver.cpp        ← solutionStep() + CMI call + param parsing
    │   └── fvstructuredsolver3d.cpp      ← rungeKuttaStep()
    ├── maia/build_gnu_production/bin/maia  ← built binary (debug, WITH_AIX=ON)
    ├── slurm/debug_dump_devel.sh           ← 20-step CPU devel run script
    ├── slurm_rebuild_debug.sh              ← incremental rebuild script
    ├── verify_inference.py                 ← Python verification script (BUG: see §10)
    ├── config.toml                         ← CMI config (values overridden at runtime)
    ├── DEBUG_PLAN.md                       ← master 5-step plan
    ├── CONTEXT.md                          ← THIS FILE
    ├── debug_dumps/                        ← C++ debug dump outputs
    │   ├── cube_layout.txt
    │   ├── manifest.txt
    │   ├── step0000_raw_input_field{0,1,2}.bin    (step 11 send)
    │   ├── step0001_raw_input_field{0,1,2}.bin    (step 15 inference input)
    │   ├── step0001_assembled_input.bin
    │   ├── step0001_raw_output.bin
    │   └── step0001_reconstructed_output_field{0,1,2}.bin
    └── logs/
        ├── output_debug_devel_1710602.txt
        └── error_debug_devel_1710602.txt

---

## 4. Simulation Parameters

| Parameter           | Value | Notes                              |
|---------------------|-------|------------------------------------|
| mlInterval          | 5     | inference every 5 sends            |
| mlInputLength       | 5     | input sequence length              |
| mlStepCoefficient   | 12    |                                    |
| mlForecastWindow    | 2     |                                    |
| mlScalingFactor     | 1     |                                    |
| mlInputStepDistance | 1     |                                    |
| mlCubeOverlap       | 0     |                                    |
| mlCubeD             | 8     |                                    |
| solutionInterval    | 50    | HDF output interval                |
| noRKSteps           | 5     |                                    |
| Restart from step   | 10    | out/restart_les_ref_medium.hdf5    |

**Derived**: `time_step_delta = mlStepCoefficient * mlForecastWindow = 12 * 2 = 24`  
**Send steps (first cycle)**: 11, 12, 13, 14, 15  
**First inference**: solver step 15 → dump filename prefix `step0001`  
  (`debug_step_counter_=0` for sends at 11, `debug_step_counter_=1` at inference step 15)

---

## 5. Cube Layout (from `debug_dumps/cube_layout.txt`)

| Field               | Value                                    |
|---------------------|------------------------------------------|
| cube_dimension      | 8                                        |
| cube_overlap        | 0                                        |
| n_cubes_per_field   | 1152                                     |
| cube_size           | 512 (= 8³)                               |
| input_seq_length    | 5                                        |
| forecast_window     | 2                                        |
| n_ghost_layers      | 2                                        |
| grid_dims (rank 0)  | [67, 69, 126] → 582,498 cells            |
| active_cells        | [63, 65, 122]                            |
| cube_start_zs       | [0, 8, 16, 24, 31, 39, 47, 55]          |
| cube_start_ys       | [0, 7, 14, 21, 29, 36, 43, 50, 57]      |
| cube_start_xs       | [0, 8, 15, 23, 30, 38, 46, 53, 61, 68, 76, 84, 91, 99, 106, 114] |

Note: some ranks have 66×69×126 = 573,804 cells (confirmed from manifest.txt).

---

## 6. Reference HDF5 Layout

**File**: `/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5`  
**Group naming**: Sequential `step_XXXX` — NOT `step_{globalTimeStep}_{type}`.  
**Attributes on each group**: `globalTimeStep` (int64), `type` (bytes: b"sent" or b"received")  
**Datasets**: `U`, `V`, `W` (shape e.g. [67,69,126]), `nCells`, `nOffsetCells`

> ⚠️ CRITICAL: Must scan attributes to find groups; name search by globalTimeStep FAILS.

Mapping for the first two inference cycles:

| HDF5 group  | globalTimeStep | type     |
|-------------|---------------|----------|
| step_0000   | 11            | sent     |
| step_0001   | 12            | sent     |
| step_0002   | 13            | sent     |
| step_0003   | 14            | sent     |
| step_0004   | 15            | sent     |
| step_0005   | 15            | received |
| step_0006   | 47            | sent     |
| step_0007   | 48            | sent     |
| step_0008   | 49            | sent     |
| step_0009   | 50            | sent     |
| step_0010   | 51            | sent     |
| step_0011   | 51            | received |

---

## 7. Debug Run Results (Job 1710602)

- **Partition**: devel (CPU-only, 24 ranks, 20 steps)
- **Completed**: 2026-07-05 ~19:15 CEST
- **Dumps**: all expected files present in `debug_dumps/`
- **CMI behaviour** (rank 0 from error log):
  - 1st `step()` call → `ml_step returned 0` (step 11, send-only)
  - 2nd `step()` call → `ml_step returned 24` (step 15, inference fires)
- Confirms: send/inference timing is correct.

---

## 8. Environment

```bash
# Load env (Python 3.11.5, torch 2.4.0+cu124, numpy 1.26.4, h5py 3.14.0):
source setup_env_claix23.sh

# Binary:
maia/build_gnu_production/bin/maia

# Account note: thes2181 has NO quota left on c23mm/c23g.
# Use default account for devel partition jobs.

# Build (incremental):
sbatch slurm_rebuild_debug.sh
# or:
cd maia/build_gnu_production
cmake . -DCMAKE_CXX_FLAGS:STRING="-Wno-array-bounds -DFLOW_DUMP_DEBUG"
cd ..
make -j96

# Debug run:
sbatch slurm/debug_dump_devel.sh

# Check jobs:
squeue -u ro092286
```

---

## 9. Progress Status

| Step | Status | Notes |
|------|--------|-------|
| Step 1: C++ instrumentation | ✅ Done | Dump blocks added to FlowExtrapolator |
| Step 2: Rebuild | ✅ Done | slurm_rebuild_debug.sh works, WITH_AIX=ON fixed |
| Step 3: Debug run | ✅ Done | Job 1710602, dumps in debug_dumps/ |
| Step 4: verify_inference.py | 🔄 Blocked | Script has HDF5 loading bug (see §10) |
| Step 5: Fix scheduling bug | ❌ Not started | After Step 4 complete |

---

## 10. Current Blocker: Bug in `verify_inference.py`

### Location
`verify_inference.py` lines 420–449 (inside the `else:` branch of the HDF5 loading block).

### Bug
The code searches for groups by name pattern `step_{step:04d}_{snap_type}` (e.g. `step_0015_sent`).
These names don't exist. The actual format uses sequential names with **attributes**.
Result: all reference comparisons silently yield "N/A (missing data)".

### Fix (drop-in replacement for lines 420–449)

```python
    else:
        ref_fields = {}
        with h5py.File(args.ref, 'r') as f:
            all_groups = list(f.keys())
            print(f"  Available groups: {all_groups[:10]}{'...' if len(all_groups)>10 else ''}")

            # Build index from (globalTimeStep, type) -> group name by scanning attrs
            attr_index = {}
            for gname in all_groups:
                grp = f[gname]
                ts = grp.attrs.get('globalTimeStep', None)
                tp = grp.attrs.get('type', None)
                if ts is not None and tp is not None:
                    if isinstance(tp, bytes):
                        tp = tp.decode()
                    attr_index[(int(ts), tp)] = gname

            print(f"  Attribute index: {len(attr_index)} entries. Sample: {list(attr_index.items())[:4]}")

            for step in send_steps:
                for snap_type in ('sent', 'received'):
                    gname = attr_index.get((step, snap_type), None)
                    if gname is not None:
                        grp = f[gname]
                        ref_fields[(step, snap_type)] = {
                            field: np.array(grp[field]).flatten().astype(np.float32)
                            for field in ('U', 'V', 'W') if field in grp
                        }
                        print(f"  Loaded ({step}, {snap_type}) from '{gname}': "
                              f"shapes = { {k: v.shape for k,v in ref_fields[(step,snap_type)].items()} }")
                    else:
                        print(f"  WARNING: group for step {step} {snap_type} not found.")
```

---

## 11. Next Immediate Steps

### A. Fix verify_inference.py (lines 420–449)
Apply the attribute-scan fix from §10.

### B. Run verify_inference.py
```bash
source setup_env_claix23.sh
python verify_inference.py \
    --dumps debug_dumps \
    --ref /hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5 \
    --model input/transformer_inference_scripted_fw2.pt \
    --inference-step 15 \
    --send-steps 11,12,13,14,15 \
    --dump-step-offset 0
```

### C. Interpret Results (DEBUG_PLAN.md decision tree)

| Python input == C++ assembled? | Python output == C++ raw_output? | Python recon == ref received? | Conclusion |
|---|---|---|---|
| ✅ | ✅ | ✅ | New code correct; diff is expected |
| ✅ | ❌ | ✅ | Provider/model version diverges |
| ✅ | ✅ | ❌ | Reference uses different algorithm |
| ❌ | — | — | Input assembly indexing bug |

### D. Fix HDF-Safe Scheduling Bug (Step 5)
In `ml_coupling_behavior_flow_extrapolator.hpp` `should_perform_inference()`:
- Add member `long long int effective_global_step_ = inference_start_step_ + global_step_offset_`
- After setting `next_inference_step_`, add `effective_global_step_ += time_step_delta()`
- Use `effective_global_step_` instead of `next_logical + global_step_offset_` for `is_hdf_unsafe()` check

---

## 12. Key Code Locations (Line References)

| File | Lines | Purpose |
|------|-------|---------|
| `ml_coupling_application_flow_extrapolator.hpp` | ~112-210 | ml_step(), debug_step_counter_ |
| `ml_coupling_application_flow_extrapolator.hpp` | ~465-568 | dump helper functions |
| `ml_coupling_behavior_flow_extrapolator.hpp` | 55-80 | `should_perform_inference()` — SCHEDULING BUG |
| `ml_coupling_behavior_flow_extrapolator.hpp` | 90-100 | `should_send_data()` |
| `fvstructuredsolver.cpp` | 249-328 | ML coupler construction + param parsing |
| `fvstructuredsolver.cpp` | 8362-8420 | `solutionStep()` — CMI integration |
| `verify_inference.py` | 420-449 | HDF5 loading — **BUG: fix with attr scan** |
| `maia/src/CMakeLists.txt` | 166 | `WITH_AIX ON` (was OFF, now fixed) |
