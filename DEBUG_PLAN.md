# Debug Plan: Post-Inference Step-15 Divergence

## Background

- **Symptom:** Steps 11-14 (pre-ML "sent" snapshots) match reference with **0 diff**. Step 15 (post-ML "received") diverges: `max_abs = 9.96e-02 (U), 1.66e-02 (V), 2.52e-02 (W)`.
- **Reference provenance:** Generated with the **old/original MMCP+CMI combination** (before CMI's rewrite). So "matches reference" = "matches old pipeline".
- **Algorithm verification:** Cube extraction, input assembly, normalization, delta computation verified **algorithmically identical** between old (`MlCouplerPhyDLL`) and new (`FlowExtrapolator`) CMI.
- **Model:** Same file `input/transformer_inference_scripted_fw2.pt` in both.
- **Old CMI code:** Available at `/tmp/cmi-maia-old` @ `9794f57` but **cannot be built** (submodule remote `CPP-ML-Interface-MAIA.git` unreachable).

## Goal

Pinpoint exactly where the post-inference divergence originates — input tensor assembly, model execution, or output reconstruction — **without** building the old C++ code. We do this by dumping all intermediate states from the new pipeline and comparing against a standalone Python reimplementation of the same algorithm.

## Architecture: Where FlowExtrapolator Is Compiled

**Key question answered:** Do we need `-DFLOW_DUMP_DEBUG` on the CMI build, the MAIA build, or both?

```
maia/src/CMakeLists.txt:168
    add_subdirectory("${SRC_DIR_ABS}/../../CPP-ML-Interface" ...)
```

- MAIA builds CMI **in-tree** via `add_subdirectory`. There is no separate CMI library linked into MAIA.
- The FlowExtrapolator template is instantiated in CMI's `src/c_api.cpp`, which `#include "generated_registry.hpp"` — and `generated_registry.hpp` includes `ml_coupling_application_flow_extrapolator.hpp`.
- `add_subdirectory` causes the child project to **inherit the parent's `CMAKE_CXX_FLAGS`**.
- Therefore: adding `-DFLOW_DUMP_DEBUG` to **only** `maia/build_gnu_production`'s `CMAKE_CXX_FLAGS` is sufficient. The standalone `cmi-build-gpu` / `cmi-build-debug` dirs are **not** used by MAIA.
- The debug flag is a preprocessor `#ifdef` inside the FlowExtrapolator header. Since the header is included and the template instantiated during the MAIA build (via in-tree CMI), the flag propagates correctly.

**Build command (answer to question #3):**
```bash
# Inside maia/build_gnu_production (the existing production build dir):
cmake . -DCMAKE_CXX_FLAGS:STRING="-Wno-array-bounds -DFLOW_DUMP_DEBUG"
# Then rebuild from maia/ directory:
make -j 96
```
This single cmake invocation covers both MAIA and in-tree CMI code. No separate CMI rebuild needed.

## Step 1: Add Comprehensive Dumps to FlowExtrapolator

### Intermediate states to dump (per ML coupling step)

**Send steps** (`should_send_data=true`, `should_perform_inference=false`):
| # | Phase | Description | Data |
|---|-------|-------------|------|
| 1 | `raw_input` | Solver fields before cube extraction | 3 × field arrays (U, V, W) as float |

**Inference steps** (`should_send_data=true`, `should_perform_inference=true`):
| # | Phase | Description | Data |
|---|-------|-------------|------|
| 1 | `raw_input` | Solver fields before cube extraction | 3 × field arrays (U, V, W) as float |
| 2 | `assembled_input` | Input tensor after preprocessing (cube assembly) | 1 × tensor as float |
| 3 | `raw_output` | Model output before postprocessing | 1 × tensor as float |
| 4 | `reconstructed_output` | Final reconstructed fields after postprocessing | 3 × field arrays (U, V, W) as float |

### Dump file naming convention

All dumps written to a configurable directory (default: `/tmp/flow_debug/`). Files named:

```
step{S:04d}_{phase}{_field{F}}.bin
```

Where:
- `S` = global solver time step (from behavior, not internal counter)
- `phase` = one of: `raw_input`, `assembled_input`, `raw_output`, `reconstructed_output`
- `F` = field index (0=U, 1=V, 2=W), only for field-level dumps

Example:
```
step0011_raw_input_field0.bin
step0011_raw_input_field1.bin
step0011_raw_input_field2.bin
step0015_assembled_input.bin
step0015_raw_output.bin
step0015_reconstructed_output_field0.bin
step0015_reconstructed_output_field1.bin
step0015_reconstructed_output_field2.bin
```

### Manifest file

A `manifest.txt` is written alongside the dumps, listing every file with its metadata:
```
# file,global_step,ml_step_type,phase,field_index,n_elements,element_size_bytes,shape
step0011_raw_input_field0.bin,11,send,raw_input,0,884736,4,[36][36][2][...]
step0015_assembled_input.bin,15,inference,assembled_input,-1, ...,4,[...]
...
```

### One-time metadata dump

At construction time, dump cube layout info to `cube_layout.txt`:
```
cube_dimension: 8
cube_overlap: 0
n_cubes_per_field: ...
cube_start_xs: [...]
cube_start_ys: [...]
cube_start_zs: [...]
grid_dims: [nz, ny, nx]
n_ghost_layers: ...
cube_volume_indices: [...]  # the local index mapping within a cube
```

### Implementation

All dump code guarded by `#ifdef FLOW_DUMP_DEBUG` so it has zero impact on production builds.

The dump step counter (global solver step) is obtained from the behavior class (which tracks `logical_step_count_`). Alternatively, maintain a static counter inside `ml_step()`.

**Rank gating:** Dumps only on MPI rank 0. Since the FlowExtrapolator doesn't have direct access to `domainId()`, pass the rank via a member variable set by the solver, or check an environment variable (`FLOW_DEBUG_RANK=0`). Simplest: check `getenv("FLOW_DEBUG_DUMP_DIR")` — if set, dump. If not set, skip.

## Step 2: Rebuild on Devel

1. **Fix the slurm script:** Remove the bad `source CPP-ML-Interface/setup_env_claix23.sh` line. Use direct cmake invocation on existing `maia/build_gnu_production` dir.
2. **Add flag:** `cmake . -DCMAKE_CXX_FLAGS:STRING="-Wno-array-bounds -DFLOW_DUMP_DEBUG"` inside `maia/build_gnu_production`.
3. **Rebuild:** `cd maia && make -j 96` (incremental; only recompiles files affected by the header change + relinks).

**slurm_rebuild_debug.sh:**
```bash
#!/usr/bin/zsh
#SBATCH --job-name=rebuild-debug
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=rebuild_debug_%j.txt
set -euxo pipefail
source setup_env_claix23.sh

# Ensure libclang for registry generation (if regeneration needed)
pip install "clang==17.0.6" "libclang==17.0.6" 2>&1
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${LD_LIBRARY_PATH:-}"

NPROC=${SLURM_CPUS_ON_NODE:-96}

# Add FLOW_DUMP_DEBUG to the existing production build's CXX flags.
# This propagates to in-tree CMI via add_subdirectory.
cd maia/build_gnu_production
CURRENT_FLAGS=$(cmake -LA . 2>/dev/null | grep "^CMAKE_CXX_FLAGS:STRING=" | sed 's/^CMAKE_CXX_FLAGS:STRING=//')
cmake . -DCMAKE_CXX_FLAGS:STRING="${CURRENT_FLAGS} -Wno-array-bounds -DFLOW_DUMP_DEBUG"
cd ..
make -j${NPROC}

echo "=== Done ==="
ls -lh build_gnu_production/maia
```

## Step 3: Short Simulation Run on c23g

- Run **20 solver steps** (enough for the first full inference cycle: steps 11-15, plus a few extra to confirm the delta jump).
- Set `FLOW_DEBUG_DUMP_DIR=/tmp/flow_debug` in the run environment.
- Use `timeSteps=20` in the config.
- After run: copy `/tmp/flow_debug/` to workspace as `debug_dumps/`.

## Step 4: Standalone Python Verification

Write `verify_inference.py` that:

1. **Loads reference snapshots** from `/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5` (groups for steps 11-15).
2. **Reads the cube layout** from `debug_dumps/cube_layout.txt`.
3. **Reimplements cube extraction** from the reference step-15 "sent" field.
4. **Assembles the 5-step input tensor** using the same history logic (steps 11-15).
5. **Loads the model** via `torch.jit.load("input/transformer_inference_scripted_fw2.pt")`.
6. **Runs inference** in Python.
7. **Reconstructs the output field** via the same scatter + normalize logic.

### Comparisons

| Comparison | What it tests |
|------------|---------------|
| Python input tensor vs `step0015_assembled_input.bin` | Input assembly correctness |
| Python model output vs `step0015_raw_output.bin` | Model execution correctness |
| Python reconstructed fields vs `step0015_reconstructed_output_*.bin` | Output reconstruction correctness |
| Python reconstructed fields vs reference step-15 "received" | Algorithm matches reference (old code) |
| `step0011_raw_input_*.bin` vs reference step-11 "sent" | Raw input matches reference (already known: 0 diff) |

### Decision tree

| Python input == C++ assembled? | Python output == C++ raw_output? | Python recon == reference received? | Conclusion |
|---|---|---|---|
| ✅ | ✅ | ✅ | New code is correct; reference snapshot uses same algorithm → the step-15 "received" diff we see is expected (old code had the same output) |
| ✅ | ❌ | ✅ | Provider/AIxeleratorService version differs between old & new CMI → investigate provider code |
| ✅ | ✅ | ❌ | Reference was made with a different algorithm → re-examine reference provenance |
| ❌ | — | — | Input assembly diverges despite algorithmic match → subtle indexing/offset bug → examine cube_layout.txt |

## Step 5: Fix HDF-Safe Scheduling Bug (After Step 4)

If the step-15 divergence is explained (e.g., it's the expected model output), the remaining issue is the **HDF-safe scheduling bug** in `should_perform_inference()`:

```
next_global = logical_step_count_ + inference_interval_ + global_step_offset_
```

This doesn't account for cumulative delta jumps (24 steps/inference). By the 3rd inference (step 73), `next_global=30` vs actual ~78, causing spurious `is_hdf_unsafe` triggers and misaligned data collection windows.

**Fix:** Track effective global step internally (accumulate `time_step_delta()` returns) rather than computing from `logical_step_count_ + inference_interval_`.

## File Checklist

- [x] `CPP-ML-Interface/include/application/ml_coupling_application_flow_extrapolator.hpp` — add `#ifdef FLOW_DUMP_DEBUG` dump blocks
- [x] `slurm_rebuild_debug.sh` — fixed rebuild script (removed standalone CMI build, only configures maia/build_gnu_production)
- [x] `slurm/debug_dump_run.sh` + `slurm/run_debug_dump_run.sh` — run scripts for 20-step debug run with `FLOW_DEBUG_DUMP_DIR`
- [x] `verify_inference.py` — standalone Python verification script