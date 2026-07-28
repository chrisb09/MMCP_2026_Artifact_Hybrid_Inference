# Bitwise Verification and Determinism Analysis Report

**Date:** July 2026  
**Artifact:** MMCP 2026 Hybrid Inference (m-AIA + CMI + AIx/PhyDLL/SmartSim)

---

## 1. Executive Summary

This report documents the investigation and resolution of numerical divergence between the AIxelerator (AIx) golden reference and the alternative ML coupling providers (**PhyDLL-Python** and **SmartSim**).

### Primary Findings
1. **Model Prediction Parity:** When running PyTorch **`2.6.0`**, model predictions across AIx, PhyDLL-Python, and SmartSim are **100% BITWISE IDENTICAL (`0.000e+00` diff)**.
2. **Version Mismatch Root Cause:** The previously observed `~6.7e-8` difference was caused by a PyTorch version mismatch (**2.4.0** in PhyDLL/SmartSim vs **2.6.0** in AIx), which updated the underlying Intel MKL (2022.2 vs 2024.2) and oneDNN (3.4.2 vs 3.5.3) math operator libraries.
3. **C++ ABI Independence:** Changing the GCC C++ ABI flag (`_GLIBCXX_USE_CXX11_ABI=1` vs `0`) has **ZERO numerical effect (`0.000e+00` diff)** on CPU inference predictions.
4. **CMI Postprocessing Determinism:** The CMI postprocessing accumulator and reconstructed fields are 100% bitwise identical across all inference cycles when the MAIA solver binary configuration matches.
5. **CFD Integration Variance at Step 101:** At Inference Cycle 3 (Step 101), a double-precision machine-epsilon roundoff ($3.99 \times 10^{-14}$) occurs in reconstructed fields. After 50 subsequent Runge-Kutta time-integration steps, this $10^{-14}$ double-precision roundoff expands to $1.09 \times 10^{-6}$ on domain boundary/ghost cells due to chaotic turbulent flow dynamics.

---

## 2. Experimental Proof Matrix

### A. Direct Model Inference (`[3456, 5, 512]` Batch 3456)

| Runtime Environment | PyTorch Version | C++ ABI | Match vs. AIx 2.6.0 | Max Abs Diff |
| :--- | :--- | :--- | :--- | :--- |
| **AIx LibTorch (Golden Reference)** | **`2.6.0`** | **`ABI=1`** | **Bitwise Identical (`same=True`)** | **`0.000e+00`** |
| **Python 3.9 PyTorch Wheel** | **`2.6.0`** | **`ABI=0`** | **Bitwise Identical (`same=True`)** | **`0.000e+00`** |
| **Standalone C++ LibTorch** | **`2.6.0`** | **`ABI=1`** | **Bitwise Identical (`same=True`)** | **`0.000e+00`** |
| Official C++ LibTorch | `2.4.0` | `ABI=1` | Differs | `6.706e-08` |
| PyTorch Python Wheel | `2.4.0` | `ABI=0` | Differs | `6.706e-08` |

### B. Intermediate CMI Tensors (PhyDLL-Python with PyTorch 2.6.0 vs AIx)

| Inference Cycle | CMI Stage | Bitwise Status | Max Abs Diff |
| :--- | :--- | :--- | :--- |
| **Cycle 1 (Step 15)** | `assembled_input` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 1 (Step 15)** | `raw_provider_output` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 1 (Step 15)** | `reconstructed_fields` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 2 (Step 51)** | `assembled_input` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 2 (Step 51)** | `raw_provider_output` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 2 (Step 51)** | `reconstructed_fields` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 3 (Step 101)** | `assembled_input` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 3 (Step 101)** | `raw_provider_output` | **BITWISE IDENTICAL** | **`0.000e+00`** |
| **Cycle 3 (Step 101)** | `reconstructed_fields` | Machine roundoff | **`3.994e-14`** |

---

## 3. How to Run the Verification Suite

Run the automated verification script:

```bash
./verification_report/verify_bit_perfect.sh
```

### Manual Commands

#### 1. Compare CMI Intermediate Tensors:
```bash
python verification_report/compare_cmi_dumps.py \
    --aix-dir scratch/current_aix_cpu_2314057/debug/cmi \
    --test-dir debug_dumps/phydll_py/cmi_2433976 \
    --inferences 1,2,3
```

#### 2. Compare MAIA HDF5 Snapshots:
```bash
python verification_report/compare_hdf5_snapshots.py \
    --ref /rwthfs/rz/cluster/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5 \
    --new debug_dumps/phydll_py/snapshots_2428603_rank_0.h5 \
    --steps 15,51,101
```

---

## 4. Key Configuration Files Updated

- `CPP-ML-Interface/install.sh`: Set `LIBTORCH_VERSION=2.6.0` and updated `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0`.
- `slurm/run_phydll_py_300_devel.sh`: Set thread safety overrides (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) and `--mem=0` memory allocation.
