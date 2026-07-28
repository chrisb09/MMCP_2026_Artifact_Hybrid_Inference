#!/usr/bin/zsh
# Master script to execute and verify bitwise parity across AIx, PhyDLL-Python, and SmartSim

set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "${project_dir}"

source "${project_dir}/setup_env_claix23.sh"

REF_HDF5="/rwthfs/rz/cluster/hpcwork/thes2181/mmcp/reference_snapshots_rank0.h5"
AIX_DUMP_DIR="${project_dir}/scratch/current_aix_cpu_2314057/debug/cmi"
AIX_SNAPSHOT="${project_dir}/scratch/current_aix_cpu_2314057/debug/snapshots/snapshots_2314057_rank_0.h5"

PHYDLL_PY_DUMP_DIR="${project_dir}/debug_dumps/phydll_py/cmi_2433976"
PHYDLL_PY_SNAPSHOT="${project_dir}/debug_dumps/phydll_py/snapshots_2433976_rank_0.h5"

echo "========================================================================"
echo "   MMCP 2026 Artifact: Hybrid Inference Bitwise Verification Suite"
echo "========================================================================"
echo ""

echo "1. VERIFYING AIX (GOLDEN REFERENCE) AGAINST HISTORICAL REFERENCE SNAPSHOTS"
python "${script_dir}/compare_hdf5_snapshots.py" \
    --ref "${REF_HDF5}" \
    --new "${AIX_SNAPSHOT}" \
    --steps 15,51,101

echo ""
echo "2. VERIFYING PHYDLL-PYTHON CMI INTERMEDIATE TENSORS AGAINST AIX"
python "${script_dir}/compare_cmi_dumps.py" \
    --aix-dir "${AIX_DUMP_DIR}" \
    --test-dir "${PHYDLL_PY_DUMP_DIR}" \
    --inferences 1,2,3

echo ""
echo "3. VERIFYING PHYDLL-PYTHON SNAPSHOTS AGAINST AIX SNAPSHOTS"
python "${script_dir}/compare_hdf5_snapshots.py" \
    --ref "${AIX_SNAPSHOT}" \
    --new "${PHYDLL_PY_SNAPSHOT}" \
    --steps 15,51,101

echo ""
echo "========================================================================"
echo "   VERIFICATION COMPLETE SUCCESSFUL"
echo "========================================================================"
