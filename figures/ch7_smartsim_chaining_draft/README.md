# SmartSim Chaining Coverage

Blocked: the canonical MAIA GPU snapshot contains only the shared-DB baseline
(labelled c0 for the requested comparison). Matching c1 and c3 measurements are
absent for all four hardware placements and both models. No chaining figure is
generated: missing observations are not zero runtimes. No CPU or terrain data,
pilots, or unmatched historical measurements are substituted. No jobs launched.

## Coverage

`coverage.csv` contains all 24 requested model/placement/chain combinations:
8 available c0 groups, each with three repetitions and five measured calls per
repetition, and 16 absent c1/c3 groups. Blank timings/configuration fields denote
absence; `n_reps=0` is a coverage count, not a zero latency. The c0 repetition
means are copied without presentation rounding from
`../../results/cmi_bench_consolidated/runs.csv`, using only provider=smartsim,
layout=shared. Cell identities join back to job IDs, selected retry lineage,
run directories and workload metadata in that table. Calls 1-2 are warmup;
only calls 3-7 (steps 73, 102, 131, 160, 189) contribute to each mean.

## Configuration Caveat

Verified against each of the 24 selected runs' `controller.log` files:

- MLP (ov4): TPQ=1 on all four placements.
- Transformer (ov0): TPQ=4 for single-node 24c1g and 96c4g; TPQ=1 for heterogeneous 24p24c1g and 96p96c4g.
- All c0 groups: INTRA=4, INTER=1, shared DB.

The single-node four-GPU Transformer baseline also uses TPQ=4; it must not be
silently relabelled TPQ=1. The baseline is not a controlled chaining comparison
across placements. Workload aggregates also differ across placements, and the
heterogeneous placements use c23g+c23g surrogates with 48/192 total solver ranks.
Matching c1/c3 data must preserve the corresponding c0 model, overlap, workload,
placement, shared-DB layout and thread settings before comparing chaining.

## Verification

`scripts/test_plot_cmi_ch7_readiness.py` checks every c0 cell and repetition mean
against the canonical table and every TPQ/INTRA/INTER setting against its selected
controller log. The readiness draft's `manifest.json` records the SHA-256 of the
same canonical `runs.csv` plus `inferences.csv` and the consolidation report.
This is a static coverage audit, intentionally separate from the readiness
renderer; regenerating readiness does not change this audit.

Canonical `runs.csv` SHA-256 at audit time:
`7a149e18053301c8563294bc0b5d326fef7f0d4cdb35bb1384e2ba4b07b32adb`.
