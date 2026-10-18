# GPU Readiness Draft

MAIA GPU data only; no CPU or terrain measurements or substitutions.
Regenerate from the repository root using the thesis Python environment:

```sh
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 scripts/plot_cmi_ch7_readiness.py
```

Requires matplotlib, numpy, pdflatex with T1/Latin Modern, and pdftocairo on PATH.
The renderer uses the thesis guide's PGF/pdflatex pipeline; PDFs embed Latin Modern
Type 1 fonts and PNGs are rasterized from those PDFs at 300 dpi.

## Figures

- [Combined preview](readiness_latency_1x2.png): 150 x 65 mm, independent y scales.
- [MLP panel](mlp_latency.pgf): 73.5 x 65 mm; PDF and PNG companions share the stem.
- [Transformer panel](transformer_latency.pgf): 73.5 x 65 mm; PDF and PNG companions share the stem.

Panels are independently authored at 73.5 mm (0.49 textwidth), separated by 3 mm
in the full-width preview, not scaled-down full-size plots. Include at native size.
Axis labels/titles are 9 pt, ticks 8 pt, local legend 6.5 pt. No repetition dots,
bar-value annotations, figure captions or subfigure letters are baked in. Exact
values are in plot_values.csv; LaTeX owns the captions. PGFs need the pgf package;
use T1 font encoding and lmodern in the including document. No thesis files edited.

## Statistics

Bars: arithmetic mean of three independent repetition means. Each repetition
uses ONLY five measured calls (calls 3-7, steps 73, 102, 131, 160, 189); calls
1-2 are warmup. Whiskers: two-sided 95% Student-t confidence interval, df=2,
t=4.30265, half-width = t * sample SD(rep means) / sqrt(3). The 15 calls are
not 15 independent repetitions. Intervals are not clipped at zero.
Timing is the synchronized solver ML-step latency, not whole-job wall time,
isolated model forward time, or a phase decomposition.

The three variants are AIx default layout/collective (reference), PhyDLL default
(blocking), and PhyDLL readiness. Labels: MLP (ov4), Transformer (ov0).
All four placements are retained: 24c1g, 24p24c1g, 96c4g, 96p96c4g.
The plus sign denotes heterogeneous CPU-side + GPU-side solver ranks (48/192
total), using c23g+c23g surrogate nodes, not CPU-only measurements. Workload
aggregates differ across placements: comparisons within a hardware group are
matched; across-group bars must not be interpreted as fixed-work strong scaling.

## Audit

plot_values.csv contains all 24 bar means, sample SDs, CI bounds, repetition means,
cell identities and job IDs. selected_runs.csv preserves all 72 selected source
rows, measured calls, workload sizes, overlap, paths and retry lineage.
manifest.json records SHA-256 and byte counts for the three input files, selection
counts, statistical constants and authored dimensions. The loader rejects missing
or duplicate repetitions, failed runs, wrong overlap, inconsistent per-call
timings, and an invalid seven-call schedule. It never selects fastest runs.
The consolidation report's missing AIx VRAM observation is unrelated to timing;
no memory measurements are plotted here. Chaining coverage is a separate audit
in ../ch7_smartsim_chaining_draft, not part of this readiness renderer.
