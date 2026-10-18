# Terrain GPU Thesis Figures: Session Handoff

## Task And Boundaries

Render thesis GPU benchmark figures for the **terrain solver**, following the
visual conventions and figure programme developed for MAIA. This is a rendering
and data-validation task, not permission to submit new benchmark jobs, modify the
thesis, or change measurement selection. Report missing data rather than inventing
bars. Ask before new measurements or profiling campaigns are needed.

The instructions below reflect the user's latest preferences. Prefer them over
older renderers and screenshots, which may still contain superseded layouts.
Keep methodology in Markdown, audit CSVs, and eventual LaTeX captions, **not as
explanatory text inside any version of a graphic**.

## Locations

These paths are absolute so this file can be passed to a session with a different
working directory:

```text
MAIA reference workspace:
/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference

Terrain / SmartSim workspace:
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim

Terrain application:
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app

Terrain C++ interface and original timeline renderers:
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface

Thesis figure guide:
/home/ro092286/Master-Thesis/notes/thesis-figures-guide.md

Shared Matplotlib style:
/home/ro092286/Master-Thesis/gitlab/sourcecode/thesis_style.py

Thesis pipeline model:
/home/ro092286/Master-Thesis/gitlab/thesis/chapters/chapter_optimizations.tex
```

The `/hpcwork/ro092286/smartsim` spelling also occurs in logs. Resolve paths before
deduplicating runs; the two spellings may identify the same storage.

## Read First

Read the figure guide, then inspect these **current MAIA implementations**. Adapt
their presentation and validation logic, not their hard-coded model names,
selected cells, warmup schedule, or data paths.

| Purpose | Reference under the MAIA workspace |
| --- | --- |
| Chapter 6 latency and capacity | `scripts/plot_cmi_ch6.py` |
| Chapter 6 provider-grouped phase columns | `scripts/plot_cmi_ch6_phases.py` |
| Readiness comparison | `scripts/plot_cmi_ch7_readiness.py` |
| Full-step pipeline prediction overview | `scripts/plot_cmi_ch7_pipeline_prediction.py` |
| Preferred paired stage / pipeline-span figure | `scripts/plot_cmi_pipeline_spans.py` |
| Detailed timelines and collapsed phase windows | `scripts/plot_cmi_aix_terrain_style.py` |
| Workgroup exchange-duration balance | `scripts/plot_cmi_aix_exchange_balance.py` |
| Three-layout stage tables | `scripts/export_cmi_aix_stage_tables.py` |
| Explicit selection / provenance example | `scripts/consolidate_cmi_bench.py` |

Read the matching `test_*.py` files and generated READMEs for contracts and scope.
The MAIA Chapter 6 phase set is incomplete because its other-provider GPU phase
captures were not found. **Do not carry that absence assumption over to terrain**:
terrain has separate profile campaigns that need their own provenance checks.

Useful visual references under the MAIA workspace:

```text
figures/ch6_gpu_draft/latency_baselines.png
figures/ch6_gpu_draft/gpu_device_peaks.png
figures/ch6_phases_draft/transformer_96c4g_phases.png
figures/aix_pipeline_spans_draft/transformer_96p96c4g_numa_pipeline_spans.png
figures/aix_pipeline_spans_draft/panels/
figures/aix_terrain_style_draft/README.md
figures/aix_exchange_balance_draft/README.md
```

## Terrain Inputs To Investigate

The following are **discovery starting points**, not an instruction to accept all
rows or combine campaigns automatically. Establish the current authoritative
selection from manifests, successful statuses, workload and configuration first.

| Terrain path relative to `mini_app/` | Known role |
| --- | --- |
| `results/retest_smartsim/summary_combined.csv` | Shared / per-GPU SmartSim retest summaries |
| `results/retest_smartsim/jobids_tpq1_batch.txt` | Retest job provenance |
| `retest_smartsim_cells.zsh` | Deployment and workload settings used by retests |
| `analyze_retest_smartsim.py` | Retest extraction and historical selection rules |
| `gpu_refresh_campaign.py` and `gpu_refresh_README.md` | Refresh campaign definitions and configurations |
| `results/gpu_refresh/gpu_refresh_20261014_thesis_native_gaps/` | Native timing manifests, including chaining c1/c3 |
| `results/gpu_refresh/gpu_refresh_20261013_optimized_profile_retry1/` | Profile retry manifests |
| `results/hh3_w8k/` | Older timeline and pipeline comparison examples |
| `results/retest_plots/` | Existing retest presentation references |

Known terrain model identifiers include `watercnn8k` and `giantmlp288`. Verify
actual model files, resolutions, chunk sizes, samples, and batching in each run.
Do not relabel these as MAIA Transformer / MLP ov0 / ov4.

Previous inspection found native-refresh cases whose nominal `solver_log` paths
did not exist, but whose actual `driver.log` contained the solver timing records.
Resolve and document such receipts explicitly. Do not turn a broken manifest path
into either a silently missing run or an arbitrary directory-search fallback.

The old retest analyzer selected the highest-job-ID sample set for some identities.
Inspect why those replacements were made and materialize an explicit selection
ledger. Do not select the fastest run or use file modification times.

Terrain retests used different thread settings from MAIA. For example, refreshed
terrain SmartSim campaigns used TPQ1 / intra-op8 / inter-op1, whereas the retained
MAIA shared runs used intra-op4 and sometimes TPQ4. Verify terrain settings rather
than copying MAIA settings or mixing older TPQ3 / resolution variants.

## Non-Negotiable Visual Rules

- Author at physical inclusion width: thesis text width is approximately **150 mm**.
- A two-model figure is **one horizontal row**, not two vertically stacked plots.
- Export independent subfigure panels, normally **73.5 mm** wide, with room for a
  roughly 3 mm gap when LaTeX assembles a 150 mm row.
- Standalone wide plots may use 150 mm where 12--16 bars would become unreadable
  at half width. Do not force everything into a half-width slot.
- No repetition dots, scatter overlays, or per-device dots. Timing error bars are
  sufficient. Capacity bars are maxima, not mean estimates with invented CIs.
- Use compact fonts and legends, narrow legend handles, and short meaningful
  labels. Keep a consistent hierarchy rather than shrinking an oversized canvas.
- Remove unused top/bottom space. Measure actual text bounds; do not reserve a
  large fractional title or legend band on every figure.
- Aim for a positive legend-to-plot/header gap of at most about **3 mm** where
  feasible. Do not stretch the plotting axes just to fill removed footer space.
- **No explanatory footers in PNG, PDF, or PGF.** This includes statistics,
  caveats, clock assumptions, prediction-model notes, sampling counts, and text
  such as "not a global makespan". Put those in README/caption material instead.
- **No baked-in `(a)` / `(b)` headings or subcaptions** in thesis panels. Let LaTeX
  own those. Pipeline-span exports are fully caption-free in all formats.
- Combined model overviews may retain concise model/dataset identification;
  independent panels should leave that identification to LaTeX subcaptions.
- Keep real missing-data markers and essential axis semantics. "No phase data"
  is not an explanatory footer and is not a zero-duration measurement.
- Use ordinary reader-facing axis labels. Prefer **Inference time (s)** to
  "Envelope (s)"; document whether transfers/protocol are included in the caption.
- For pipeline-span bars, print the mean above the upper error-bar cap, or below
  the lower cap if necessary. Three significant figures is the current convention.

Current useful dimensions, not universal requirements:

| Figure | Authored dimensions |
| --- | --- |
| Chapter 6 two-model latency / VRAM overview | 150 x 55 mm; plotting axes 32 mm high |
| Paired five-stage / pipeline-span figure | 150 x 33.592 mm; plotting axes 20.592 mm high |
| Independent pipeline stage or comparison panel | 73.5 x 33.592 mm |

Inspect at actual thesis size. A 192-rank trace cannot make every rank label
legible on one ordinary thesis page; provide per-WG detail or zoomable PDFs rather
than pretending an unreadable reduction is presentation-ready.

## Colors And Typography

Use `thesis_style.apply()` before rendering and `thesis_style.save()` for PGF,
self-contained PDF, and PDF-derived 300 dpi PNG. It uses pdflatex / Latin Modern.
Read the guide for the required preamble. Avoid Type 3 fonts.

Do not use tight-bounding-box export for physically authored thesis bar figures:
it changes their intended inclusion width. Adjust the interior layout instead.

| Meaning | Color |
| --- | --- |
| AIx baseline / collective sum | `#577787` |
| PhyDLL baseline | `#C97AA7` |
| SmartSim shared baseline | `#8064A2` |
| Predicted pipeline | `#C99449`, with `///` hatch |
| Measured P2P exchange | `#568775` |
| Gather / input phase | `#4C78A8` |
| H2D | `#72B7B2` |
| Forward | `#F58518` |
| D2H | `#54A24B` |
| Scatter / output phase | `#B279A2` |

Choose and document a distinct, consistent fourth deployment color for SmartSim
per-GPU. Do not make different measurement scopes look equivalent merely because
they share a color.

For the compact pipeline figures, current labels are 9 pt, y ticks 8 pt, x ticks
6.5--7 pt, legends 5.5 pt, and numeric annotations 5.5 pt. Other figures can use
larger legends where space permits. These are authored sizes, not sizes after
scaling a huge image down to the thesis width.

## Data And Statistical Contracts

1. Identify model, hardware, provider, deployment, layout, communication mode,
   repetition, job ID, workload and software/configuration provenance explicitly.
2. Determine the warmup and measured calls from the **terrain campaign**. Do not
   import MAIA's seven-call schedule or its two warmups. Some inspected native
   terrain cases had 101 calls with one warmup for WaterCNN and six calls with one
   warmup for GiantMLP; profile campaigns can differ again.
3. Compute a measured-call mean within each independent repetition, then the mean
   and CI across repetition means. Calls, ranks and GPUs are not independent runs.
4. Use a 95% Student-t CI with sample SD (`ddof=1`) and the actual repetition count.
   For three repetitions, `t = 4.302652729911275`, df=2; do not reuse that multiplier
   if the terrain dataset has a different number of repetitions.
5. Preserve signed CI bounds in the audit. Do not silently clamp inconvenient
   negative lower bounds or conceal large repetition variability.
6. Capacity plots show maximum observed whole-node RAM or per-device VRAM, not
   summed independent peaks. Keep solver-side and inference-side hosts separate.
7. MiB / 1024 = GiB. Whole-node used RAM is not process/cgroup RSS. Device-wide
   VRAM is not process allocation. A sampled peak is not a guaranteed instantaneous
   allocation maximum. Do not silently subtract background usage.
8. Missing data is blank / explicitly unavailable, never zero. A repeat can supply
   a separately tagged capacity observation, but cannot retrospectively measure
   an old run or silently replace its timing data.
9. Match batch sizes, model/input shapes, TPQ, intra/inter-op threads, chaining,
   workload, and deployment before attributing a difference to one optimization.
10. These placements are not automatically fixed-work strong scaling. Use logged
    aggregate counts and actual total solver ranks rather than rank-zero counts.

Typical placement names used by these campaigns are `24c1g`, `24p24c1g`, `96c4g`,
and `96p96c4g`. The `p` forms can mean 24+24 and 96+96 solver ranks. Verify actual
allocated node types and whether CPU-side GPUs were hidden in surrogate runs.

## Figure Programme

### Chapter 6: Baselines And Deployment

Use separate content folders and figures for latency, memory, and phases.

Compare AIx **default / collective**, PhyDLL **default / blocking**, SmartSim
**shared DB**, and SmartSim **DB-per-GPU** where measured. DB-per-GPU is a baseline
deployment alternative, **not a Chapter 7 optimization**. On one GPU, the two DB
topologies coincide; do not duplicate one observation as an independent result.

For latency and VRAM overviews, put the two terrain models side by side, with
hardware-grouped provider/deployment columns within each model panel. Also export
caption-free independent panels. Do not mix latency and memory into phase figures.

For phases, render one figure per model/hardware combination, following the middle
panel of the CPU comparison examples:

```text
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/figures/cpu_final/main/
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/plot_cpu_comparison.py
```

That phase panel uses **adjacent phase columns within provider groups**, not a
stacked provider-total bar. Extend AIx to its five GPU-relevant stages. Use a proper
phase legend and show native PhyDLL / SmartSim phases when supported by verified
captures. Do not infer phases by multiplying total latency by guessed fractions.

The CPU phase names are provider-specific: PhyDLL input receive/frame-copy,
forward, output send, and other measured work; SmartSim put, model request,
retrieval/unpack, and chunk planning. A synchronous model-request interval is not
automatically CUDA forward time. Check nested regions before adding them together.

### Chapter 7: Readiness And Chaining

Keep these separate from the Chapter 6 provider/deployment figures:

- PhyDLL readiness versus blocking, with AIx default/collective as an optional
  reference, matching the CPU readiness presentation.
- SmartSim chaining c0/c1/c3 by model and hardware, under matched DB/thread/batch
  settings. Do not confuse **SmartSim chaining c1** with **AIx P2P Credit=1**.
- AIx layout/pipeline comparisons. Do not crowd all layouts and all providers into
  one omnibus plot.

### Chapter 7: Preferred Paired Pipeline Figure

Produce one figure per model and hardware, optionally repeated for each of
default, affine, and NUMA layouts. Match collective and P2P layouts exactly.

**Left panel:** five adjacent collective phase columns, each with its own mean,
CI, phase color, and numeric value. Labels: Gather, H2D, Fwd, D2H, Scatter.

**Right panel:** five bars arranged as one reference and two pairs:

```text
Collective 5-stage sum     P2P Credit=1          P2P Fullcredit
                          Predicted Measured    Predicted Measured
```

The individual Predicted/Measured x labels are omitted in the actual graphic;
color/hatching and the legend distinguish them. The group labels remain.

Both legends occupy the same compact row: phase legend left-aligned, comparison
legend right-aligned. Comparison legend entries are exactly **Collective 5-stage
sum**, **Predicted**, **Measured**. Do not append `(ideal)` to the legend.

Use independent y scales: **Phase time (s)** and **Inference time (s)**. No title,
subcaption, or methodological footer is baked into any export. Export the combined
150 mm figure and two independent 73.5 mm caption-free panels.

The additional 16-bar-per-model full-ML-step prediction overview remains useful:
four hardware groups, each containing collective, predicted, P2P Credit=1, and
P2P Fullcredit. Export a horizontal two-model preview plus readable independent
model figures. **Do not mix its full-step prediction with the pipeline-only
prediction used in the preferred paired figure.**

## Prediction Definitions

Read the thesis equations and definitions around `chapter_optimizations.tex`
lines 274--405. Verify current source rather than treating line numbers as fixed.
For N ranks in one workgroup, including the controller:

```text
tau_step = sum(tau_send_input, tau_H2D, tau_forward, tau_D2H, tau_send_output)
tau_max = max(the same five stages)
T_collective = N * tau_step
T_pipeline = tau_step + (N - 1) * tau_max
savings = (N - 1) * (tau_step - tau_max)
```

The synthetic example with N=10 and aggregate stages `[50,30,200,20,80]` ms gives
collective 380 ms and pipeline 218 ms. Include this as a unit test.

The current collective-data adapter uses:

```text
A_g = [gather_host_ms, H2D_CUDA_ms, forward_CUDA_ms, D2H_CUDA_ms, scatter_host_ms]
tau_g = A_g / N_g
B_g = sum(A_g)
P_g = B_g / N_g + (N_g - 1) / N_g * max(A_g)
B = max_g B_g
P = max_g P_g
```

Use logged workgroup membership for N. Never substitute total job ranks or GPU
count. Before/after maximizing workgroups can differ. For the displayed five
collective stages, retain the **entire vector from the single WG maximizing B_g**
for that call. Do not independently maximize each stage over different WGs.

The adapter is an amortized homogeneous-rank model, not measured per-rank service
times. It assumes independent stages, unchanged batching efficiency and no
inter-WG contention. It has **no credit-count term**: the Credit=1 and Fullcredit
predictions repeat the same P. Do not invent different predictions for them.

For the paired pipeline-only figure, use B as collective reference and P as each
predicted bar. For the separate full-step overview:

```text
predicted_full_ML_step = observed_collective_full_ML_step - (B - P)
```

Never add savings across GPUs. Never replace an observed full ML-step by a stage
sum. Maxima here assume independent WG zeros; they are not a measured global
critical path. Compute per-call quantities first, then repetition means and CIs.

`device_cpu_s` includes GPU activity and must **not** be added to H2D+forward+D2H.
Keep native host gather/device/scatter and mixed host/CUDA five-stage accounting
distinct. Device-host duration minus CUDA sums can be negative; preserve that
diagnostic and do not invent an exclusive "Other" overhead phase from it.

## Measured Pipeline Span

Prefer a same-controller-clock span covering input, inference, and output. In the
current AIx event schema the MAIA reference uses:

```text
input_listen_start -> last result_send_complete
```

Validate that the terrain data/source revision has these events and the same
semantics. The end marker is after output `MPI_Waitall`, not an individual worker
receipt. Validate all input/output peers and exactly-once ready-range coverage.
Subtract endpoints on each controller's own clock, then take the maximum local
duration across WGs. Missing endpoints should fail explicitly, not silently fall
back to a different interval.

This interval includes setup, scheduling, waiting and completion protocol. It is
not pure CUDA forward, a ready-range-only window, the full solver ML-step, or a
synchronized global first-worker-send to last-worker-receipt span. The caption
must explain the scope despite the concise reader-facing Inference time axis.

## Timelines And Collapsed Windows

The original terrain renderers are:

```text
../CPP-ML-Interface/scripts/plot_p2p_timeline.py
../CPP-ML-Interface/scripts/analyze_p2p_inference_phases.py
```

These paths are relative to `mini_app/`. The detailed drawing is `render_step()`
in the former; the original collapsed view is `render_phase_bars()` in the latter.
Reusing the actual detailed renderer is preferred over inventing a different
visual language. The current MAIA adapter demonstrates narrow mapping/style
overrides without editing the original renderer.

Historical examples:

```text
mini_app/results/hh3_w8k/s5_aix_p2p_numa/p2p_timeline_step_010_vs_coll.png
mini_app/results/hh3_w8k/s6_phase_bars_coll_vs_s6_aix_p2p_fullcredits_numa.png
```

**Do not blindly reuse the old analysis assumptions.** Known issues include:

- Modulo-based fallback WG membership is wrong for arbitrary affine/NUMA layouts.
- Collective WGs can be keyed by ordinal while P2P groups use controller world
  ranks; intersecting those keys drops groups or matches the wrong ones.
- The wrapper can pick one NUMA collective reference for multiple layouts.
- Step/warmup offsets, titles and model names can be hard-coded.
- Collective overlays and timelines can use different zero origins.
- The old worker-clock correction equates send/ACK completion with controller
  input-ready logging. Those are different causal events, not a clock calibration.

Parse actual memberships and GPU/controller maps from logs. Match all WGs by
verified memberships / stable WG identities, never only intersect controller ranks.

MPI_Wtime origins can differ between processes even on one node. An additive
worker offset based on the old ACK equality can produce negative starts, input
credit-order violations and negative controller-send to worker-receipt intervals.
Do not clamp, silently omit, or claim these prove a real communication-order bug.
Export offsets, raw endpoints and causal feasibility bounds; retain visible
invalid-pair markers. Local-zero alternatives are useful, but do not show global
cross-rank concurrency. Keep essential independent-zero / estimated-time axis
labels; longer qualifications belong in documentation and captions.

Detailed presentation requirements:

- All worker sends, credit waits and output transfers, plus a concise controller
  lane with ready ranges and CUDA operations.
- Compact rank spacing, same small font for controller/worker rank labels,
  smaller credit triangles, and no excessive legend band.
- Ready-range labels about 4.5 pt; detect overlap with rendered bounding boxes,
  stagger colliding labels, and draw faint leaders to their actual range starts.
- Preserve legibility around controller annotations; do not overlap adjacent
  worker rows merely to reduce height.

Collapsed presentation requirements:

- One input, inference, and output window per WG, each first-start to last-end,
  **including gaps**, not sums of request durations.
- Input spans worker input sends, inference spans ready-range processing, output
  spans controller result posting through worker receipt when alignment permits.
- For multiple GPUs, use **one enclosing axes with horizontal WG dividers** and
  a common scale, not four separate subplot boxes with repeated margins.
- Clearly distinguish independent controller zeros from globally aligned time.
- Collective reference labels are **Collective gather end**, **Collective
  inference end**, and **Collective scatter end**, not Coll or g/d/s.
- If collective input has durations only, dashed boundaries are independently
  zeroed, concatenated schematic references with gaps omitted. If verified OTF2
  entry/exit timestamps exist, retain their actual scope/origin instead.

The original analyzer expects OTF2 regions `solver_step_ml_steady`,
`gatherInputData`, `inferenceDevice`, `scatterOutputData`. Check tick resolution,
location-to-rank mapping, call identity, and warmups. Do not accept its silent
2.1 GHz fallback without verifying the trace frequency.

## Tables And Workgroup Balance

Export one stage/prediction table per default, affine and NUMA layout, with rows
for every terrain model/hardware combination. Include five stages, their sum,
observed full ML-step, pipeline-only prediction, full-step prediction, measured
Credit=1/Fullcredit values, and exact definitions of any measured windows.
Duplicated credit-independent predictions must be documented, not presented as
separate credit-aware models. Retain per-call and per-WG audit rows.

If reproducing the workgroup-balance figure, use the three-panel presentation:
WG-by-WG paired collective/P2P exchange durations; an exchange-duration balance
score; and maximum WG proxy versus full ML-step. For four WGs:

```text
D_g = max rank-local exchange duration among ALL members of WG g
balance = mean(D_g) / max(D_g)
```

Do not call it computational load balance. Equal elapsed durations do not imply
equal work. Collective stage-sum and continuous P2P exchange scopes differ; neither
is automatically a globally synchronized WG makespan.

## Suggested Output Structure

Use a new terrain-owned output root rather than overwriting historical galleries:

```text
mini_app/results/thesis_gpu_figures/
  ch6_baselines/
  ch6_phases/
  ch7_readiness/
  ch7_smartsim_chaining/
  ch7_pipeline_prediction/
  aix_pipeline_spans/panels/
  aix_timelines/
  aix_exchange_balance/
  tables/
```

Each folder should have a gallery README, exact plotted values, run-level and
call-level inputs where relevant, explicit source paths/job IDs, missing-data
coverage, and SHA256 provenance. A provenance record must distinguish raw timing,
profile-derived phase data and supplemental capacity repeats. Preserve existing
measurements and keep new scripts in the terrain workspace.

## Verification Before Handoff

1. Recompute means and CIs independently from repetition means.
2. Verify measured-call/warmup selection for each terrain campaign.
3. Verify all WG/rank/GPU mappings, all phases, and call identities before rendering.
4. Check three-run CI math, no dots, no inferred missing bars, and no GPU-count
   multiplication of durations or savings.
5. Test the 380 -> 218 ms synthetic pipeline case and critical-WG switching.
6. Assert no explanatory figure text, no subcaption headings, and no standalone
   embedded titles in caption-free panels; permit only genuine numeric annotations.
7. Check legend and label bounds on an actual rendered canvas, including both
   compact and high-variance examples. Numeric labels must clear CI caps.
8. Verify physical PDF/PGF dimensions and embedded fonts; inspect PNGs at authored
   page width. Check for raster companions needed by PGF.
9. Hash numeric exports before and after layout-only edits; confirm unchanged data.
10. Report unavailable profiles/configurations and scope limitations honestly in
    the README and session response, not as paragraph-sized text in the images.

Do not submit, cancel, or modify existing terrain refresh, profiling, CPU, monitor,
notification, or MAIA per-GPU jobs as a side effect of this rendering task.

## Suggested First Message To The Next Session

> Use this file as the rendering specification. Inspect the current MAIA example
> image and terrain manifests first. Establish an explicit successful-run selection
> for the two terrain models and four placements. Then implement the compact,
> caption-free paired pipeline figures and Chapter 6 baseline/provider-phase
> columns, preserving source scopes and statistics. Reuse the current style and
> original terrain detailed renderer where appropriate, but fix its known mapping
> and alignment assumptions. Export PNG/PDF/PGF, independent LaTeX panels, exact
> values and provenance. Do not launch jobs or modify the thesis. Report data gaps
> instead of substituting MAIA, CPU, or unmatched historical measurements.
