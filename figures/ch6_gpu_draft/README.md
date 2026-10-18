# Chapter 6 GPU Drafts

Baseline comparisons authored at thesis page width (150 mm).

## Figures

| Draft | Preview | Vector |
| --- | --- | --- |
| Combined latency | [PNG](latency_baselines.png) | [PDF](latency_baselines.pdf) |
| Transformer latency | [PNG](transformer_latency.png) | [PDF](transformer_latency.pdf) |
| MLP latency | [PNG](mlp_latency.png) | [PDF](mlp_latency.pdf) |
| Transformer memory | [PNG](transformer_memory.png) | [PDF](transformer_memory.pdf) |
| MLP memory | [PNG](mlp_memory.png) | [PDF](mlp_memory.pdf) |
| Per-GPU observed maxima | [PNG](gpu_device_peaks.png) | [PDF](gpu_device_peaks.pdf) |

Every table entry also has a same-stem PGF export. Individual memory panels are available as `{transformer,mlp}_{cpu_ram,ml_host_ram,vram}.{png,pdf,pgf}`. The 12 original figures remain 150 mm wide: standalones are 54 mm high, two-model plates are one horizontal row (1x2), 55 mm high, and three-scope single-model memory plates remain stacked, 149 mm high. Paired model slots are 73.5 mm wide with a 3 mm gap. Eight additional `{transformer,mlp}_{latency,cpu_ram,ml_host_ram,vram}_halfwidth.{png,pdf,pgf}` exports are authored at 73.5 x 64 mm, with a vertical legend, for independent subfigures. Each plotting axis is 32 mm high, roughly half the previous physical y-axis height. Include at authored width, without further shrinking. Overview model labels identify Transformer (ov0) and MLP (ov4). Independent panels have no embedded titles or subcaptions in any format; supply model/overlap and memory scope in LaTeX subcaptions. cpu_ram is shared/CPU-side RAM, ml_host_ram is GPU-side ML-host RAM, and vram is maximum per-GPU VRAM. Statistical and coverage explanations are caption/README-only.

## Font Style

All formats use thesis_style's pdflatex/Latin Modern typography. Axis labels are 9 pt, ticks 8 pt, legends 7.5 pt and latency values 6.5 pt; overview headers are 9--10 pt. Removing explanatory footers and independent titles trims canvas margins only, without enlarging fonts or stretching the 32 mm plotting axes. Paired overviews use an 11 mm top band and 3 pt model-header padding; measured legend/header bounds must not overlap and their gap must be at most 3 mm. The paired VRAM overview identifies maximum per-GPU VRAM in its y-axis labels; the metric remains device-wide. Narrow three-row legends retain their existing 20 mm top band.

## Methodology

Baseline only: AIx default/coll, PhyDLL default, SmartSim sharedDB; 72 runs, 24 latency bars. Hardware groups are placements, not a fixed-work strong-scaling experiment.

Mean of 3 repetition means (5 steady calls each).
Error bars: 95% Student-t CI, df=2. SD uses ddof=1; CI = mean +/- 4.302652729911275 * SD / sqrt(3). The 15 calls per bar are NOT 15 independent observations. Negative CI bounds are retained and warned.

Bars: maximum observed peak, not sums.
No CI for capacity metrics; no imputation. RAM is whole-node usage, not CPU-process RSS; GPU-side RAM is the separate ML host in heterogeneous placements and N/A for shared-host placements. VRAM is device-wide usage, maximized across expected ML-host devices within each repetition, then across repetitions, never summed. Background usage is not subtracted. No missing observations are zero-filled. AIx Transformer 24c/1GPU repetition 2 VRAM is missing (aix-24c1g-default-coll-transformer-r2, job 5009522): historically n=2 original observations; the original rep2_gib remains blank in the audit. A validated supplemental repeat, job 5106286, now contributes only its sampled device-wide peak to the maximum, marked with an availability '*'. That bar has 2 original observations + 1 supplemental repeat, not three original repetitions. effective_rep2_gib is plot-derived, not a canonical replacement. device_peaks.csv retains the explicit original missing row plus a separate supplemental_repeat row with source_jobid/measurement_jobid/original_jobid provenance, and excludes incidental solver-host GPUs. Raw device/repetition values remain in the CSV only; there are no dots or scatter overlays in any timing or memory figure. Observed maxima are capacity summaries, not estimators of a mean: no misleading CI or range whisker is attached to them.

The heterogeneous surrogate and variable workload require caution: solver rank counts, overlap, aggregate samples and batch sizes vary across placements/models/providers. See metadata.json workloads; these are not workload-normalized speedups or application-level performance claims.

## Coverage And Confounds

The 12 SmartSim DB-per-GPU measurements for 96c4g and 96p96c4g are NOT LOCATED (2 models x 2 four-GPU placements x 3 repetitions). This is missing measurement coverage, not a plot-code omission: the current 288-run matrix contains only shared c0 SmartSim. No DB-per-GPU bars are depicted or fabricated. Available per-device memory peaks are not perGPU timing evidence. No cross-library GPU Score-P phase dataset is available; no phase plots are fabricated.

All selected shared SmartSim runs use DB=1, intra=4, inter=1. Transformer 24c1g and 96c4g use TPQ=4, while heterogeneous 24p24c1g and 96p96c4g use TPQ=1; MLP uses TPQ=1 throughout. This TPQ confound prevents attributing shared/heterogeneous Transformer differences solely to placement.

Canonical inputs are runs.csv, resources.csv and report.json in the consolidated snapshot. If present, the fixed results/cmi_bench_telemetry_retry_validation.json explicitly designates job 5106286; raw artifact and ledger hashes, original identity, configuration, seven steps, workload and expected device scope are validated before aggregation. There is no latest/fastest discovery and no automatic canonical replacement. Timing and all other memory observations are unchanged. SHA256 hashes and counts are in metadata.json. CSV audit units are seconds and GiB. PNG is rasterized from the PDF at 300 dpi; PDF/PGF use the thesis_style pdflatex/Latin Modern pipeline without tight bounding-box resizing. No thesis LaTeX is modified.

The linear MLP axis preserves large variability but compresses fast AIx bars. Memory panels keep host roles separate; raw imbalance can be audited in device_peaks.csv, but is not shown by maximum-only bars.
