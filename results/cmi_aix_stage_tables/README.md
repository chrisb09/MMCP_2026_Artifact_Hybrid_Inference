# AIx Stage Tables

Reproduce with `python scripts/export_cmi_aix_stage_tables.py`. No jobs are submitted;
no figures or thesis files are changed. Each of default, affine, numa has one
eight-row table (two models x four hardware placements), in CSV, Markdown and LaTeX.
LaTeX snippets require booktabs, graphicx and pdflscape; wide tables use landscape.

## Reading the Tables

All durations are in milliseconds. Each row averages three independent
repetition means, each from five measured calls. c1 and full-credit ideal columns
are aliases of ONE COMMON credit-independent prediction, not separate models.
Measured ML-step times are canonical MAIA full-step latencies. Ready-range windows
are measured controller-local spans, NOT five-stage elapsed times or global
makespans. See README.md for definitions, assumptions, and uncertainty.

CSV preserves full precision; presentation formats round to 0.001 ms. Each stage
comes from the SAME collective workgroup selected per call by argmax sum(A).
Thus the five displayed stage means sum to the displayed stage-sum mean (up to
presentation rounding). They are NOT independent per-stage workgroup maxima.
The selected workgroup can change across calls. Full collective ML-step remains
the observed canonical latency, NOT the five-stage sum.

## Exact Model and Adapter

Source: thesis chapter_optimizations.tex lines 274--405, equations tau_step,
tau_max, collective and pipelined (source SHA256 in metadata.json).
N is workgroup membership INCLUDING its controller, never GPU count.
tau_step = sum(tau); tau_max = max(tau);
T_collective = N sum(tau); T_pipeline = sum(tau) + (N-1) max(tau).
The N=10 synthetic example is 380 -> 218 ms.

For each collective controller, A = (1000*gather_collective_s, h2d_gpu_ms,
forward_gpu_ms, d2h_gpu_ms, 1000*scatter_collective_s), tau = A/N.
P_g = sum(A_g)/N_g + (N_g-1)/N_g max(A_g).
B = max_g sum(A_g); P = max_g P_g; R = observed collective ML step - B.
Ideal pipeline = P; ideal full ML step = R + P = observed collective - (B-P).
Before/after maxima are independent ZERO-ORIGIN workgroup envelopes; the critical
workgroup may switch. Never sum GPU savings or substitute stage sum for ML step.
The c1 and full ideal columns deliberately repeat P and R+P: the thesis ideal
formula has NO credit term. They are not credit-specific performance forecasts.

Homogeneous ranks, baseline equivalence, independent stages, unchanged batch
efficiency and non-stage residual, and no inter-workgroup contention are assumed.
Gather/scatter include protocol/waiting/imbalance. GPU stages are CUDA-event sums
over serial collective sub-batches; device_cpu_s is diagnostic, never added.
Negative residuals are retained as scope/instrumentation discrepancies, NOT
literal negative overhead. Negative predictions raise rather than clamp. This
conditional ideal proxy is not a strict bound on observed P2P speedup.

## Measured Pipeline Window

For each P2P controller/call, ready-range window = 1000 *
(last range_inference_end - first range_inference_start). Take the maximum of
these durations across workgroups only AFTER local subtraction, independent of
clock origin. This is a ready-range inference envelope, NOT true five-stage
elapsed time: it omits input readiness before the first range and may omit
communication outside the range events. It is not a synchronized global span.
It includes gaps between ranges. Controller ML-step envelope (max of locally
measured ml_step intervals) is additionally exported in call_metrics.csv and
per-workgroup controller elapsed in workgroup_call_stages.csv; neither replaces
the canonical full MAIA ML-step timing.

## Sampling and Audit

Canonical consolidated snapshot only: 288 runs, 2016 inferences, clean report.
Select 216 successful AIx runs across all layouts, all three repetitions, and
coll/p2p1/p2pfull; MLP overlap 4 and Transformer overlap 0. Provenance hashes and
exact logged workgroup/GPU memberships are validated by the prediction collector.
Diagnostic calls 2..6 match inference calls 3..7 (MAIA steps 73,102,131,160,189).
Compute per-call predictions/maxima BEFORE five-call run means, then average
three independent repetitions. Hardware rows are not fixed-work strong scaling.

summary_statistics.csv supplies all metric means, three run means, sample SD,
and 95% Student t CI (df=2, t=4.302652729911275). Calls and GPUs are not independent
samples. CIs quantify repetition variability, not model uncertainty; negative
CI bounds are retained. call_metrics.csv holds 360 paired collective/P2P calls;
call_predictions.csv preserves per-call before/after identities and residuals;
workgroup_call_stages.csv holds raw/adapted stages, mapping, sources, and P2P local
windows for every WG/call. metadata.json hashes all consumed sources and records
selected canonical runs. No cross-controller absolute timestamps are subtracted.

## Observed Diagnostics

```json
{
  "negative_residuals": 0,
  "residual_min_ms": 15.0025369210598,
  "residual_max_ms": 816.348515339,
  "critical_wg_switches": 46,
  "negative_summary_ci_bounds": 5,
  "max_repetition_cv": 1.4922770212877745,
  "negative_ci_metrics": [
    {
      "layout": "default",
      "model": "mlp",
      "placement": "24p24c1g",
      "metric": "forward_ms",
      "mean_ms": 80.78490861288,
      "sd_ms": 120.55346278983365,
      "ci_low_ms": -218.68649459669834,
      "ci_high_ms": 380.2563118224583
    },
    {
      "layout": "default",
      "model": "mlp",
      "placement": "24p24c1g",
      "metric": "d2h_ms",
      "mean_ms": 92.46926956185334,
      "sd_ms": 39.56213788985088,
      "ci_low_ms": -5.808529131487859,
      "ci_high_ms": 190.74706825519453
    },
    {
      "layout": "default",
      "model": "transformer",
      "placement": "24p24c1g",
      "metric": "d2h_ms",
      "mean_ms": 11.207722600298666,
      "sd_ms": 4.686178237519304,
      "ci_low_ms": -0.43338948394452714,
      "ci_high_ms": 22.84883468454186
    },
    {
      "layout": "affine",
      "model": "transformer",
      "placement": "24p24c1g",
      "metric": "d2h_ms",
      "mean_ms": 10.912567536034667,
      "sd_ms": 4.929657578144719,
      "ci_low_ms": -1.3333807603109715,
      "ci_high_ms": 23.158515832380303
    },
    {
      "layout": "numa",
      "model": "transformer",
      "placement": "24p24c1g",
      "metric": "d2h_ms",
      "mean_ms": 10.905117845542,
      "sd_ms": 4.930317927915351,
      "ci_low_ms": -1.342470850571873,
      "ci_high_ms": 23.152706541655874
    }
  ]
}
```

The maximum coefficient of variation above is sample SD / mean across the
three repetition means, over all exported table metrics. Negative residuals
and wide/negative CIs are reported, not hidden or clamped.
