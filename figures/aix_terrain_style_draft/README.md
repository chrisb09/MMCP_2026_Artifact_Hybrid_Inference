# AIx Terrain Style Draft

Illustrative ACK-anchored alignment, NOT measured synchronization. Each WG independently zeros its controller ml_step_start; no cross-WG concurrency interpretation.

## Reproduce

```sh
python scripts/plot_cmi_aix_terrain_style.py
```

Narrative notes and statistics are README/caption-only in every graphic version. Axes retain independent-zero and estimated-alignment semantics; red crosses retain invalid receipt flags. Saved canvases crop unused bottom space without stretching axes.

Output: 32 timeline pairs (envelope/detail), 64 PNG and 64 PDF files; no PGF. 2800 workers; 44 invalid ACK-aligned output pairs; 2800 feasible tightened causal intervals.

## Semantics

Worker time is shifted by c_input_ready - w_input_send_complete using the original
synchronize_step_clocks with the exact solver-log membership, then every event in each WG
subtracts that controller's ml_step_start. Equality ignores ACK/communication delay and is
not trustworthy globally. alignment.csv records additive worker offsets and causal bounds
(controller time = worker time + offset); bounds are evidence, not error bars or synchronization.
Some aligned worker receipts precede controller sends: signed output spans and validity flags
are exported, never clamped. Invalid output pairs are marked by red crosses in detail plots
at their aligned receipt times; the original renderer cannot draw their negative bars.
Envelope widths themselves are validated nonnegative; an envelope can conceal invalid pair order.
Negative worker starts are retained, without clipping. Approximate clock alignment can shift
workers before controller ml_step_start; these logs cannot diagnose early arrival versus an
alignment artifact. Negative coordinates do not establish physically early worker arrival.
Detailed figures use the original render_step directly,
including controller lanes and worker send/receive display. Its final earliest-event zero
is recorded and applied to all collective references; tick labels undo that display shift
so zero explicitly denotes each WG's own controller start. Original global start/end lines are
only extrema across independently zeroed WGs, not a global measured interval.

Phase envelopes are windows INCLUDING gaps, not sums: input = earliest worker input_send_start
to latest worker input_send_complete (equal to latest worker-targeted input_ready under the
anchor); inference = first range_inference_start to last range_inference_end, a CPU-host
ready-range window, NOT torch_forward-only; output = first worker-targeted controller
result_send_start to last worker result_received. Controller self-copy is excluded from
input/output; all controller ready-range inference is included. CUDA phases in detail retain
their original host anchors; inter-request GPU overlap is approximate.

Dashed references are collective controller durations only: g = gather_collective_ms,
d = device_cpu_ms, s = scatter_collective_ms. Ends are g, g+d, g+d+s, independently
zeroed and concatenated schematically; arrival waits and gaps are omitted. These are NOT
observed collective timestamps or globally synchronized clocks. No collective bars are drawn.
Comparisons match collective/P2P by logged WG ID, never by controller-rank intersection.
One selected measured call is shown, not repetition averages. World memberships come from
base.parse_mapping solver logs; local peer IDs come from actual rank CSVs, never modulo.


All four placements get envelopes and details by default: 24c1g, 24p24c1g, 96c4g, 96p96c4g (32 comparisons, 32 envelopes and 32 details, each PNG/PDF). Use --placement (alias --placements) to select placements. Details retain original lanes at save-time height min(original height / 2, 1.5 + 0.12 inches per rank), with uniform 5 pt rank labels. Ready-range labels are 4.5 pt, allocated by rendered bounding boxes into staggered rows in a reserved controller band with faint leaders. Legend space uses its actual bounding box. At up to 192 ranks, view the vector PDF at full page height to read individual ranks; a thesis-width reduction cannot preserve every label.

[Alignment](alignment.csv) | [Causality audit](output_causality_report.md) | [Phase envelopes](phase_envelopes.csv) | [Metrics](workgroup_metrics.csv) | [Source hashes](metadata.json)

## Gallery

### transformer_24c1g_default_p2p1_r1_call6_phase_envelopes

[PDF](transformer_24c1g_default_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_24c1g_default_p2p1_r1_call6_phase_envelopes](transformer_24c1g_default_p2p1_r1_call6_phase_envelopes.png)

### transformer_24c1g_default_p2p1_r1_call6_terrain_detail

[PDF](transformer_24c1g_default_p2p1_r1_call6_terrain_detail.pdf)
![transformer_24c1g_default_p2p1_r1_call6_terrain_detail](transformer_24c1g_default_p2p1_r1_call6_terrain_detail.png)

### transformer_24c1g_default_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_24c1g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_24c1g_default_p2pfull_r1_call6_phase_envelopes](transformer_24c1g_default_p2pfull_r1_call6_phase_envelopes.png)

### transformer_24c1g_default_p2pfull_r1_call6_terrain_detail

[PDF](transformer_24c1g_default_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_24c1g_default_p2pfull_r1_call6_terrain_detail](transformer_24c1g_default_p2pfull_r1_call6_terrain_detail.png)

### transformer_24c1g_numa_p2p1_r1_call6_phase_envelopes

[PDF](transformer_24c1g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_24c1g_numa_p2p1_r1_call6_phase_envelopes](transformer_24c1g_numa_p2p1_r1_call6_phase_envelopes.png)

### transformer_24c1g_numa_p2p1_r1_call6_terrain_detail

[PDF](transformer_24c1g_numa_p2p1_r1_call6_terrain_detail.pdf)
![transformer_24c1g_numa_p2p1_r1_call6_terrain_detail](transformer_24c1g_numa_p2p1_r1_call6_terrain_detail.png)

### transformer_24c1g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_24c1g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_24c1g_numa_p2pfull_r1_call6_phase_envelopes](transformer_24c1g_numa_p2pfull_r1_call6_phase_envelopes.png)

### transformer_24c1g_numa_p2pfull_r1_call6_terrain_detail

[PDF](transformer_24c1g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_24c1g_numa_p2pfull_r1_call6_terrain_detail](transformer_24c1g_numa_p2pfull_r1_call6_terrain_detail.png)

### transformer_24p24c1g_default_p2p1_r1_call6_phase_envelopes

[PDF](transformer_24p24c1g_default_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_24p24c1g_default_p2p1_r1_call6_phase_envelopes](transformer_24p24c1g_default_p2p1_r1_call6_phase_envelopes.png)

### transformer_24p24c1g_default_p2p1_r1_call6_terrain_detail

[PDF](transformer_24p24c1g_default_p2p1_r1_call6_terrain_detail.pdf)
![transformer_24p24c1g_default_p2p1_r1_call6_terrain_detail](transformer_24p24c1g_default_p2p1_r1_call6_terrain_detail.png)

### transformer_24p24c1g_default_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_24p24c1g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_24p24c1g_default_p2pfull_r1_call6_phase_envelopes](transformer_24p24c1g_default_p2pfull_r1_call6_phase_envelopes.png)

### transformer_24p24c1g_default_p2pfull_r1_call6_terrain_detail

[PDF](transformer_24p24c1g_default_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_24p24c1g_default_p2pfull_r1_call6_terrain_detail](transformer_24p24c1g_default_p2pfull_r1_call6_terrain_detail.png)

### transformer_24p24c1g_numa_p2p1_r1_call6_phase_envelopes

[PDF](transformer_24p24c1g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_24p24c1g_numa_p2p1_r1_call6_phase_envelopes](transformer_24p24c1g_numa_p2p1_r1_call6_phase_envelopes.png)

### transformer_24p24c1g_numa_p2p1_r1_call6_terrain_detail

[PDF](transformer_24p24c1g_numa_p2p1_r1_call6_terrain_detail.pdf)
![transformer_24p24c1g_numa_p2p1_r1_call6_terrain_detail](transformer_24p24c1g_numa_p2p1_r1_call6_terrain_detail.png)

### transformer_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes](transformer_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes.png)

### transformer_24p24c1g_numa_p2pfull_r1_call6_terrain_detail

[PDF](transformer_24p24c1g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_24p24c1g_numa_p2pfull_r1_call6_terrain_detail](transformer_24p24c1g_numa_p2pfull_r1_call6_terrain_detail.png)

### transformer_96c4g_default_p2p1_r1_call6_phase_envelopes

[PDF](transformer_96c4g_default_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_96c4g_default_p2p1_r1_call6_phase_envelopes](transformer_96c4g_default_p2p1_r1_call6_phase_envelopes.png)

### transformer_96c4g_default_p2p1_r1_call6_terrain_detail

[PDF](transformer_96c4g_default_p2p1_r1_call6_terrain_detail.pdf)
![transformer_96c4g_default_p2p1_r1_call6_terrain_detail](transformer_96c4g_default_p2p1_r1_call6_terrain_detail.png)

### transformer_96c4g_default_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_96c4g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_96c4g_default_p2pfull_r1_call6_phase_envelopes](transformer_96c4g_default_p2pfull_r1_call6_phase_envelopes.png)

### transformer_96c4g_default_p2pfull_r1_call6_terrain_detail

[PDF](transformer_96c4g_default_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_96c4g_default_p2pfull_r1_call6_terrain_detail](transformer_96c4g_default_p2pfull_r1_call6_terrain_detail.png)

### transformer_96c4g_numa_p2p1_r1_call6_phase_envelopes

[PDF](transformer_96c4g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_96c4g_numa_p2p1_r1_call6_phase_envelopes](transformer_96c4g_numa_p2p1_r1_call6_phase_envelopes.png)

### transformer_96c4g_numa_p2p1_r1_call6_terrain_detail

[PDF](transformer_96c4g_numa_p2p1_r1_call6_terrain_detail.pdf)
![transformer_96c4g_numa_p2p1_r1_call6_terrain_detail](transformer_96c4g_numa_p2p1_r1_call6_terrain_detail.png)

### transformer_96c4g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_96c4g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_96c4g_numa_p2pfull_r1_call6_phase_envelopes](transformer_96c4g_numa_p2pfull_r1_call6_phase_envelopes.png)

### transformer_96c4g_numa_p2pfull_r1_call6_terrain_detail

[PDF](transformer_96c4g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_96c4g_numa_p2pfull_r1_call6_terrain_detail](transformer_96c4g_numa_p2pfull_r1_call6_terrain_detail.png)

### transformer_96p96c4g_default_p2p1_r1_call6_phase_envelopes

[PDF](transformer_96p96c4g_default_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_96p96c4g_default_p2p1_r1_call6_phase_envelopes](transformer_96p96c4g_default_p2p1_r1_call6_phase_envelopes.png)

### transformer_96p96c4g_default_p2p1_r1_call6_terrain_detail

[PDF](transformer_96p96c4g_default_p2p1_r1_call6_terrain_detail.pdf)
![transformer_96p96c4g_default_p2p1_r1_call6_terrain_detail](transformer_96p96c4g_default_p2p1_r1_call6_terrain_detail.png)

### transformer_96p96c4g_default_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_96p96c4g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_96p96c4g_default_p2pfull_r1_call6_phase_envelopes](transformer_96p96c4g_default_p2pfull_r1_call6_phase_envelopes.png)

### transformer_96p96c4g_default_p2pfull_r1_call6_terrain_detail

[PDF](transformer_96p96c4g_default_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_96p96c4g_default_p2pfull_r1_call6_terrain_detail](transformer_96p96c4g_default_p2pfull_r1_call6_terrain_detail.png)

### transformer_96p96c4g_numa_p2p1_r1_call6_phase_envelopes

[PDF](transformer_96p96c4g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![transformer_96p96c4g_numa_p2p1_r1_call6_phase_envelopes](transformer_96p96c4g_numa_p2p1_r1_call6_phase_envelopes.png)

### transformer_96p96c4g_numa_p2p1_r1_call6_terrain_detail

[PDF](transformer_96p96c4g_numa_p2p1_r1_call6_terrain_detail.pdf)
![transformer_96p96c4g_numa_p2p1_r1_call6_terrain_detail](transformer_96p96c4g_numa_p2p1_r1_call6_terrain_detail.png)

### transformer_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](transformer_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![transformer_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes](transformer_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes.png)

### transformer_96p96c4g_numa_p2pfull_r1_call6_terrain_detail

[PDF](transformer_96p96c4g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![transformer_96p96c4g_numa_p2pfull_r1_call6_terrain_detail](transformer_96p96c4g_numa_p2pfull_r1_call6_terrain_detail.png)

### mlp_24c1g_default_p2p1_r1_call6_phase_envelopes

[PDF](mlp_24c1g_default_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_24c1g_default_p2p1_r1_call6_phase_envelopes](mlp_24c1g_default_p2p1_r1_call6_phase_envelopes.png)

### mlp_24c1g_default_p2p1_r1_call6_terrain_detail

[PDF](mlp_24c1g_default_p2p1_r1_call6_terrain_detail.pdf)
![mlp_24c1g_default_p2p1_r1_call6_terrain_detail](mlp_24c1g_default_p2p1_r1_call6_terrain_detail.png)

### mlp_24c1g_default_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_24c1g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_24c1g_default_p2pfull_r1_call6_phase_envelopes](mlp_24c1g_default_p2pfull_r1_call6_phase_envelopes.png)

### mlp_24c1g_default_p2pfull_r1_call6_terrain_detail

[PDF](mlp_24c1g_default_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_24c1g_default_p2pfull_r1_call6_terrain_detail](mlp_24c1g_default_p2pfull_r1_call6_terrain_detail.png)

### mlp_24c1g_numa_p2p1_r1_call6_phase_envelopes

[PDF](mlp_24c1g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_24c1g_numa_p2p1_r1_call6_phase_envelopes](mlp_24c1g_numa_p2p1_r1_call6_phase_envelopes.png)

### mlp_24c1g_numa_p2p1_r1_call6_terrain_detail

[PDF](mlp_24c1g_numa_p2p1_r1_call6_terrain_detail.pdf)
![mlp_24c1g_numa_p2p1_r1_call6_terrain_detail](mlp_24c1g_numa_p2p1_r1_call6_terrain_detail.png)

### mlp_24c1g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_24c1g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_24c1g_numa_p2pfull_r1_call6_phase_envelopes](mlp_24c1g_numa_p2pfull_r1_call6_phase_envelopes.png)

### mlp_24c1g_numa_p2pfull_r1_call6_terrain_detail

[PDF](mlp_24c1g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_24c1g_numa_p2pfull_r1_call6_terrain_detail](mlp_24c1g_numa_p2pfull_r1_call6_terrain_detail.png)

### mlp_24p24c1g_default_p2p1_r1_call6_phase_envelopes

[PDF](mlp_24p24c1g_default_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_24p24c1g_default_p2p1_r1_call6_phase_envelopes](mlp_24p24c1g_default_p2p1_r1_call6_phase_envelopes.png)

### mlp_24p24c1g_default_p2p1_r1_call6_terrain_detail

[PDF](mlp_24p24c1g_default_p2p1_r1_call6_terrain_detail.pdf)
![mlp_24p24c1g_default_p2p1_r1_call6_terrain_detail](mlp_24p24c1g_default_p2p1_r1_call6_terrain_detail.png)

### mlp_24p24c1g_default_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_24p24c1g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_24p24c1g_default_p2pfull_r1_call6_phase_envelopes](mlp_24p24c1g_default_p2pfull_r1_call6_phase_envelopes.png)

### mlp_24p24c1g_default_p2pfull_r1_call6_terrain_detail

[PDF](mlp_24p24c1g_default_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_24p24c1g_default_p2pfull_r1_call6_terrain_detail](mlp_24p24c1g_default_p2pfull_r1_call6_terrain_detail.png)

### mlp_24p24c1g_numa_p2p1_r1_call6_phase_envelopes

[PDF](mlp_24p24c1g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_24p24c1g_numa_p2p1_r1_call6_phase_envelopes](mlp_24p24c1g_numa_p2p1_r1_call6_phase_envelopes.png)

### mlp_24p24c1g_numa_p2p1_r1_call6_terrain_detail

[PDF](mlp_24p24c1g_numa_p2p1_r1_call6_terrain_detail.pdf)
![mlp_24p24c1g_numa_p2p1_r1_call6_terrain_detail](mlp_24p24c1g_numa_p2p1_r1_call6_terrain_detail.png)

### mlp_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes](mlp_24p24c1g_numa_p2pfull_r1_call6_phase_envelopes.png)

### mlp_24p24c1g_numa_p2pfull_r1_call6_terrain_detail

[PDF](mlp_24p24c1g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_24p24c1g_numa_p2pfull_r1_call6_terrain_detail](mlp_24p24c1g_numa_p2pfull_r1_call6_terrain_detail.png)

### mlp_96c4g_default_p2p1_r1_call6_phase_envelopes

[PDF](mlp_96c4g_default_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_96c4g_default_p2p1_r1_call6_phase_envelopes](mlp_96c4g_default_p2p1_r1_call6_phase_envelopes.png)

### mlp_96c4g_default_p2p1_r1_call6_terrain_detail

[PDF](mlp_96c4g_default_p2p1_r1_call6_terrain_detail.pdf)
![mlp_96c4g_default_p2p1_r1_call6_terrain_detail](mlp_96c4g_default_p2p1_r1_call6_terrain_detail.png)

### mlp_96c4g_default_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_96c4g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_96c4g_default_p2pfull_r1_call6_phase_envelopes](mlp_96c4g_default_p2pfull_r1_call6_phase_envelopes.png)

### mlp_96c4g_default_p2pfull_r1_call6_terrain_detail

[PDF](mlp_96c4g_default_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_96c4g_default_p2pfull_r1_call6_terrain_detail](mlp_96c4g_default_p2pfull_r1_call6_terrain_detail.png)

### mlp_96c4g_numa_p2p1_r1_call6_phase_envelopes

[PDF](mlp_96c4g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_96c4g_numa_p2p1_r1_call6_phase_envelopes](mlp_96c4g_numa_p2p1_r1_call6_phase_envelopes.png)

### mlp_96c4g_numa_p2p1_r1_call6_terrain_detail

[PDF](mlp_96c4g_numa_p2p1_r1_call6_terrain_detail.pdf)
![mlp_96c4g_numa_p2p1_r1_call6_terrain_detail](mlp_96c4g_numa_p2p1_r1_call6_terrain_detail.png)

### mlp_96c4g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_96c4g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_96c4g_numa_p2pfull_r1_call6_phase_envelopes](mlp_96c4g_numa_p2pfull_r1_call6_phase_envelopes.png)

### mlp_96c4g_numa_p2pfull_r1_call6_terrain_detail

[PDF](mlp_96c4g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_96c4g_numa_p2pfull_r1_call6_terrain_detail](mlp_96c4g_numa_p2pfull_r1_call6_terrain_detail.png)

### mlp_96p96c4g_default_p2p1_r1_call6_phase_envelopes

[PDF](mlp_96p96c4g_default_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_96p96c4g_default_p2p1_r1_call6_phase_envelopes](mlp_96p96c4g_default_p2p1_r1_call6_phase_envelopes.png)

### mlp_96p96c4g_default_p2p1_r1_call6_terrain_detail

[PDF](mlp_96p96c4g_default_p2p1_r1_call6_terrain_detail.pdf)
![mlp_96p96c4g_default_p2p1_r1_call6_terrain_detail](mlp_96p96c4g_default_p2p1_r1_call6_terrain_detail.png)

### mlp_96p96c4g_default_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_96p96c4g_default_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_96p96c4g_default_p2pfull_r1_call6_phase_envelopes](mlp_96p96c4g_default_p2pfull_r1_call6_phase_envelopes.png)

### mlp_96p96c4g_default_p2pfull_r1_call6_terrain_detail

[PDF](mlp_96p96c4g_default_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_96p96c4g_default_p2pfull_r1_call6_terrain_detail](mlp_96p96c4g_default_p2pfull_r1_call6_terrain_detail.png)

### mlp_96p96c4g_numa_p2p1_r1_call6_phase_envelopes

[PDF](mlp_96p96c4g_numa_p2p1_r1_call6_phase_envelopes.pdf)
![mlp_96p96c4g_numa_p2p1_r1_call6_phase_envelopes](mlp_96p96c4g_numa_p2p1_r1_call6_phase_envelopes.png)

### mlp_96p96c4g_numa_p2p1_r1_call6_terrain_detail

[PDF](mlp_96p96c4g_numa_p2p1_r1_call6_terrain_detail.pdf)
![mlp_96p96c4g_numa_p2p1_r1_call6_terrain_detail](mlp_96p96c4g_numa_p2p1_r1_call6_terrain_detail.png)

### mlp_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes

[PDF](mlp_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes.pdf)
![mlp_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes](mlp_96p96c4g_numa_p2pfull_r1_call6_phase_envelopes.png)

### mlp_96p96c4g_numa_p2pfull_r1_call6_terrain_detail

[PDF](mlp_96p96c4g_numa_p2pfull_r1_call6_terrain_detail.pdf)
![mlp_96p96c4g_numa_p2pfull_r1_call6_terrain_detail](mlp_96p96c4g_numa_p2pfull_r1_call6_terrain_detail.png)

