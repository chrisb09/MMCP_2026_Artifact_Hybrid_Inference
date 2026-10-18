# AIx Pipeline Draft

Each controller has its own zero; rows are NOT globally synchronized. CUDA-event durations are reliable; host-anchored inter-request overlap is approximate. GPU work SUM is not elapsed time or a critical path.

Only exact selected AIx cells in consolidated runs.csv and inferences.csv are used. Collective measurements are durations only; no collective timestamps are reconstructed. No worker/controller or cross-controller clock alignment is assumed. Send triangles mark controller post/completion observations, not worker receipt or network transfer duration.

Calls are zero-based in diagnostics, one-based in inferences.csv; calls 0 and 1 are warmup. GPU phases use CUDA-event-derived durations anchored to each request's host clock; row order need not be chronological. Each timeline row resets to its own controller ml_step_start. Rows share the same time scale; dashed vertical lines mark each controller's exchange end. Range bars are host spans, GPU envelopes are first-phase-start to last-phase-end. Collective host device_cpu elapsed is a different scope, not a GPU envelope. Small intervals retain their real widths with hairline edges; none are inflated.

[Workgroup metrics](workgroup_metrics.csv) include CPU gather/scatter audit durations, controller exchange duration, overall MAIA makespan and request counts. [Events](events.csv) include paired intervals and raw controller observations. [Metadata](metadata.json) records selected cells, their hashes, source CSV/diagnostic/log SHA256 hashes, and counts. Default GPU ordinals follow the logged node-local controller rank, as assigned by RoundRobinDistribution::createWorkgroups; affine/NUMA use explicit logged GPU selections. Missing/conflicting mappings are errors.

## Gallery

### transformer | 96c4g | default | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96c4g_default_p2p1_r1_call6_timeline.pdf)
![timeline](transformer_96c4g_default_p2p1_r1_call6_timeline.png)

[gpu_work PDF](transformer_96c4g_default_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96c4g_default_p2p1_r1_call6_gpu_work.png)

### transformer | 96c4g | default | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96c4g_default_p2pfull_r1_call6_timeline.pdf)
![timeline](transformer_96c4g_default_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](transformer_96c4g_default_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96c4g_default_p2pfull_r1_call6_gpu_work.png)

### transformer | 96c4g | numa | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96c4g_numa_p2p1_r1_call6_timeline.pdf)
![timeline](transformer_96c4g_numa_p2p1_r1_call6_timeline.png)

[gpu_work PDF](transformer_96c4g_numa_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96c4g_numa_p2p1_r1_call6_gpu_work.png)

### transformer | 96c4g | numa | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96c4g_numa_p2pfull_r1_call6_timeline.pdf)
![timeline](transformer_96c4g_numa_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](transformer_96c4g_numa_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96c4g_numa_p2pfull_r1_call6_gpu_work.png)

### transformer | 96p96c4g | default | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96p96c4g_default_p2p1_r1_call6_timeline.pdf)
![timeline](transformer_96p96c4g_default_p2p1_r1_call6_timeline.png)

[gpu_work PDF](transformer_96p96c4g_default_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96p96c4g_default_p2p1_r1_call6_gpu_work.png)

### transformer | 96p96c4g | default | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96p96c4g_default_p2pfull_r1_call6_timeline.pdf)
![timeline](transformer_96p96c4g_default_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](transformer_96p96c4g_default_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96p96c4g_default_p2pfull_r1_call6_gpu_work.png)

### transformer | 96p96c4g | numa | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96p96c4g_numa_p2p1_r1_call6_timeline.pdf)
![timeline](transformer_96p96c4g_numa_p2p1_r1_call6_timeline.png)

[gpu_work PDF](transformer_96p96c4g_numa_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96p96c4g_numa_p2p1_r1_call6_gpu_work.png)

### transformer | 96p96c4g | numa | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](transformer_96p96c4g_numa_p2pfull_r1_call6_timeline.pdf)
![timeline](transformer_96p96c4g_numa_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](transformer_96p96c4g_numa_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](transformer_96p96c4g_numa_p2pfull_r1_call6_gpu_work.png)

### mlp | 96c4g | default | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96c4g_default_p2p1_r1_call6_timeline.pdf)
![timeline](mlp_96c4g_default_p2p1_r1_call6_timeline.png)

[gpu_work PDF](mlp_96c4g_default_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96c4g_default_p2p1_r1_call6_gpu_work.png)

### mlp | 96c4g | default | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96c4g_default_p2pfull_r1_call6_timeline.pdf)
![timeline](mlp_96c4g_default_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](mlp_96c4g_default_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96c4g_default_p2pfull_r1_call6_gpu_work.png)

### mlp | 96c4g | numa | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96c4g_numa_p2p1_r1_call6_timeline.pdf)
![timeline](mlp_96c4g_numa_p2p1_r1_call6_timeline.png)

[gpu_work PDF](mlp_96c4g_numa_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96c4g_numa_p2p1_r1_call6_gpu_work.png)

### mlp | 96c4g | numa | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96c4g_numa_p2pfull_r1_call6_timeline.pdf)
![timeline](mlp_96c4g_numa_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](mlp_96c4g_numa_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96c4g_numa_p2pfull_r1_call6_gpu_work.png)

### mlp | 96p96c4g | default | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96p96c4g_default_p2p1_r1_call6_timeline.pdf)
![timeline](mlp_96p96c4g_default_p2p1_r1_call6_timeline.png)

[gpu_work PDF](mlp_96p96c4g_default_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96p96c4g_default_p2p1_r1_call6_gpu_work.png)

### mlp | 96p96c4g | default | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96p96c4g_default_p2pfull_r1_call6_timeline.pdf)
![timeline](mlp_96p96c4g_default_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](mlp_96p96c4g_default_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96p96c4g_default_p2pfull_r1_call6_gpu_work.png)

### mlp | 96p96c4g | numa | collective vs p2p1 | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96p96c4g_numa_p2p1_r1_call6_timeline.pdf)
![timeline](mlp_96p96c4g_numa_p2p1_r1_call6_timeline.png)

[gpu_work PDF](mlp_96p96c4g_numa_p2p1_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96p96c4g_numa_p2p1_r1_call6_gpu_work.png)

### mlp | 96p96c4g | numa | collective vs p2pfull | rep 1 | call 6 / MAIA step 189

[timeline PDF](mlp_96p96c4g_numa_p2pfull_r1_call6_timeline.pdf)
![timeline](mlp_96p96c4g_numa_p2pfull_r1_call6_timeline.png)

[gpu_work PDF](mlp_96p96c4g_numa_p2pfull_r1_call6_gpu_work.pdf)
![gpu_work](mlp_96p96c4g_numa_p2pfull_r1_call6_gpu_work.png)

## Selected Sources

- `aix-96c4g-default-coll-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-collective_gpu_5009530_Gb07lW/solver.log)
- `aix-96c4g-default-p2p1-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-p2p1_gpu_5009530_xEFjgQ/solver.log)
- `aix-96c4g-default-p2pfull-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-p2pfull_gpu_5009530_d8UReH/solver.log)
- `aix-96c4g-numa-coll-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-collective_gpu_5009530_IpasOn/solver.log)
- `aix-96c4g-numa-p2p1-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-p2p1_gpu_5009530_avdsba/solver.log)
- `aix-96c4g-numa-p2pfull-transformer-r1`, job `5009530`: [solver.log](../../scratch/cmi_smokes/aix-p2pfull_gpu_5009530_4bOAsK/solver.log)
- `aix-96p96c4g-default-coll-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-coll-transformer-r1_5009534_6DvZ18/solver.log)
- `aix-96p96c4g-default-p2p1-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-p2p1-transformer-r1_5009534_kFiTgR/solver.log)
- `aix-96p96c4g-default-p2pfull-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-p2pfull-transformer-r1_5009534_3imolw/solver.log)
- `aix-96p96c4g-numa-coll-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-coll-transformer-r1_5009534_cozQYB/solver.log)
- `aix-96p96c4g-numa-p2p1-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-p2p1-transformer-r1_5009534_jNncF1/solver.log)
- `aix-96p96c4g-numa-p2pfull-transformer-r1`, job `5009534`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-p2pfull-transformer-r1_5009534_8Ga6SO/solver.log)
- `aix-96c4g-default-coll-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-collective_gpu_5046471_ahXVA0/solver.log)
- `aix-96c4g-default-p2p1-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-p2p1_gpu_5046471_0WFVLM/solver.log)
- `aix-96c4g-default-p2pfull-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-p2pfull_gpu_5046471_OrfLXJ/solver.log)
- `aix-96c4g-numa-coll-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-collective_gpu_5046471_d28NTU/solver.log)
- `aix-96c4g-numa-p2p1-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-p2p1_gpu_5046471_bHMSG5/solver.log)
- `aix-96c4g-numa-p2pfull-mlp-r1`, job `5046471`: [solver.log](../../scratch/cmi_smokes/aix-p2pfull_gpu_5046471_ErGG4c/solver.log)
- `aix-96p96c4g-default-coll-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-coll-mlp-r1_5046475_itUKtu/solver.log)
- `aix-96p96c4g-default-p2p1-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-p2p1-mlp-r1_5046475_gBLB2f/solver.log)
- `aix-96p96c4g-default-p2pfull-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-default-p2pfull-mlp-r1_5046475_8uDGD0/solver.log)
- `aix-96p96c4g-numa-coll-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-coll-mlp-r1_5046475_FY4ilM/solver.log)
- `aix-96p96c4g-numa-p2p1-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-p2p1-mlp-r1_5046475_qMVjFi/solver.log)
- `aix-96p96c4g-numa-p2pfull-mlp-r1`, job `5046475`: [solver.log](../../scratch/cmi_bench/aix-96p96c4g-numa-p2pfull-mlp-r1_5046475_osw4pO/solver.log)
