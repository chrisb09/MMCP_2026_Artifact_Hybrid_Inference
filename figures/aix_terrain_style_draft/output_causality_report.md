# Output Causality Audit

Read-only analysis of raw timestamps; no offsets or durations are changed in the figures.

## Event Evidence

Source: `extern/AIxeleratorService/src/communicationStrategy/collectiveCommunication.cpp` (external renderer repository), lines 239-258, 323-331, 360-377, 398-409. Worker send start follows credit receipt; input send complete follows blocking input ACK receipt; result received follows blocking output receipt. Controller input ready precedes ACK Isend, and result send start precedes output Isend. Thus input ready and send complete are not simultaneous.

`src/utils/p2pTimeline.h:16-21,90-106` uses MPI_Wtime plus optional AIX_P2P_CLOCK_OFFSET_SECONDS, sampled before taking the logging mutex. MPI_Wtime does not establish a common rank clock. No measured clock synchronization or drift estimate is available. Source inspection describes the available source, not independent proof of the exact binary used.

## Bounds

With controller time = worker time + delta, input causality gives `L=max(credit-send_start, ready-send_complete)`, `U=ready-send_start`. Output causality adds `delta >= result_send-result_received`; tightened `L2=max(L, result_send-result_received)`. Feasibility is `L2 <= U`. These are per-worker, per-call necessary causal bounds, not a measured synchronization; a midpoint is not used. Equality at ACK is only one component of the input lower bound, and can even violate the credit lower bound. It assumes zero ACK latency.

Workers: 2800; negative ACK-aligned output spans: 44; feasible tightened intervals: 2800; infeasible: 0.

Validated raw same-clock worker and controller endpoint orders: 2800; ACK offsets satisfying all input bounds: 1885.

All workers: tightened width min/max = 3.569685/6125.91087 ms.
Negative ACK-span workers: tightened width min/max = 14.927374/368.755962 ms.

All negative cases below preserve raw same-clock durations. The signed span equals worker ACK-to-receipt minus controller ready-to-send. Feasible intervals show that these cases do not require a measured communication-order bug; ACK equality is incompatible with output causality. They do not prove an exact offset or rule out other bugs.

| Comparison | WG | Worker | ACK-to-receipt ms | Ready-to-send ms | Signed output ms | L2 s | U s | Width ms | Feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformer_96p96c4g_default_p2pfull_r1_call6 | 0 | 12 | 313.743027 | 2278.55204 | -1964.80901 | -0.023558269 | 0.16083421 | 184.392479 | True |
| transformer_96p96c4g_default_p2pfull_r1_call6 | 2 | 10 | 323.770597 | 2055.99284 | -1732.22225 | -0.028060787 | 0.020746221 | 48.807008 | True |
| transformer_96p96c4g_default_p2pfull_r1_call6 | 2 | 18 | 278.594769 | 1898.28453 | -1619.68976 | -0.029277475 | 0.17725596 | 206.533435 | True |
| transformer_96p96c4g_numa_p2p1_r1_call6 | 3 | 95 | 252.387427 | 294.02652 | -41.639093 | 0.036808956 | 0.082291607 | 45.482651 | True |
| transformer_96p96c4g_numa_p2pfull_r1_call6 | 0 | 8 | 2134.28003 | 2136.99179 | -2.711755 | 0.21092285 | 0.579678812 | 368.755962 | True |
| transformer_96p96c4g_numa_p2pfull_r1_call6 | 0 | 12 | 404.577308 | 2305.22642 | -1900.64911 | 0.207550067 | 0.408060254 | 200.510187 | True |
| transformer_96p96c4g_numa_p2pfull_r1_call6 | 0 | 20 | 354.109435 | 2231.34837 | -1877.23894 | 0.207179917 | 0.481568314 | 274.388397 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 0 | 4 | 37.958995 | 113.675345 | -75.71635 | 0.108556568 | 0.133859773 | 25.303205 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 0 | 12 | 39.324147 | 112.300998 | -72.976851 | 0.104633199 | 0.131323063 | 26.689864 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 0 | 20 | 39.537592 | 112.320981 | -72.783389 | 0.107588752 | 0.1342586 | 26.669848 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 1 | 13 | 37.877885 | 72.730869 | -34.852984 | 0.104276562 | 0.131475541 | 27.198979 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 1 | 21 | 38.29295 | 75.381378 | -37.088428 | 0.10104129 | 0.125573369 | 24.532079 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 2 | 6 | 40.557079 | 110.833262 | -70.276183 | 0.103674616 | 0.130822852 | 27.148236 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 2 | 14 | 40.941013 | 110.845889 | -69.904876 | 0.106442488 | 0.133578104 | 27.135616 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 2 | 22 | 37.524929 | 112.116994 | -74.592065 | 0.105147816 | 0.130984557 | 25.836741 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 3 | 7 | 42.124932 | 110.168291 | -68.043359 | 0.102257031 | 0.130968811 | 28.71178 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 3 | 15 | 41.240396 | 112.734973 | -71.494577 | 0.100466045 | 0.12659728 | 26.131235 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 3 | 23 | 41.503644 | 112.747419 | -71.243775 | 0.105306718 | 0.131426885 | 26.120167 | True |
| mlp_96p96c4g_default_p2pfull_r1_call6 | 3 | 63 | 42.681609 | 111.818146 | -69.136537 | 0.099733416 | 0.126778704 | 27.045288 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 0 | 41.221984 | 82.47549 | -41.253506 | 0.013570717 | 0.034118259 | 20.547542 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 8 | 52.374005 | 83.179535 | -30.80553 | 0.004624628 | 0.024458888 | 19.83426 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 16 | 53.019002 | 83.404455 | -30.385453 | 0.011673582 | 0.031272009 | 19.598427 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 20 | 45.042493 | 82.799008 | -37.756515 | 0.011923766 | 0.032130232 | 20.206466 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 24 | 41.299803 | 82.644546 | -41.344743 | 0.008164691 | 0.028526592 | 20.361901 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 0 | 36 | 30.314819 | 31.52538 | -1.210561 | -0.008367746 | 0.063214405 | 71.582151 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 1 | 1 | 33.631512 | 84.883791 | -51.252279 | 0.015139485 | 0.036654491 | 21.515006 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 1 | 9 | 31.627824 | 86.90044 | -55.272616 | 0.007357235 | 0.026838625 | 19.48139 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 1 | 17 | 31.385414 | 86.672254 | -55.28684 | 0.007366692 | 0.027075672 | 19.70898 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 1 | 25 | 33.186294 | 84.90349 | -51.717196 | 0.01514827 | 0.036618562 | 21.470292 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 2 | 30.893798 | 86.045364 | -55.151566 | 0.001642127 | 0.022497647 | 20.85552 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 10 | 65.129961 | 88.101588 | -22.971627 | 0.012072151 | 0.030861392 | 18.789241 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 18 | 31.129351 | 86.077036 | -54.947685 | 0.008869581 | 0.029695928 | 20.826347 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 26 | 30.827914 | 86.044093 | -55.216179 | 0.009198691 | 0.030050779 | 20.852088 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 34 | 30.24576 | 35.890842 | -5.645082 | -0.010799841 | 0.060321992 | 71.121833 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 2 | 58 | 30.163916 | 75.383676 | -45.21976 | -0.008716003 | 0.022793878 | 31.509881 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 3 | 46.912441 | 87.959398 | -41.046957 | 0.013065756 | 0.029604324 | 16.538568 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 11 | 55.350655 | 89.557621 | -34.206966 | 0.022049719 | 0.036977093 | 14.927374 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 15 | 47.076411 | 87.754558 | -40.678147 | 0.013547592 | 0.03028948 | 16.741888 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 19 | 35.0101 | 85.032122 | -50.022022 | 0.013650367 | 0.033117669 | 19.467302 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 23 | 45.805183 | 87.589198 | -41.784015 | 0.006550825 | 0.023449619 | 16.898794 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 27 | 46.997412 | 87.687077 | -40.689665 | 0.01445396 | 0.031255588 | 16.801628 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 39 | 35.685153 | 76.644325 | -40.959172 | -0.004579644 | 0.023268046 | 27.84769 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 55 | 37.746323 | 74.925762 | -37.179439 | -0.00670471 | 0.022857979 | 29.562689 | True |
| mlp_96p96c4g_numa_p2pfull_r1_call6 | 3 | 59 | 35.707369 | 75.191959 | -39.48459 | -0.005953522 | 0.023338755 | 29.292277 | True |
