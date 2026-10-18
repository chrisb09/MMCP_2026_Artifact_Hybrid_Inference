# AIx affine Stage Table

All durations are in milliseconds. Each row averages three independent
repetition means, each from five measured calls. c1 and full-credit ideal columns
are aliases of ONE COMMON credit-independent prediction, not separate models.
Measured ML-step times are canonical MAIA full-step latencies. Ready-range windows
are measured controller-local spans, NOT five-stage elapsed times or global
makespans. See README.md for definitions, assumptions, and uncertainty.

| Model | Hardware | Send input | H2D | Forward | D2H | Send output | Stage sum | Coll. ML step | c1 ideal pipeline | c1 ideal ML step | Full ideal pipeline | Full ideal ML step | c1 measured ML step | Full measured ML step | c1 ready-range window | Full ready-range window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 24c/1GPU | 275.025 | 95.776 | 11.196 | 49.292 | 161.977 | 593.266 | 1091.609 | 288.285 | 786.627 | 288.285 | 786.627 | 985.772 | 998.748 | 428.233 | 393.454 |
| mlp | 24+24c/1GPU | 239.896 | 95.783 | 11.191 | 48.865 | 131.521 | 527.255 | 753.226 | 245.882 | 471.853 | 245.882 | 471.853 | 609.433 | 616.555 | 349.925 | 343.693 |
| mlp | 96c/4GPUs | 161.983 | 29.231 | 2.885 | 17.356 | 65.271 | 276.727 | 416.502 | 172.159 | 311.934 | 172.159 | 311.934 | 401.379 | 322.711 | 220.849 | 151.843 |
| mlp | 96+96c/4GPUs | 113.349 | 26.477 | 2.606 | 13.688 | 58.277 | 214.397 | 279.251 | 115.522 | 180.376 | 115.522 | 180.376 | 220.163 | 215.820 | 139.183 | 124.740 |
| transformer | 24c/1GPU | 42.281 | 15.459 | 5547.717 | 8.023 | 26.683 | 5640.164 | 5740.513 | 5551.569 | 5651.918 | 5551.569 | 5651.918 | 5905.723 | 5938.921 | 5832.624 | 5864.210 |
| transformer | 24+24c/1GPU | 41.121 | 15.455 | 5585.479 | 10.913 | 21.637 | 5674.605 | 5712.562 | 5587.336 | 5625.292 | 5587.336 | 5625.292 | 6589.234 | 6398.746 | 6553.913 | 6368.145 |
| transformer | 96c/4GPUs | 25.619 | 5.064 | 1629.890 | 4.233 | 11.360 | 1676.166 | 1721.305 | 1632.026 | 1677.165 | 1632.026 | 1677.165 | 2001.904 | 2027.483 | 1970.054 | 2004.630 |
| transformer | 96+96c/4GPUs | 24.036 | 5.355 | 1657.194 | 7.366 | 12.846 | 1706.798 | 1735.297 | 1661.029 | 1689.529 | 1661.029 | 1689.529 | 2497.127 | 2479.864 | 2475.894 | 2459.552 |
