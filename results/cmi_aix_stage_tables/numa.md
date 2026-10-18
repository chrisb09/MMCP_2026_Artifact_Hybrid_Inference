# AIx numa Stage Table

All durations are in milliseconds. Each row averages three independent
repetition means, each from five measured calls. c1 and full-credit ideal columns
are aliases of ONE COMMON credit-independent prediction, not separate models.
Measured ML-step times are canonical MAIA full-step latencies. Ready-range windows
are measured controller-local spans, NOT five-stage elapsed times or global
makespans. See README.md for definitions, assumptions, and uncertainty.

| Model | Hardware | Send input | H2D | Forward | D2H | Send output | Stage sum | Coll. ML step | c1 ideal pipeline | c1 ideal ML step | Full ideal pipeline | Full ideal ML step | c1 measured ML step | Full measured ML step | c1 ready-range window | Full ready-range window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 24c/1GPU | 270.705 | 95.779 | 11.191 | 49.350 | 166.423 | 593.447 | 1095.537 | 284.153 | 786.243 | 284.153 | 786.243 | 980.690 | 994.536 | 422.176 | 406.417 |
| mlp | 24+24c/1GPU | 240.922 | 95.763 | 11.195 | 48.841 | 131.320 | 528.041 | 756.781 | 246.904 | 475.644 | 246.904 | 475.644 | 600.361 | 622.581 | 348.236 | 347.733 |
| mlp | 96c/4GPUs | 136.823 | 29.145 | 2.899 | 17.654 | 59.369 | 245.891 | 379.532 | 141.619 | 275.260 | 141.619 | 275.260 | 375.332 | 320.397 | 215.930 | 158.553 |
| mlp | 96+96c/4GPUs | 105.636 | 26.407 | 2.608 | 13.835 | 67.904 | 216.390 | 284.937 | 107.950 | 176.497 | 107.950 | 176.497 | 212.334 | 208.764 | 136.523 | 122.420 |
| transformer | 24c/1GPU | 42.111 | 15.459 | 5548.480 | 8.019 | 27.303 | 5641.373 | 5743.332 | 5552.351 | 5654.310 | 5552.351 | 5654.310 | 5902.534 | 5944.452 | 5829.631 | 5871.105 |
| transformer | 24+24c/1GPU | 40.293 | 17.781 | 5686.409 | 10.905 | 21.043 | 5776.432 | 5847.299 | 5688.285 | 5759.151 | 5688.285 | 5759.151 | 6468.997 | 6407.741 | 6439.417 | 6376.610 |
| transformer | 96c/4GPUs | 21.701 | 5.004 | 1631.556 | 4.463 | 12.459 | 1675.182 | 1715.475 | 1634.169 | 1674.462 | 1634.169 | 1674.462 | 2054.560 | 2087.914 | 2028.447 | 2061.158 |
| transformer | 96+96c/4GPUs | 21.461 | 5.369 | 1665.822 | 6.161 | 14.703 | 1713.517 | 1746.397 | 1667.649 | 1700.528 | 1667.649 | 1700.528 | 2518.512 | 2536.794 | 2496.646 | 2513.759 |
