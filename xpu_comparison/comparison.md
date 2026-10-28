# TabICL XPU benchmark: main vs PR #162

Same machine, same pixi env, runs executed back to back with `jupyter nbconvert --execute`. PR: https://github.com/soda-inria/tabicl/pull/162

## Environment

```
           package     version
0           tabicl       2.2.0
1            torch  2.14.1+xpu
2            numpy       2.5.3
3     scikit-learn       1.9.1
4            scipy      1.18.1
5           einops       0.8.2
6           psutil       7.2.2
7  huggingface-hub      1.32.0
8           pandas       3.0.6
9           python     3.13.15

  device  available               name  allocated_gb  total_gb  available_gb    runtime
0   cuda      False                NaN           NaN       NaN           NaN        NaN
1    xpu       True  Intel(R) Graphics         25.42     30.66         5.246  Intel XPU
2    mps      False                NaN           NaN       NaN           NaN        NaN
```

## Dataset

```
             value
n_train       1000
n_test        1000
n_features      50
noise          1.5
y_train_std  5.114
hgb_test_r2 0.8472
```

## `fit_s`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 0.4121 | 0.3918 |
| xpu / amp=False / kv_cache=False | 0.3404 | 0.4126 |
| xpu / amp=True / kv_cache=False | 0.3751 | 0.4569 |
| cpu / amp=False / kv_cache=True | 7.154 | 7.098 |
| xpu / amp=False / kv_cache=True | 2.848 | 2.364 |
| xpu / amp=True / kv_cache=True | 1.483 | 1.316 |

## `predict_s`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 9.091 | 10.77 |
| xpu / amp=False / kv_cache=False | 4.997 | 4.589 |
| xpu / amp=True / kv_cache=False | 3.289 | 2.105 |
| cpu / amp=False / kv_cache=True | 4.427 | 5.083 |
| xpu / amp=False / kv_cache=True | 1.809 | 1.594 |
| xpu / amp=True / kv_cache=True | 0.7683 | 0.6977 |

## `total_s`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 9.503 | 11.16 |
| xpu / amp=False / kv_cache=False | 5.338 | 5.002 |
| xpu / amp=True / kv_cache=False | 3.664 | 2.562 |
| cpu / amp=False / kv_cache=True | 11.58 | 12.18 |
| xpu / amp=False / kv_cache=True | 4.657 | 3.958 |
| xpu / amp=True / kv_cache=True | 2.252 | 2.014 |

## `speedup_vs_cpu`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | NaN | NaN |
| xpu / amp=False / kv_cache=False | 1.78 | 2.231 |
| xpu / amp=True / kv_cache=False | 2.594 | 4.355 |
| cpu / amp=False / kv_cache=True | NaN | NaN |
| xpu / amp=False / kv_cache=True | 2.487 | 3.078 |
| xpu / amp=True / kv_cache=True | 5.144 | 6.049 |

## `r2`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 0.9041 | 0.9041 |
| xpu / amp=False / kv_cache=False | 0.9041 | 0.9041 |
| xpu / amp=True / kv_cache=False | 0.904 | 0.904 |
| cpu / amp=False / kv_cache=True | 0.9041 | 0.9041 |
| xpu / amp=False / kv_cache=True | 0.9041 | 0.9041 |
| xpu / amp=True / kv_cache=True | 0.904 | 0.904 |

## `max_abs_diff_vs_cpu`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | NaN | NaN |
| xpu / amp=False / kv_cache=False | 1.431e-05 | 1.621e-05 |
| xpu / amp=True / kv_cache=False | 0.01837 | 0.01837 |
| cpu / amp=False / kv_cache=True | NaN | NaN |
| xpu / amp=False / kv_cache=True | 1.431e-05 | 1.621e-05 |
| xpu / amp=True / kv_cache=True | 0.01837 | 0.01853 |

## `host_rss_peak_increment_gb`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 2.72 | 0.9066 |
| xpu / amp=False / kv_cache=False | 2.023 | 1.839 |
| xpu / amp=True / kv_cache=False | 1.294 | 1.257 |
| cpu / amp=False / kv_cache=True | 2.11 | 1.364 |
| xpu / amp=False / kv_cache=True | 0.1209 | 0.08368 |
| xpu / amp=True / kv_cache=True | 0.07733 | 0.08578 |

## `device_mem_peak_increment_gb`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | 0 | 0 |
| xpu / amp=False / kv_cache=False | 3.169 | 0.8552 |
| xpu / amp=True / kv_cache=False | 0.8726 | 0.5886 |
| cpu / amp=False / kv_cache=True | 0 | 0 |
| xpu / amp=False / kv_cache=True | 2.209 | 1.286 |
| xpu / amp=True / kv_cache=True | 1.117 | 0.7588 |

## `swap_grew`

| run | main (0dbff3e) | PR #162 (4ac890e) |
|---|---|---|
| cpu / amp=False / kv_cache=False | False | False |
| xpu / amp=False / kv_cache=False | True | False |
| xpu / amp=True / kv_cache=False | True | False |
| cpu / amp=False / kv_cache=True | True | False |
| xpu / amp=False / kv_cache=True | True | False |
| xpu / amp=True / kv_cache=True | False | False |

## Run log: main (0dbff3e)

```
=== kv_cache=False ===

Running cpu / amp=False / kv_cache=False ...

  CPU done in 9.503s (fit 0.412s, predict 9.091s), R²=0.9041
Running xpu / amp=False / kv_cache=False ...

  xpu / amp=False / kv_cache=False: 5.338s (1.78× vs CPU), R²=0.9041, max|Δ|=1.431e-05
Running xpu / amp=True / kv_cache=False ...

  xpu / amp=True / kv_cache=False: 3.664s (2.59× vs CPU), R²=0.9040, max|Δ|=0.01837

=== kv_cache=True ===

Running cpu / amp=False / kv_cache=True ...

  CPU done in 11.581s (fit 7.154s, predict 4.427s), R²=0.9041
Running xpu / amp=False / kv_cache=True ...

  xpu / amp=False / kv_cache=True: 4.657s (2.49× vs CPU), R²=0.9041, max|Δ|=1.431e-05
Running xpu / amp=True / kv_cache=True ...

  xpu / amp=True / kv_cache=True: 2.252s (5.14× vs CPU), R²=0.9040, max|Δ|=0.01837
```

## Run log: PR #162 (4ac890e)

```
=== kv_cache=False ===

Running cpu / amp=False / kv_cache=False ...

  CPU done in 11.157s (fit 0.392s, predict 10.766s), R²=0.9041
Running xpu / amp=False / kv_cache=False ...

  xpu / amp=False / kv_cache=False: 5.002s (2.23× vs CPU), R²=0.9041, max|Δ|=1.621e-05
Running xpu / amp=True / kv_cache=False ...

  xpu / amp=True / kv_cache=False: 2.562s (4.35× vs CPU), R²=0.9040, max|Δ|=0.01837

=== kv_cache=True ===

Running cpu / amp=False / kv_cache=True ...

  CPU done in 12.181s (fit 7.098s, predict 5.083s), R²=0.9041
Running xpu / amp=False / kv_cache=True ...

  xpu / amp=False / kv_cache=True: 3.958s (3.08× vs CPU), R²=0.9041, max|Δ|=1.621e-05
Running xpu / amp=True / kv_cache=True ...

  xpu / amp=True / kv_cache=True: 2.014s (6.05× vs CPU), R²=0.9040, max|Δ|=0.01853
```
