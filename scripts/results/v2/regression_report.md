# Regression report (diagnostic, not gating)

- generated: 2026-09-02T02:58:14
- numerics HEAD: 715a7f9; jax: 0.9.0.1; python: 3.12.13; platform: macOS-26.6.1-arm64-arm-64bit
- baseline: committed CSVs under scripts/results/ (tag v1-jax-suite, GPU float32 XLA, May 2026)
- exact tier compares total_resources (int), converged (bool), final_error (rel 1e-9)
- statistical tier: per-seed alpha = -OLS slope of log T on log eps over converged rows, full range, >= 10 points;
  cell alpha = mean over seeds, SEM = std/sqrt(n); flag |delta alpha| > 2 sqrt(SEM_v1^2 + SEM_v2^2) or |delta conv frac| > 0.02

## Exact tier

runner: in-process `engine.run_epsilon` (run.py advertises no eps-selection flag, or does not exist yet); same per-eps code path as run.py

| cell | seed | eps | v1 T | v2 T | v1 conv | v2 conv | v1 err | v2 err | match (T, conv, err) |
|---|---|---|---|---|---|---|---|---|---|
| bare_adaptive_g0 | 1 | 0.07912 | 673 | 673 | True | True | 4.495174e-03 | 4.495174e-03 | True, True, True |
| bare_adaptive_g0 | 1 | 0.08895 | 1163 | 2401 | True | True | 1.618981e-03 | 6.844103e-03 | False, True, False |
| bare_adaptive_g0 | 1 | 0.1 | 1583 | 1583 | True | True | 4.205406e-04 | 4.205406e-04 | True, True, True |
| bare_adaptive_g0 | 2 | 0.07912 | 1221 | 1221 | True | True | 2.481848e-03 | 2.481848e-03 | True, True, True |
| bare_adaptive_g0 | 2 | 0.08895 | 3123 | 3123 | True | True | 7.561147e-04 | 7.561147e-04 | True, True, True |
| bare_adaptive_g0 | 2 | 0.1 | 528 | 1639 | True | True | 1.501513e-02 | 1.420021e-02 | False, True, False |
| bare_fixed_L1_g0.05 | 1 | 0.07912 | 1998 | 1998 | True | True | 1.043707e-03 | 1.043707e-03 | True, True, True |
| bare_fixed_L1_g0.05 | 1 | 0.08895 | 1641 | 1641 | True | True | 2.359807e-03 | 2.359807e-03 | True, True, True |
| bare_fixed_L1_g0.05 | 1 | 0.1 | 3027 | 3027 | True | True | 1.209319e-03 | 1.209319e-03 | True, True, True |
| bare_fixed_L1_g0.05 | 2 | 0.07912 | 1635 | 1635 | True | True | 3.318548e-03 | 3.318548e-03 | True, True, True |
| bare_fixed_L1_g0.05 | 2 | 0.08895 | 1998 | 1998 | True | True | 2.146274e-03 | 2.146274e-03 | True, True, True |
| bare_fixed_L1_g0.05 | 2 | 0.1 | 3345 | 3345 | True | True | 4.351377e-03 | 4.351377e-03 | True, True, True |
| ed_fl_L1_g0.01 | 1 | 0.07912 | 1803 | 1803 | True | True | 2.385944e-03 | 2.385944e-03 | True, True, True |
| ed_fl_L1_g0.01 | 1 | 0.08895 | 25080 | 25080 | True | True | 2.985299e-04 | 2.985299e-04 | True, True, True |
| ed_fl_L1_g0.01 | 1 | 0.1 | 5436 | 5436 | True | True | 6.340742e-04 | 6.819963e-04 | True, True, False |
| ed_fl_L1_g0.01 | 2 | 0.07912 | 12933 | 12933 | True | True | 1.187533e-03 | 1.187533e-03 | True, True, True |
| ed_fl_L1_g0.01 | 2 | 0.08895 | 2577 | 2577 | True | True | 5.715400e-03 | 5.763322e-03 | True, True, False |
| ed_fl_L1_g0.01 | 2 | 0.1 | 9921 | 9921 | True | True | 1.043707e-03 | 1.043707e-03 | True, True, True |
| ed_ps_L1_g0.01 | 1 | 0.07912 | 2328 | 2328 | True | True | 1.090765e-05 | 1.090765e-05 | True, True, True |
| ed_ps_L1_g0.01 | 1 | 0.08895 | 1155 | 1155 | True | True | 3.464520e-04 | 3.464520e-04 | True, True, True |
| ed_ps_L1_g0.01 | 1 | 0.1 | 537 | 537 | True | True | 3.510296e-03 | 3.510296e-03 | True, True, True |
| ed_ps_L1_g0.01 | 2 | 0.07912 | 645 | 645 | True | True | 1.321968e-02 | 1.321968e-02 | True, True, True |
| ed_ps_L1_g0.01 | 2 | 0.08895 | 2481 | 2481 | True | True | 6.050944e-03 | 6.002992e-03 | True, True, False |
| ed_ps_L1_g0.01 | 2 | 0.1 | 1206 | 1206 | True | True | 1.880437e-03 | 1.880437e-03 | True, True, True |
| combined_fl_L1_g0.01_s0.01 | 1 | 0.07912 | 8550 | 8550 | True | True | 9.419709e-02 | 9.419709e-02 | True, True, True |
| combined_fl_L1_g0.01_s0.01 | 1 | 0.08895 | 18360 | 18360 | True | True | 9.630632e-02 | 9.634227e-02 | True, True, False |
| combined_fl_L1_g0.01_s0.01 | 1 | 0.1 | 9450 | 9450 | True | True | 9.176430e-02 | 9.176430e-02 | True, True, True |
| combined_fl_L1_g0.01_s0.01 | 2 | 0.07912 | 8820 | 8820 | True | True | 9.489217e-02 | 9.494013e-02 | True, True, False |
| combined_fl_L1_g0.01_s0.01 | 2 | 0.08895 | 7830 | 7830 | True | True | 9.299868e-02 | 9.299868e-02 | True, True, True |
| combined_fl_L1_g0.01_s0.01 | 2 | 0.1 | 13860 | 13860 | True | True | 1.966506e-03 | 1.990467e-03 | True, True, False |
| combined_ps_L1_g0.01_s0.01 | 1 | 0.07912 | 71181 | 45054 | True | True | 6.100535e-05 | 9.428099e-02 | False, True, False |
| combined_ps_L1_g0.01_s0.01 | 1 | 0.08895 | 3600 | 3600 | True | True | 2.156079e-03 | 2.156079e-03 | True, True, True |
| combined_ps_L1_g0.01_s0.01 | 1 | 0.1 | 11709 | 19809 | True | True | 2.110302e-03 | 9.343010e-02 | False, True, False |
| combined_ps_L1_g0.01_s0.01 | 2 | 0.07912 | 11718 | 54954 | True | True | 9.446076e-02 | 4.303455e-04 | False, True, False |
| combined_ps_L1_g0.01_s0.01 | 2 | 0.08895 | 195462 | 4500 | True | True | 9.706134e-02 | 6.232888e-03 | False, True, False |
| combined_ps_L1_g0.01_s0.01 | 2 | 0.1 | 18918 | 29718 | True | True | 2.637625e-03 | 2.287924e-04 | False, True, False |
| combined_marg_L1_g0.01_s0.01 | 1 | 0.07912 | 16110 | 16110 | True | True | 6.700158e-04 | 7.059872e-04 | True, True, False |
| combined_marg_L1_g0.01_s0.01 | 1 | 0.08895 | 44010 | 44010 | True | True | 9.894285e-02 | 9.891888e-02 | True, True, False |
| combined_marg_L1_g0.01_s0.01 | 1 | 0.1 | 83520 | 83520 | True | True | 4.205406e-04 | 4.564822e-04 | True, True, False |
| combined_marg_L1_g0.01_s0.01 | 2 | 0.07912 | 97290 | 97290 | True | True | 4.662871e-04 | 4.782677e-04 | True, True, False |
| combined_marg_L1_g0.01_s0.01 | 2 | 0.08895 | 7830 | 7830 | True | True | 9.323835e-02 | 9.327433e-02 | True, True, False |
| combined_marg_L1_g0.01_s0.01 | 2 | 0.1 | 13860 | 13860 | True | True | 1.103640e-03 | 1.079679e-03 | True, True, False |
| sequential_L1_s0 | 1 | 0.07912 | 1440 | 1440 | True | True | 4.361170e-02 | 4.361170e-02 | True, True, True |
| sequential_L1_s0 | 1 | 0.08895 | 1440 | 1440 | True | True | 5.456307e-02 | 5.456307e-02 | True, True, True |
| sequential_L1_s0 | 1 | 0.1 | 720 | 720 | True | True | 5.475689e-03 | 5.475689e-03 | True, True, True |
| sequential_L1_s0 | 2 | 0.07912 | 1440 | 1440 | True | True | 4.361170e-02 | 4.361170e-02 | True, True, True |
| sequential_L1_s0 | 2 | 0.08895 | 1440 | 1440 | True | True | 4.361170e-02 | 4.361170e-02 | True, True, True |
| sequential_L1_s0 | 2 | 0.1 | 720 | 720 | True | True | 5.475689e-03 | 5.475689e-03 | True, True, True |

rows compared: 48; rows identical in all three columns: 29

## Statistical tier

| cell | v1 n_seeds | v1 alpha | v1 SEM | v2 n_seeds | v2 alpha | v2 SEM | delta | 2 sigma | v1 conv | v2 conv | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bare_adaptive_g0 | | | | 0 | | | | | | | no v2 rows |
| bare_fixed_L1_g0.05 | | | | 0 | | | | | | | no v2 rows |
| ed_fl_L1_g0.01 | | | | 0 | | | | | | | no v2 rows |
| ed_ps_L1_g0.01 | | | | 0 | | | | | | | no v2 rows |
| combined_fl_L1_g0.01_s0.01 | | | | 0 | | | | | | | no v2 rows |
| combined_ps_L1_g0.01_s0.01 | | | | 0 | | | | | | | no v2 rows |
| combined_marg_L1_g0.01_s0.01 | | | | 0 | | | | | | | no v2 rows |
| sequential_L1_s0 | | | | 0 | | | | | | | no v2 rows |

