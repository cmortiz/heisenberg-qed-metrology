# Data and reproducibility manifest

This repository contains the simulation code and committed raw data that recreate
the numerical results in *Encoded Quantum Signal Processing for Heisenberg-Limited
Metrology*. Every table and figure in the paper is reproducible from the raw CSVs
in `scripts/results/` via the analysis scripts listed below.

## Pipeline

```
simulation_*.py  ->  scripts/results/*.csv  (raw, committed)  ->  analyze_*.py  ->  paper table/figure
```

Run the full simulation suite with `scripts/run_suite.sh` (GPU, long). The raw CSVs
are committed, so the analysis scripts reproduce every reported number without re-running
the suite.

## Table / figure -> data -> script map

| Paper element | Raw data (`scripts/results/`) | Simulation script | Analysis script |
|---|---|---|---|
| Table `tab:scaling-results` (bare GHZ, depolarizing) | `bare_ghz.csv` | `simulation.py --mode ghz` | `analyze_results.py` |
| Table `tab:error-detected-results` + `tab:prefactor-results` (ED-FL vs bare GHZ, Hamiltonian) | `bare_hamiltonian.csv`, `ed_full_likelihood.csv`, `ed_postselect.csv` | `simulation_bare_hamiltonian.py`, `simulation_error_detected.py` | `reanalysis_within_hamiltonian.py` |
| Table `tab:combined-results` (combined protocol) | `combined_full_likelihood.csv`, `combined_postselect.csv`, `bare_ghz.csv` | `simulation_combined.py` | `analyze_combined.py` |
| Table `tab:sequential-results` (sequential binary search) | `sequential.csv` | `simulation_sequential.py` | `analyze_sequential.py` |
| Per-shot Fisher-saturation ratios (Numerical-Simulations section) | `ed_full_likelihood.csv`, `combined_full_likelihood.csv` | (as above) | `analyze_fisher_saturation.py` |
| Quenched-vs-marginalized estimator comparison (Methods aside) | `combined_full_likelihood{,_marg,_marg_qt,_cond_only}.csv`, `ed_full_likelihood_cond_only.csv` | `simulation_combined.py`, `simulation_error_detected.py` (with `--marg` / `--cond-only`) | `compare_quenched_vs_marg.py`, `compare_qt_vs_quenched.py` |
| Figures 1-3 | (table CSVs above) | -- | `plot_redesigned.py` |

## Fit convention (scaling exponent alpha)

All tables fit `log T = -alpha * log(eps_tar) + c` by ordinary least squares on the
**converged** rows, per seed, then report the across-seed mean and standard error
(SEM). Two fit windows appear in the paper, and which one each table uses is fixed:

| Window | min_points | Tables |
|---|---|---|
| **Full-range** (`eps_tar` in `[1e-4, 1e-1]`) | 10 | `tab:scaling-results`, `tab:sequential-results` |
| **Asymptotic** (`eps_tar < 1e-2`) | 5 | `tab:error-detected-results`, `tab:combined-results`, Fisher-saturation, quenched-vs-marg |

The two windows differ by the polylogarithmic overhead and are **not directly
comparable** across tables. `analyze_results.py` reports both windows for every
config; `analyze_sequential.py` reuses its exact fit code. The `Seeds` column of a
table is the number of seeds retaining at least `min_points` converged targets (so
it can be below 40 in low-convergence regimes, e.g. `sigma_eps = 0.05` in
`tab:sequential-results`).

## What is intentionally NOT in this repository

Exploratory pilots, methodology-decision scripts, and regenerable intermediate
analyzer outputs are kept local (see `.gitignore`); they are not needed to recreate
any paper result. Regenerable derived summaries are produced on demand by the
analysis scripts above from the committed raw CSVs.
