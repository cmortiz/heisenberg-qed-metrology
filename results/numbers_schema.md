# numbers.json schema

Written by `scripts/analyze.py` to `<results-dir>/numbers.json` (default `results/v2/`, or
`scripts/results/` with `--v1`). One key per paper element. Workstream G reads only this file;
`scripts/plot.py` takes every printed alpha from it and never fits.

## Value object

Every key maps to a dict with exactly the fields
`value, sem, window, min_points, source_csv, n_seeds, n_rows`, plus for fitted alphas
`ci95` (t-based 95% half-width), `intercept` (mean per-seed intercept of log T = intercept - alpha log eps,
used by plot.py to draw the fit line), `converged_pct` (over all rows of the cell), `n_seeds_run`,
and for every other entry `note` (the recipe). `null` in `sem` means the entry has no seed spread
(a ratio, a formula, a count); `null` in `window` / `min_points` means no fit window applies.
Scalar entries (`scalar_entry` in analyze.py) always carry `min_points: null`, even when they
name a `window` (the window then says which rows fed the recipe, not a per-seed fit rule).

`window` is `"full"` (all 60 targets, min_points 10), `"asymptotic"` (eps < 1e-2, min_points 5), or
`"below_floor"` (eps below the marginalised-inference noise floor sigma/sqrt(N_total); the four
`text.marg.alpha_below_floor*` keys only, min_points 5 borrowed from the asymptotic window); see
`eqsp/fit.py` for the fit convention (converged rows only, per-seed OLS, mean and SEM over seeds
retaining at least min_points converged targets).

## Key grammar

Cell tokens: `L{L}` (repetition block `[[2L+1, 1]]`: N_code = 2L+1, corrects L flips);
`g{gamma:.2f}` (g0.00, g0.01, g0.05, g0.10 in v2; `--v1` also g0.30, g0.45);
`s{sigma:.2f}` (s0.00, s0.01, s0.05, s0.10); `q{q:g}` (q0.001, q0.01, q0.03); `k{k:g}` (k0.99, k0.995,
k0.999); `F{F:g}` (F0.99, sequential prep runs, which set F only); estimator `mismatched` | `aware`
(absent for sequential, whose majority vote has no estimator).
Mode tokens: `edfl` (full-likelihood), `rej` (rejection sampling / post-selection), `marg`
(marginalised inference, offsets redrawn per shot), `margcap` (marg with M capped at
M_opt = ceil(1/(2 sigma sqrt(N_total)))).

Every table cell `<prefix>.<cell>[.<mode>]` carries the bundle
`alpha` (in the table's window), `alpha_full` or `alpha_asymptotic` (the other window),
`window_shift` (alpha_full minus alpha_asymptotic), `converged_pct`, `C` (prefactor: geometric mean of
T*eps over converged rows, full range; `sem` is of log C), and the log-law members
`logfit_a`, `logfit_b` (T eps = a + b ln(1/eps), full range), `logfit_rel_rms`, `powfit_rel_rms`
(relative RMS residual of the log law and of the power law on the same rows).

## Prefixes

| Prefix | Cells | Window of `.alpha` | Extra keys |
|---|---|---|---|
| `fig1.alpha.bare_{g}` | bare GHZ adaptive N, gamma in {0.01, 0.05, 0.10} (v2 `bare_adaptive.csv`; `--v1`: depolarizing `bare_ghz.csv`) | full | |
| `fig1.alpha.edfl_L1_g0.10` | ED-FL L=1, gamma=0.10 | full | |
| `fig2.alpha.bare_{g}`, `fig2.conv.bare_{g}` | bare GHZ adaptive N (incl. g0.00) | asymptotic | conv = converged_pct |
| `fig2.alpha.ed_L{L}_{g}_{edfl,rej}`, `fig2.conv.ed_...` | ED cells | asymptotic | |
| `fig3.alpha.combined_L{L}_{g}_s0.01` | combined ED-FL (1,0.01), (1,0.10), (2,0.10) | asymptotic | |
| `fig4.window.{s}.N_star`, `.N_star_half` | 1/(2 sigma^2), 1/(4 sigma^2) | n/a | formula |
| `fig4.window.{s}.N_argmin`, `.N_argmin_median` | sampled N minimising mean / median T (cells with >= 50 % convergence) | n/a | |
| `fig4.window.{s}.T_mean_N{N}`, `.T_median_N{N}`, `.converged_pct_N{N}` | per (sigma, N); T over converged seeds; `sem` of the mean | n/a | |
| `tab_prefactor.L{L}_{g}.bare_C`, `.bare_alpha_full`, `.bare_alpha_asym`, `.bare_window_shift`, `.bare_converged_pct` | bare GHZ fixed N = 2L+1 (`bare_fixed`), gamma in {0.01, 0.05, 0.10} (`--v1` also 0.30, 0.45) | full / asymptotic as named | |
| `tab_prefactor.L{L}_{g}.{edfl,rej}_C`, `.{edfl,rej}_alpha_asym`, `.{edfl,rej}_window_shift` | ED cells with a bare counterpart | asymptotic | |
| `tab_prefactor.L{L}_{g}.{edfl,rej}_over_bare`, `.bare_over_{edfl,rej}` | C ratios | n/a | |
| `tab_ed.L{L}_{g}.{edfl,rej}.<bundle>`, `.accept_pct` | tab:error-detected-results | asymptotic | |
| `tab_combined.L{L}_{g}_{s}.{edfl,rej,marg,margcap}.<bundle>`, `.accept_pct` | tab:combined-results (sigma 0.01 and 0.05; marg and margcap at L1 g0.01 s0.01; `--v1` also the cut sigma scan) | asymptotic | |
| `tab_sequential.L{L}_{s}.<bundle>` | tab:sequential-results | full | `alpha.n_seeds` is the Seeds column |
| `tab_bare_within_model.{g}.<bundle>` | bare GHZ adaptive N by gamma incl. g0.00 (replaces tab:scaling-results) | full | |
| `tab_realistic.readout.{bare,ed,ed_rej,combined,combined_rej}[_L{L}]_{q}_{estimator}.<bundle>`, `.F`; `tab_realistic.readout.sequential_L{L}_{q}.<bundle>` | readout-flip cells (q > 0) | asymptotic | |
| `tab_realistic.prep.{bare,ed,ed_rej,combined,combined_rej}[_L{L}]_{k}_{estimator}.<bundle>`, `.F`; `tab_realistic.prep.sequential_L{L}_{F}.<bundle>` | prep-infidelity cells (k < 1 or F < 1) | asymptotic | |
| `text.prefactor.bare_over_{edfl,rej}_{min,max}` | over the five tab:prefactor rows (L=1 x 3 gammas, L=2 and L=3 at gamma 0.10) | n/a | `_all` suffix: over all matched cells |
| `text.prefactor.bare_alpha_asym_range`, `.bare_conv_min` | bare fixed-N rows of the prefactor table | asymptotic / n/a | |
| `text.combined.prefactor_ratio_L{L}_{g}_{s}`, `.prefactor_ratio_{min,max}` | C_rejection / C_EDFL | n/a | |
| `text.combined.prefactor_ratio_edfl_over_bare_fixed_L{L}_{g}_{s}` | C_EDFL(combined) / C_bare(fixed N=2L+1) at same (L, gamma); probe sizes differ | n/a | |
| `text.combined.per_block_accept_pct_L{L}_{g}_{s}`, `.per_block_accept_gap_L1_g0.05_s0.01` | cube root of the joint rejection-sampling acceptance, percent; gap to the ED L=1 g0.05 acceptance | n/a | |
| `text.combined.alpha_asym_max_edfl` | largest asymptotic alpha over the combined ED-FL cells | asymptotic | |
| `text.ed.alpha_asym_{edfl,rej}_range`, `.alpha_asym_max_dev`, `.conv_edfl_min` | ranges over the ED table; largest deviation of an ED-FL asymptotic alpha from 1; smallest ED-FL convergence | asymptotic / n/a | |
| `text.sequential.M_exec_max_eps1e-2` | deepest executed round depth at the grid eps nearest 1e-2: v2 `M_exec_max` column; `--v1` fallback replays the deterministic schedule (note says which) | n/a | |
| `text.sequential.M_max_used_eps1e-2` | legacy column ceil((pi/2)/(2 eps sqrt N)) (only when the column exists), labelled not-the-executed-depth | n/a | |
| `text.sequential.M2sigma2_{s}_eps1e-2`, `.visibility_{s}_eps1e-2`, `.eps_quoted` | M_exec_max^2 sigma^2; exp(-M_exec_max^2 sigma^2 / 2); the eps grid point used | n/a | |
| `text.sequential.alpha_full_{noiseless,s0.01,s0.05}`, `.alpha_asym_{tag}_range`, `.conv_{tag}_range` | per sigma: full-window alpha (over L), [min, max] over L of the asymptotic alpha and of the convergence | full / asymptotic / n/a | |
| `text.sequential.alpha_schedule_range`, `.alpha_schedule_asym_range`, `.T_seed_spread_pct` | fits of the seed-mean cost at all 60 targets (the deterministic schedule); largest per-target seed spread of T | full / asymptotic / n/a | |
| `text.sequential.nonconv_error_over_eps_median_s0` | median error / eps over the non-converged noiseless rows (wrong-bit rounds) | n/a | |
| `text.marg.eps_floor`, `.eps_min`, `.M_cap`, `.M_max_{capped,uncapped}_min_eps`, `.eps_reach_{capped,uncapped}`, `.conv_{capped,uncapped}_min_eps`, `.alpha_below_floor_{capped,uncapped}`, `.alpha_below_floor_converged_{capped,uncapped}` | marginalised-inference cell (L1 g0.01 s0.01): noise floor, M_opt, depth reached, smallest eps converging on >= 90 % of seeds, convergence at eps_min, below-floor alpha (targets converging on >= 90 % of seeds / all converged targets) | below_floor | |
| `text.logfit.alpha_implied_{full,asymptotic}` | alpha a pure eps^-1 ln(1/eps) law fits as on the 60-point grid | full / asymptotic | formula |
| `text.logfit.bare_adaptive_g0_logfit_{a,b,rel_rms}`, `.bare_adaptive_g0_powfit_rel_rms` | the gamma = 0 bare adaptive cell's log law | full | |
| `text.logfit.{bare_fixed,edfl,rej,combfl}_{alpha_asym,b,rel_rms,pow_rel_rms}_range` | [min, max] over the cells of each family | full (b, rms) / asymptotic (alpha) | |
| `text.fisher_saturation.{edfl_L{L}_{g},combined_L{L}_{g}_{s}}.{mean,seed_min}` | per-cell saturation ratio sum_F_obs / (sum_M_sq 4 N^2 (1-(gamma/omega)^2)), per-seed mean on eps < 1e-2 | asymptotic | |
| `text.fisher_saturation.{edfl,combined}_{min,max}`, `.min` | over table cells (combined: sigma in {0.01, 0.05}); `.min` is the smallest per-seed mean | asymptotic | |
| `text.weak_noise_map.gamma_ham_p{p}` | sqrt(p) omega / sin(omega) at omega = 0.3 for p in {0.01, 0.05, 0.1} | n/a | formula |
| `text.readout.n_breakdown_{q}` | floor(-1 / ln(1 - 2q)), the parity size at which (1-2q)^n = 1/e (approx. 1/(2q)) | n/a | formula |
| `text.prep.n_star_{k}` | -1/ln k | n/a | formula |
| `text.fit.window_shift_max` | largest abs(alpha_full - alpha_asymptotic) over ED and combined (ED-FL, rejection) cells | n/a | note names the cell |
| `text.fit.window_shift_max_{protocol}` | same per protocol (bare_adaptive / bare_depol, bare_fixed, ed, combined, sequential); marg modes excluded | n/a | |
| `text.fit.window_shift_{edfl,combfl}_range` | [min, max] of the window shift over ED-FL / combined ED-FL cells | n/a | |

## Companion files

`derived/cells.csv` (one row per cell, both windows, prefactor, acceptance), `derived/hwindow_summary.csv`,
`derived/fisher.csv`, `tables/*.tex` (rows in the paper's column order: tab_prefactor, tab_error_detected,
tab_combined, tab_sequential, tab_bare_within_model, tab_bare_fixed, tab_realistic_readout,
tab_realistic_prep, tab_window), and `compare_v1.md` with `--compare v1` (flag rule: |diff| > 2 sqrt(sem_v2^2 + sem_v1^2)).
