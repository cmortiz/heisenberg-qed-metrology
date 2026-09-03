# Data and reproducibility manifest

Every number, table row, and figure in the numerics section of *Encoded Quantum Signal
Processing for Heisenberg-Limited Metrology* is a key in `results/v2/numbers.json`,
written by `scripts/analyze.py` from the merged suite CSVs `results/v2/<protocol>.csv`.
The paper reads only `numbers.json` (via `scripts/paper_numbers.py`); `scripts/plot.py`
takes every printed alpha from it and performs no fit of its own. Key names follow
`results/numbers_schema.md`, which is the source of truth for the grammar below.

## Pipeline

```
scripts/suite.py  ->  results/v2/shards/<protocol>/<config_key>_seed<S>.csv   (one shard per run; gitignored)
suite.py --merge  ->  results/v2/<protocol>.csv                                (sorted, deduplicated)
scripts/analyze.py -> results/v2/numbers.json, tables/*.tex, derived/{cells,hwindow_summary,fisher}.csv
scripts/plot.py    -> ../figures/fig{1..5}_*.pdf                                (alphas read from numbers.json)
```

Same pipeline on the committed v1 data: `scripts/analyze.py --v1` reads
`scripts/results/*.csv` (legacy headers) and writes `scripts/results/{numbers.json,tables/,derived/}`;
`scripts/analyze.py --compare v1` additionally writes `results/v2/compare_v1.md`, flagging every cell whose
v2 alpha differs from its v1 value by more than $2\sqrt{\mathrm{sem}_{v2}^2 + \mathrm{sem}_{v1}^2}$.

## Paper element -> numbers.json -> CSV

`analyze` below means `uv run python scripts/analyze.py` (v2) or `... --v1` (committed data). The window
and `min_points` are those of the element's `.alpha` entries; every entry also carries `window`, `min_points`,
`source_csv`, `n_seeds`, and `n_rows`, so the table is checkable against the file itself.

| Paper element (`sensing.tex` label) | numbers.json key prefix | Source CSV (`results/v2/`) | Window, min_points | Command |
|---|---|---|---|---|
| Fig. 1 `fig:error-detection` (bare GHZ adaptive N at $\gamma \in \{0.01, 0.05, 0.10\}$; ED-FL L=1 at $\gamma = 0.10$) | `fig1.alpha.bare_{g}`, `fig1.alpha.edfl_L1_g0.10` | `bare_adaptive.csv`, `ed.csv` | full, 10 | analyze; `plot.py --figs fig1` -> `fig1_error_detection` |
| Table `tab:prefactor-results` (prefactor C, bare fixed N vs ED-FL and rejection, ratios) | `tab_prefactor.L{L}_{g}.bare_C`, `.{edfl,rej}_C`, `.{edfl,rej}_over_bare`, `.bare_over_{edfl,rej}`, `.bare_alpha_{full,asym}`, `.{edfl,rej}_alpha_asym`, `.*_window_shift`; `text.prefactor.bare_over_{edfl,rej}_{min,max}` | `bare_fixed.csv`, `ed.csv` | C: full range, pooled over converged rows (no min_points); bare alpha: full, 10 and asymptotic, 5; ED alpha: asymptotic, 5 | analyze -> `tables/tab_prefactor.tex` |
| Fig. 4 `fig:window` (Heisenberg window, T(N) at $\sigma_\varepsilon \in \{0.05, 0.10\}$) | `fig4.window.{s}.N_star`, `.N_star_half`, `.N_argmin`, `.N_argmin_median`, `.T_mean_N{N}`, `.T_median_N{N}`, `.converged_pct_N{N}` | `hwindow.csv` | none (T mean and median over converged seeds at each N) | analyze -> `derived/hwindow_summary.csv`, `tables/tab_window.tex`; `plot.py --figs fig4` -> `fig4_heisenberg_window` |
| Table `tab:bare-within-model` (bare GHZ adaptive N by $\gamma$; replaces the depolarizing `tab:scaling-results`) | `tab_bare_within_model.{g}.alpha`, `.alpha_asymptotic`, `.window_shift`, `.converged_pct`, `.C` | `bare_adaptive.csv` | full, 10 | analyze -> `tables/tab_bare_within_model.tex` |
| Fig. 2 `fig:inference-comparison` (asymptotic alpha and Converged % bars, bare and ED L=1,2,3, PS vs FL) | `fig2.alpha.bare_{g}`, `fig2.conv.bare_{g}`, `fig2.alpha.ed_L{L}_{g}_{edfl,rej}`, `fig2.conv.ed_...` | `bare_adaptive.csv`, `ed.csv` | asymptotic, 5 (conv over all 60 targets) | analyze; `plot.py --figs fig2` -> `fig2_inference_comparison` |
| Table `tab:error-detected-results` | `tab_ed.L{L}_{g}.{edfl,rej}.alpha`, `.alpha_full`, `.window_shift`, `.converged_pct`, `.C`, `.accept_pct` | `ed.csv` | asymptotic, 5 | analyze -> `tables/tab_error_detected.tex` |
| Table `tab:combined-results` ($\sigma_\varepsilon \in \{0.01, 0.05\}$; ED-FL, rejection, marg) and the quenched-vs-marginalized paragraph (`text.marg.*`: noise floor, $M^{\mathrm{opt}}$, below-floor alpha for the uncapped and capped runs) | `tab_combined.L{L}_{g}_{s}.{edfl,rej,marg,margcap}.alpha`, `.alpha_full`, `.window_shift`, `.converged_pct`, `.C`, `.accept_pct` | `combined.csv` | asymptotic, 5 | analyze -> `tables/tab_combined.tex` |
| Fig. 3 `fig:combined-protocol` (combined ED-FL at (L=1, 1%), (L=1, 10%), (L=2, 10%), $\sigma = 0.01$) | `fig3.alpha.combined_L{L}_{g}_s0.01` | `combined.csv` | asymptotic, 5 | analyze; `plot.py --figs fig3` -> `fig3_combined_protocol` |
| Table `tab:sequential-results` | `tab_sequential.L{L}_{s}.alpha` (`.n_seeds` is the Seeds column), `.alpha_asymptotic`, `.window_shift`, `.converged_pct`, `.C` | `sequential.csv` | full, 10 | analyze -> `tables/tab_sequential.tex` |
| Table `tab:realistic-noise` and Fig. 5 (readout flips q; prep infidelity k, F; mismatched vs aware) | `tab_realistic.readout.{bare,ed,ed_rej,combined,combined_rej,sequential}[_L{L}]_{q}_{estimator}.alpha`, `.converged_pct`, `.C`; `tab_realistic.prep.{...}[_L{L}]_{k}_{estimator}.*`; `text.readout.n_breakdown_{q}`; `text.prep.n_star_{k}` | rows with `q > 0` or `k < 1` or `F < 1` in `bare_adaptive.csv`, `ed.csv`, `combined.csv`, `sequential.csv` | asymptotic, 5 | analyze -> `tables/tab_realistic_{readout,prep}.tex`; `plot.py --figs fig5` -> `fig5_realistic` |
| Text: combined prefactor ratios ("3 to 26x"), per-block acceptance | `text.combined.prefactor_ratio_L{L}_{g}_{s}`, `.prefactor_ratio_{min,max}`, `.prefactor_ratio_edfl_over_bare_fixed_L{L}_{g}_{s}`, `.per_block_accept_pct_L{L}_{g}_{s}` | `combined.csv`, `bare_fixed.csv` | C: full range pooled; ratios n/a | analyze |
| Text: sequential executed depth and $M^2\sigma^2$ at $\varepsilon \approx 10^{-2}$ | `text.sequential.M_exec_max_eps1e-2`, `.M_max_used_eps1e-2` (v1 only), `.M2sigma2_{s}_eps1e-2`, `.eps_quoted` | `sequential.csv` (`M_exec_max` column; `--v1` replays the deterministic schedule) | n/a | analyze |
| Text: Fisher saturation | `text.fisher_saturation.{edfl_L{L}_{g},combined_L{L}_{g}_{s}}.{mean,seed_min}`, `.{edfl,combined}_{min,max}`, `.min` | `ed.csv`, `combined.csv` (FL rows; `sum_F_obs`, `sum_M_sq`) | asymptotic rows ($\varepsilon < 10^{-2}$), per-seed mean | analyze -> `derived/fisher.csv` |
| Text: weak-noise map $\gamma_{\rm Ham} = \sqrt{p}\,\omega/\sin\omega$ | `text.weak_noise_map.gamma_ham_p{p}` | none (formula at $\omega = 0.3$) | n/a | analyze |
| Text: fit-window shift (replaces the WLS sentence) | `text.fit.window_shift_max`, `.window_shift_max_{protocol}` | all protocol CSVs | full minus asymptotic per cell | analyze -> `derived/cells.csv` |
| Text: Heisenberg scaling with logarithmic overhead (seed-mean $T\varepsilon = a + b\ln(1/\varepsilon)$ over all converged targets, against a pure power law on the same averaged data) | per cell `<prefix>.logfit_{a,b,rel_rms}`, `.powfit_rel_rms`; `text.logfit.alpha_implied_{full,asymptotic}` (exponent a power-law fit assigns to $\ln(1/\varepsilon)/\varepsilon$ on the 60-target grid), `text.logfit.{edfl,rej,bare_fixed,combfl}_{rel_rms,pow_rel_rms,b,alpha_asym}_range`, `text.logfit.bare_adaptive_g0_{logfit_a,logfit_b,logfit_rel_rms,powfit_rel_rms}` | all Bayesian CSVs | full range, seed-mean T per target | analyze -> `derived/cells.csv` |

Table bodies are pasted from `tables/*.tex`; `tables/tab_bare_fixed.tex` (every `bare_fixed` cell) is a companion, not a paper table.

## Fit convention (`eqsp/fit.py`)

Per cell (a group of `GROUP_KEYS`) and per seed: ordinary least squares of $\log T$ on $\log\varepsilon$ with an
intercept over the rows with `converged == True` inside the window, $\alpha_{\rm seed} = -\text{slope}$, no weighting.
Across seeds: $\alpha$ = mean, sem = std(ddof=1)/$\sqrt{n}$, ci95 = $t_{0.975,n-1}$ sem. Exactly two windows:

| Window | Targets | min_points | Elements |
|---|---|---|---|
| full | all 60 ($\varepsilon \in [10^{-4}, 10^{-1}]$) | 10 | Fig. 1, `tab:bare-within-model`, `tab:sequential-results`, bare rows of `tab:prefactor-results`, Fig. 4 |
| asymptotic | $\varepsilon < 10^{-2}$ | 5 | Fig. 2, Fig. 3, Fig. 5, `tab:error-detected-results`, `tab:combined-results`, `tab:realistic-noise`, ED rows of `tab:prefactor-results`, Fisher saturation |

**Seeds retained.** A seed enters a fit only if it has at least `min_points` converged targets inside the window;
`n_seeds` is the retained count and can be below the 40 seeds run (the Seeds column of `tab:sequential-results`
at $\sigma_\varepsilon = 0.05$). **Converged (%)** is always the converged fraction over every row of the cell
(all 60 targets times all seeds), never over the window subset. **Window shift** is $\alpha_{\rm full} - \alpha_{\rm asymptotic}$
per cell. **Prefactor C** is the geometric mean of $T\varepsilon$ over the converged rows of the cell pooled across
seeds over the full range; its `sem` is the standard error of $\log C$. Rows are deduplicated on
(cell, seed, epsilon) keeping the last row before any fit, the same rule as `suite.py --merge`.

## v2 CSV files (`results/v2/<protocol>.csv`)

`bare_adaptive.csv`, `bare_fixed.csv`, `ed.csv`, `combined.csv`, `sequential.csv`, `hwindow.csv`, produced by
`suite.py --merge` from the shards. One column set for every protocol (`eqsp/config.py` `CSV_COLUMNS`), NaN
where not applicable, floats as `%.10e`:

```
protocol, mode, seed, epsilon, gamma, h, sigma_epsilon, L, N_code, N_total, N, q, k, F, estimator, model,
M_max, M_exec_max, grid_m, total_resources, converged, final_error,
n_rounds, n_updates, n_accepted, n_rejected, acceptance_rate,
n_syndrome_used, mean_d, sum_F_obs, sum_M_sq, visibility, stop_rule, posterior_std,
engine_version, jax_version, timestamp
```

Grouping keys (`GROUP_KEYS`): `protocol, mode, gamma, h, sigma_epsilon, L, q, k, F, estimator, model, stop_rule`. `mode` keeps the
legacy labels (`bare_hamiltonian`, `ed_full_likelihood`, `ed_postselect`, `combined_full_likelihood`,
`combined_postselect`, `combined_full_likelihood_marg`, `sequential`) plus `bare_hamiltonian_adaptive`, `hwindow`, and
`combined_full_likelihood_marg_capped` (marg with the depth capped at $M^{\mathrm{opt}} = 1/(2\sigma_\varepsilon\sqrt{N_{\mathrm{total}}})$; `--marg-cap`).
The alpha fit reads only `seed, epsilon, total_resources, converged`. `--merge` sorts by the grouping keys plus
`seed, epsilon` (plus `N` for hwindow) and `timestamp`, then deduplicates on those keys keeping the last row.
`engine_version` is the numerics git short SHA (`-dirty` if the tree had uncommitted tracked changes),
`jax_version` is `jax.__version__` (`none` for `sequential`).

## Committed v1 data (`scripts/results/`, pre-rerun)

Produced by the legacy scripts at tag `v1-jax-suite` (GPU float32 XLA, May 2026); read by `analyze.py --v1`,
`plot.py --v1`, `pytest -m regression`, and `checks/spot_check_legacy.py`. 40 seeds, 60 targets per cell.

| File | Rows | Cells | Content | Read by `analyze.py --v1` |
|---|---|---|---|---|
| `bare_ghz.csv` | 16,920 | 7 | depolarizing bare GHZ, adaptive N (`mode = ghz`): $\gamma \in \{0, 0.01, 0.05, 0.10, 0.15, 0.20\}$ at $h = 0$, and $\gamma = 0.05$ at $h = 0.3$ | yes (as `bare_depol`; the $\gamma = 0$ arm is the engine check) |
| `bare_hamiltonian.csv` | 26,400 | 11 | bare GHZ, Hamiltonian noise, fixed N = 2L+1: $L \in \{1,2,3\} \times \gamma \in \{0.01, 0.05, 0.10\}$, plus $\gamma \in \{0.30, 0.45\}$ at L=2 | yes (`bare_fixed`) |
| `ed_full_likelihood.csv` | 21,660 | 9 | ED full likelihood (`mode = full_likelihood`), $L \in \{1,2,3\} \times \gamma$ | yes (`ed`) |
| `ed_postselect.csv` | 21,660 | 9 | ED rejection sampling (`mode = error_detected`) | yes (`ed`) |
| `combined_full_likelihood.csv` | 38,400 | 16 | combined ED-FL: 12 table cells plus the cut $\sigma_\varepsilon \in \{0.10, 0.15, 0.20, 0.30\}$ scan at L=2, $\gamma = 0.05$ | yes (`combined`) |
| `combined_postselect.csv` | 28,800 | 12 | combined rejection sampling (`mode = combined`) | yes (`combined`) |
| `combined_full_likelihood_marg.csv` | 28,800 | 12 | combined, per-shot offsets, marginalised inference | yes (`combined`) |
| `sequential.csv` | 21,600 | 9 | sequential binary search, $L \in \{1,2,3\} \times \sigma_\varepsilon \in \{0, 0.01, 0.05\}$; column `M_max_used` is the analytic bound, not the executed depth | yes (`sequential`) |
| `combined_full_likelihood_marg_qt.csv` | 6,480 | 4 (partial) | quenched-truth marg variant; fed the quenched-vs-marginalised aside only | no |
| `combined_full_likelihood_cond_only.csv` | 9,600 | 4 | cut cond-only ablation | no |
| `ed_full_likelihood_cond_only.csv` | 9,600 | 4 | cut cond-only ablation | no |
| `analysis_summary.csv` | 7 | | legacy summary table | no |

Legacy headers (no `protocol`, `q`, `k`, `F`, `estimator`, `model`, `engine_version`, `jax_version`;
`analyze.py --v1` fills the missing grouping keys with `q = 0, k = F = 1, estimator = mismatched, model = legacy`):

```
bare_ghz:            seed,gamma,h,epsilon,mode,N,total_resources,converged,final_error,timestamp
bare_hamiltonian:    seed,gamma,h,L,N_code,epsilon,mode,total_resources,converged,final_error,n_rounds,timestamp
ed_*:                seed,gamma,h,L,N_code,epsilon,mode,total_resources,converged,final_error,n_accepted,n_rejected,
                     acceptance_rate,n_syndrome_used,mean_d,sum_F_obs,sum_M_sq,timestamp
combined_*:          seed,gamma,h,sigma_epsilon,L,N_code,N_total,epsilon,mode,total_resources,converged,final_error,
                     n_accepted,n_rejected,acceptance_rate,n_syndrome_used,mean_d,sum_F_obs,sum_M_sq,timestamp
sequential:          seed,gamma,sigma_epsilon,L,N_code,epsilon,mode,total_resources,converged,final_error,n_rounds,
                     M_max_used,timestamp
```

**Known v1 duplicates.** `ed_full_likelihood.csv` and `ed_postselect.csv` carry seed 1 of (L=1, $\gamma = 0.01$)
twice (identical rows: alpha unchanged, pooled prefactor 197.4 without the duplicate against the 198.7 the paper
rounded to 199); `bare_ghz.csv` carries $\gamma = 0$ seeds 1 and 2 twice with different resources (full-range alpha
1.151 with the duplicates, 1.152 without). None of the legacy analyzers deduplicated; `eqsp/fit.py` and
`suite.py --merge` deduplicate on the grouping keys plus `seed` and `epsilon`, keeping the last row.

The committed `scripts/results/numbers.json`, `scripts/results/tables/*.tex`, and `scripts/results/derived/{cells,fisher}.csv`
are the output of `analyze.py --v1` on these files (742 keys) and are the pre-rerun provenance of the paper's numbers.

## Not in this repository

Shards and logs (`results/v2/shards/`, `results/v2/logs/`, `scripts/logs/`), the local `archive/`, and the legacy
per-analyzer summaries listed in `.gitignore` are regenerable or provenance-only and are not needed to recreate any
paper element. The v1 scripts themselves are preserved at git tag `v1-jax-suite`.
