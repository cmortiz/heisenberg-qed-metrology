# EQSP numerics

Bayesian phase-estimation simulations for the paper *Encoded Quantum Signal
Processing for Heisenberg-Limited Metrology* (`../sensing.tex`). One grid-Bayesian
engine, six protocol adapters, one runner, one analyzer, one plotter. Every number
printed in the paper's numerics section is a key in `results/v2/numbers.json`
written by `scripts/analyze.py`; see `DATA.md` for the element-by-element map.

## Install

Python 3.12 via [uv](https://docs.astral.sh/uv/); never `pip`.

```bash
uv sync --extra test                    # macOS / CPU (JAX CPU backend)
uv sync --extra cuda --extra test       # CUDA 13 GPU server (jax[cuda13])
```

`sequential` never imports JAX; `--device cpu` sets `JAX_PLATFORMS=cpu` before the
first JAX import. All commands below run from this directory.

## Layout

```
pyproject.toml                  deps numpy, pandas, scipy, matplotlib, tqdm, jax>=0.9; extras cuda, test
eqsp/config.py                  EngineConfig, RunSpec (gamma, h, L, sigma, inference, marg, q, k, F, estimator, model),
                                OMEGA_TRUE = 0.3, epsilon_grid, grid_resolution, window_N_grid, CSV_COLUMNS, GROUP_KEYS
eqsp/io.py                      eps_rng / seed_rng / stream_rng, config_key, shard_path, make_row, write_rows (flock), git_short_sha
eqsp/engine.py                  make_grid, init_log_posterior, log_update, map_error, GridCache, Counters, run_epsilon, run_seed(s)
eqsp/closed_forms.py            float64 NumPy reference likelihoods (the physics contract; see below)
eqsp/hamiltonian.py             float32 JAX kernels: legacy-verbatim theorem kernels and the new exact-model kernels
eqsp/fit.py                     the one OLS alpha fit (two windows), prefactor, window shift, converged %
eqsp/protocols/bare_adaptive.py bare GHZ, Hamiltonian noise, N = clip(floor(1/eps), 10, 16384), n ~ U{1..N}, M = 1
eqsp/protocols/bare_fixed.py    bare GHZ, Hamiltonian noise, fixed odd N = 2L+1, M ~ U{1..M_max}
eqsp/protocols/ed.py            error-detected GHZ, one [[2L+1, 1]] repetition block, inference fl | ps
eqsp/protocols/combined.py      logical GHZ over three blocks, per-qubit offsets, inference fl | ps | marg (--marg-cap caps M at ceil(1/(2 sigma sqrt(N_total))))
eqsp/protocols/sequential.py    sequential binary search with product-state probes (NumPy only)
eqsp/protocols/hwindow.py       Heisenberg-window sweep T(N) at fixed eps_tar with shot-to-shot offsets
scripts/run.py                  CLI: one protocol, one or more seeds, one shard per seed
scripts/suite.py                suite table (groups 1-8), GPU and CPU pools, resume, --quick, --dry-run, --merge
scripts/analyze.py              cells.csv, numbers.json, tables/*.tex, fisher.csv, hwindow_summary.csv, compare_v1.md
scripts/plot.py                 paper figures; every printed alpha read from numbers.json
scripts/paper_numbers.py        fills the paper's \NUM{key} placeholders from numbers.json
scripts/results/                committed v1 data (legacy CSVs) plus the v1 numbers.json, tables/, derived/
results/numbers_schema.md       numbers.json key grammar (source of truth for key names)
results/v2/                     suite output: shards/ (gitignored), <protocol>.csv, numbers.json, tables/, derived/
tests/                          conftest.py, test_closed_forms_pennylane.py, test_rng_streams.py, test_engine.py, test_regression.py
checks/                         one-off kernel and legacy spot checks (check_kernels.py, spot_check_*.py); not tests
archive/                        local provenance only (gitignored)
```

## Physics

`eqsp/closed_forms.py` is the reference. Every likelihood the engine uses is derived
there in float64 NumPy from the circuit $U_k = \exp(-iM(\omega Z_k + \gamma_k X_k))$
on a GHZ state, followed by $Z_i Z_{i+1}$ syndromes within each repetition block and
an $X^{\otimes N}$ parity readout. `tests/test_closed_forms_pennylane.py` checks these
functions against an independent PennyLane state-vector circuit; the float32 JAX
kernels in `eqsp/hamiltonian.py` must agree with them.

- **Repetition-code GHZ (ED, combined), `model=exact` (default).** The exact circuit
  likelihood is a classical mixture over the two error patterns consistent with each
  block's syndrome (the pattern and its logical-X coset partner, relative probability
  weight $(p/(1-p))^{N-2d}$), `ghz_code_probs` / `ghz_code_probs_mixture`.
- **`model=theorem`.** Reproduces the closed form of Theorem `thm:bitflip` and the
  legacy simulators (minimum-weight pattern only), `ghz_code_probs_theorem`. Used
  by the regression diagnostic; not for production numbers.
- **Bare GHZ parity.** $P(\pm) = \tfrac12\,[1 \pm (r^n \cos n\theta + [n\ \text{even}]\, C^n)]$,
  `bare_parity`; the even-$n$ population term $C^n$ is kept.
- **Readout flips.** Each measured bit flips with probability $q$; the parity
  visibility is $(1-2q)^n$ (`readout_visibility`); the q-aware estimator sums over
  patterns consistent with the observed syndrome (`qaware_probs`).
- **Preparation infidelity.** Visibility $V = k^{n-1} F$ (`prep_visibility`) scales
  the coherent term only, never the population term.
- **Sequential.** Product-state probes, majority vote; prep flip $(1-F)/2$ per
  qubit, readout flip $q$ per outcome.
- **hwindow.** $P(+) = \tfrac12\,[1 + \lambda_N \cos(2N\omega - \theta_i)]$ with
  $\lambda_N = e^{-2N\sigma^2}$, per-shot readout phase $\theta_i$, grid on one
  period $[0, \pi/N)$.

## Engine contract

1. Float32 log-space grid posterior on $2^m$ points, `jax_enable_x64` off. The prior
   support is one period of the coarsest ($M = 1$) parity fringe: $[0, \pi)$ for
   `bare_adaptive` ($n = 1$ shots), and a window of width $\pi/N$ centred on
   $\omega$ for `bare_fixed`, `ed` ($N = N_{\rm code}$) and `combined`
   ($N = N_{\rm total}$); `hwindow` uses $[0, \pi/N)$ with the truth wrapped. Every
   GHZ shot carries a preparation phase $\theta_i \sim U[0, 2\pi)$ known to the
   estimator, $(|0\cdots0\rangle + e^{-i\theta_i}|1\cdots1\rangle)/\sqrt2$, so the
   parity likelihood is $\propto \cos(2\Phi - \theta_i)$ and the sign of the phase is
   identifiable. Fixed-axis parity alone is even in the phase and $\pi/N$-periodic up
   to $O(\gamma^2/\Omega^2)$ amplitude terms, so the legacy $[0, 2\pi)$ prior carried
   near-degenerate mirror modes (v1 non-converged `bare_hamiltonian` rows cluster at
   discrete errors: $3.138 \approx \pi$ (the parity mirror at $\omega + \pi$) and
   $2.095 = 2\pi/3$, $1.257 = 2\pi/5$, $0.449 = \pi/7$, multiples of the $\pi/N$ period at
   $N = 3, 5, 7$) and the v1 depolarizing bare
   likelihood $\cos(2n\omega)$ was exactly bimodal on $[0, \pi)$; the truth-based
   legacy stop masked both (finding 2026-09-02). `model=theorem` keeps the legacy
   grids and no phase for the regression diagnostic.
2. Per-target RNG stream `default_rng(seed + int(eps*1e8) % 2**31)` (`eps_rng`);
   new draws (readout, prep, hwindow) use `stream_rng(seed, eps, *extra)` and are
   appended after the legacy draws, so `q = 0, k = F = 1` reproduces legacy streams.
3. Stopping rule (`--stop posterior`, the default for every Bayesian protocol):
   after every accepted update the engine computes the posterior circular std
   about the MAP and stops when the 95% credible half-width `2 * std <= eps`.
   The run is `converged` iff the MAP circular error at that stop is `< 1.2 * eps`.
   The rule never reads the truth, so `T` is what a real experiment would spend.
   Exhausting `max_experiments` (10000 bare_adaptive, 50000 bare_fixed/ed/combined,
   2000000 hwindow) is never "converged". `posterior_std` at the stop is a CSV
   column. `--stop legacy` reproduces the v1 scripts: stop at the first cadence
   check (item 5) whose truth-based MAP error is `< 1.2 * eps`. The legacy rule
   stops on lucky MAP hits while the posterior is still wide (in v1 `bare_fixed`
   at gamma = 0.10, 91% of runs stop at the first check), which makes `T` depend
   on the check cadence and biases the prefactor comparison between protocols
   with different cadences. It is kept only for the regression diagnostic
   (`tests/test_regression.py`, `checks/`), keyed `..._legacy` in the shard name.
4. Grid rule `m = max(16, ceil(log2(4 N floor(1/eps_min))))` (`grid_resolution`) for
   `bare_fixed`, `ed`, `combined`; `m = 16` for `bare_adaptive` and `hwindow`.
5. Legacy check cadence (`--stop legacy` only): `bare_adaptive` every 100 rounds
   including round 0; `bare_fixed` every 100 rounds from round 1; FL every 10
   updates; PS every 100 accepted shots; `hwindow` every 10 shots (its posterior-stop
   device loop checks every update). Rejected rounds are
   charged but never update or check, under either rule.

Numerical guard (finding of the v2 full run): a syndrome probability is clipped to `[1e-30, P_MAX]` with
`P_MAX = 1 - 1e-7` before `log(1 - p)`. The earlier `1 - 1e-30` is a no-op in float32, and where the prior
window crosses omega = 0 (ED with N <= 5) `p = sin^2(M Omega) gamma^2 / Omega^2` reaches exactly 1, so the
w = N term produced `0 * (-inf) = NaN` and a NaN posterior (36 exhausted rows at gamma = 0.10, L <= 2,
eps <= 1.8e-4; rerun after the fix). Rows without a saturated grid point are bitwise unchanged.


## Suite (`scripts/suite.py --dry-run`)

40 seeds, 60 log-spaced targets $\varepsilon \in [10^{-4}, 10^{-1}]$, $\omega = 0.3$.

| Group | Protocol and grid | Configs |
|---|---|---|
| 1 | `bare_adaptive`, $\gamma \in \{0, 0.01, 0.05, 0.10\}$ | 4 |
| 2 | `bare_fixed`, $L \in \{1,2,3\} \times \gamma \in \{0.01, 0.05, 0.10\}$ | 9 |
| 3 | `ed`, same grid $\times$ {fl, ps} | 18 |
| 4 | `combined`, $L \in \{1,2\} \times \gamma \times \sigma_\varepsilon \in \{0.01, 0.05\} \times$ {fl, ps}, plus marg and marg-capped at $(1, 0.01, 0.01)$ | 26 |
| 5 | `sequential`, $L \in \{1,2,3\} \times \sigma_\varepsilon \in \{0, 0.01, 0.05\}$ | 9 |
| 6 | `hwindow`, $\sigma_\varepsilon \in \{0.05, 0.10\}$, $\varepsilon_{\rm tar} = 4 \times 10^{-2}\sigma_\varepsilon^2$ ($10^{-4}$, $4 \times 10^{-4}$), 12 N values in $[N^*/16, 4N^*]$ | 2 |
| 7 | readout $q \in \{10^{-3}, 10^{-2}, 3\cdot10^{-2}\} \times$ {mismatched, aware} $\times$ {bare_adaptive $\gamma{=}0.10$, ed-fl $L{=}1..3$, combined-fl $L{=}1,2$ $\sigma{=}0.01$}, plus sequential $L{=}1..3$ (q only) | 45 |
| 8 | prep $k \in \{0.99, 0.995, 0.999\}$, $F = 0.99$, same protocols and estimators; sequential $F$ only | 39 |

152 configs $\times$ 40 seeds = 6,080 jobs, 360,960 rows (the two window configs
produce 12 rows per seed instead of 60). `--quick` runs 4 seeds and one config per group.

### Running

```bash
# one protocol, one seed (shard at results/v2/shards/<protocol>/<config_key>_seed<S>.csv)
uv run python scripts/run.py ed --seed 1 --gamma 0.05 --L 1 --inference fl
uv run python scripts/run.py combined --seeds 1-4 --gamma 0.10 --L 2 --sigma-eps 0.01 --q 0.01 --estimator aware
uv run python scripts/run.py hwindow --seed 1 --sigma-eps 0.05 --eps-tar 1e-3 --N-grid auto
uv run python scripts/run.py sequential --seed 1 --L 1 --sigma-eps 0.01 --device cpu

# quick suite (smoke test), full suite, resume, merge
uv run python scripts/suite.py --quick
nohup uv run python scripts/suite.py --label full > scripts/logs/full.out 2>&1 &
uv run python scripts/suite.py --label full          # resume: existing shards are skipped
uv run python scripts/suite.py --merge               # shards -> results/v2/<protocol>.csv
uv run python scripts/suite.py --only 1,6 --seeds 1-10
```

GPU pool: 22 slots (11 per card on a 2-GPU server), each with
`XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.08`,
`CUDA_VISIBLE_DEVICES = slot % n_gpus`, a shared `JAX_COMPILATION_CACHE_DIR`, and one
warm-up process per GPU per distinct (protocol, grid m) before the pool opens. CPU
pool: 64 slots, `OMP_NUM_THREADS=1`, `--device cpu`; `sequential` is routed there by
default (`--cpu-protocols`). Without GPUs everything runs in the CPU pool. Logs:
`scripts/logs/<label>.log`, per-job logs in `scripts/logs/<label>/`, exit codes in
`scripts/logs/<label>.joblog.tsv`. `--merge` concatenates shards per protocol, sorts,
and deduplicates on `GROUP_KEYS + seed + epsilon` (plus `N` for hwindow), keeping the
last row.

## Analysis and figures

```bash
uv run python scripts/analyze.py                 # results/v2 -> numbers.json, tables/*.tex, derived/*.csv
uv run python scripts/analyze.py --v1            # committed scripts/results/*.csv (legacy headers)
uv run python scripts/analyze.py --compare v1    # also results/v2/compare_v1.md (flag: |diff| > 2 sqrt(sem_v2^2 + sem_v1^2))
uv run python scripts/plot.py                    # results/v2 -> ../figures/fig{1..5}_*.{pdf,png}
uv run python scripts/plot.py --v1 --out /tmp/v1figs --figs fig1 fig2 fig3
```

Fit convention (`eqsp/fit.py`): per-seed OLS of $\log T$ on $\log\varepsilon$ over
converged rows; mean and SEM over seeds retaining at least `min_points` converged
targets in the window. Two windows only: **full** (all targets, `min_points` 10) for
bare GHZ, sequential, and the window sweep; **asymptotic** ($\varepsilon < 10^{-2}$,
`min_points` 5) for ED, combined, and realistic noise. Converged (%) is always over
all 60 targets of every seed. Every `numbers.json` entry carries `window`,
`min_points`, `source_csv`, `n_seeds`. `plot.py` performs no fit: slopes, intercepts,
and CI widths come from `numbers.json`; the CSVs supply only the data markers.
Outputs: `fig1_error_detection`, `fig2_inference_comparison`, `fig3_combined_protocol`,
`fig4_heisenberg_window`, `fig5_realistic`.

Besides the two power-law windows, `analyze.py` fits the seed-mean cost of every cell as
`T * eps = a + b ln(1/eps)` (Heisenberg scaling with a logarithmic overhead) and reports the RMS relative
residual of that model and of a pure power law on the same averaged data (`derived/cells.csv`,
`text.logfit.*`). On the 60-target grid a pure `ln(1/eps)/eps` law fits as alpha = 1.19 (full range) and
1.15 (asymptotic); the v2 Bayesian cells sit there, the sequential binary search (geometric schedule) at 1.00.


## Tests

```bash
uv run pytest                  # 189 tests (uv run pytest --co -q per file for the split): closed forms vs
                               # PennyLane (139), RNG replay (17), engine (32), always-on exact
                               # reproduction of committed sequential.csv rows (1)
uv run pytest -m regression    # diagnostic tiers against scripts/results/*.csv; writes
                               # results/v2/regression_report.md; reported, never gating
```

The regression exact tier reruns seeds 1 and 2 of one cell per protocol with
`--model theorem` at the three largest targets and compares `total_resources,
converged, final_error`; bitwise agreement is expected only for `sequential` (pure
NumPy), since the committed JAX rows came from a GPU float32 build. The statistical
tier (alpha within 2 SEM, converged fraction within 2 points) runs once merged v2
CSVs exist. `checks/` holds the one-off kernel checks used during the rewrite.

## Provenance

Every CSV row carries `engine_version` (numerics git short SHA, `-dirty` when the
tree has uncommitted tracked changes) and `jax_version` (`jax.__version__`, `none`
for `sequential`). Floats are written `%.10e` so `gamma` and `sigma_epsilon`
round-trip to the doubles the analyzer groups on.

## Legacy

The v1 scripts (`simulation*.py`, `analyze_*.py`, `plot_redesigned.py`, `run_suite*.sh`)
live at git tag `v1-jax-suite`; their committed output stays in `scripts/results/`
(see `DATA.md`). Physical changes in v2:

- The exact coset-mixture likelihood for the repetition-code GHZ is the default
  (`--model exact`); the legacy minimum-weight closed form survives as
  `--model theorem` for the regression diagnostic only.
- Bare GHZ is simulated under the same Hamiltonian (transverse-field) noise as the
  encoded protocols, with adaptive $N = \lfloor 1/\varepsilon \rfloor$
  (`bare_adaptive`), so bare-versus-encoded comparisons are within one noise model.
- The depolarizing bare-GHZ likelihood (legacy `simulation.py`) was dropped, with the
  heterogeneous-$\gamma$ rows (`h > 0`), the cond-only ablation, the
  $\sigma_\varepsilon$ breakdown scan, and the $\gamma \in \{0.30, 0.45\}$ pilot rows.
- Preparation infidelity scales only the coherent term (a test finding; the first
  draft scaled the whole likelihood).
- New: readout-flip and preparation-infidelity models with mismatched and aware
  estimators, the Heisenberg-window sweep, and the `M_exec_max` column (deepest
  executed round depth) for `sequential`.
