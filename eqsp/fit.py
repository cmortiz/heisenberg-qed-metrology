"""Scaling-exponent and prefactor fits shared by scripts/analyze.py and scripts/plot.py.

Convention (the one the committed paper tables were built from; see "Legacy
disagreements" below for where the old analyzers differed):

Rows entering the alpha fit
    For one cell (a group of GROUP_KEYS) and one seed, the rows with
    ``converged == True`` and ``epsilon < eps_max`` of the window.  T is the
    row's ``total_resources`` (one value per (seed, epsilon); rows are
    deduplicated on (cell, seed, epsilon) keeping the last row before any fit,
    the same rule as suite.py --merge).  Non-converged targets are dropped from
    the fit and are never imputed.
Per-seed fit
    Ordinary least squares of log T on log epsilon with an intercept
    (numpy.linalg.lstsq); alpha_seed = -slope.  No weighting.
Seeds retained (DATA.md "Seeds" column)
    A seed enters the fit only if it has at least ``min_points`` converged
    targets inside the window.  ``n_seeds`` is the retained count; it can be
    below the 40 seeds run.
Across seeds
    alpha = mean of alpha_seed over retained seeds; sem = std(ddof=1)/sqrt(n);
    ci95 = t_{0.975, n-1} * sem.  With n = 1 the sem is 0; with n = 0 all
    statistics are NaN.
Windows (exactly two)
    "full"        all targets (eps_max = inf),  min_points = 10
    "asymptotic"  epsilon < 1e-2,               min_points = 5
Converged (%)
    Always the converged fraction over every row of the cell (all 60 targets
    times all seeds), for both windows.  It never uses the window subset.
Window shift
    shift = alpha_full - alpha_asymptotic for the cell (replaces the paper's
    untraceable WLS sentence).
Prefactor C (tab:prefactor-results)
    Geometric mean of total_resources * epsilon over the converged rows of the
    cell pooled across seeds, over the full range (reanalysis_within_hamiltonian
    .py:52-64); sem_logC = std(ddof=1)/sqrt(n_rows) of log(T*eps).

Legacy disagreements (all scripts under scripts/ at tag v1-jax-suite)
    analyze_results.py:16-37, analyze_combined.py:21-87, analyze_bare_hamiltonian
    .py:21-35, analyze_sequential.py, plot_redesigned.py:79-91 all use the
    convention above (lstsq, full min_points 10, asymptotic min_points 5).
    reanalysis_within_hamiltonian.py:30-49 uses numpy.polyfit, full range only,
    min_points 4 (its alpha column is not printed in the paper; on the
    committed bare_hamiltonian.csv min_points 4 and 10 retain the same 40 seeds
    and give identical alpha).  compare_quenched_vs_marg.py:22-38 uses
    scipy.stats.linregress on eps < 1e-2 with min_points 3 (only feeds the
    quenched-vs-marginalized aside).  analyze_results.py also has a third
    "deep" window (eps < 1e-3, min_points 3) that no table uses; dropped.
    None of the legacy scripts deduplicated rows: ed_full_likelihood.csv and
    ed_postselect.csv carry seed 1 of (L=1, gamma=0.01) twice (identical rows,
    so alpha is unchanged but the pooled prefactor moves from 197.4 to 198.7,
    which is where the paper's 199 comes from), and bare_ghz.csv carries seeds
    1 and 2 of gamma=0 twice with different resources (full-range alpha 1.151
    with duplicates, 1.152 without).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

EPS_ASYMPTOTIC = 1e-2
WINDOWS: dict[str, tuple[float, int]] = {
    "full": (np.inf, 10),
    "asymptotic": (EPS_ASYMPTOTIC, 5),
}
FIT_COLUMNS = ["seed", "epsilon", "total_resources", "converged"]


@dataclass
class AlphaFit:
    window: str
    eps_max: float
    min_points: int
    alpha: float
    sem: float
    ci95: float
    intercept: float            # mean per-seed intercept of log T = intercept - alpha log eps
    n_seeds: int                # seeds retained (>= min_points converged targets in window)
    n_rows: int                 # converged rows that entered the fit
    converged_frac: float       # over all rows of the cell, both windows
    n_rows_total: int           # all rows of the cell (seeds x targets)
    per_seed: dict[int, tuple[float, float]] = field(default_factory=dict)  # seed -> (alpha, intercept)

    def as_dict(self) -> dict:
        return {
            "value": _f(self.alpha), "sem": _f(self.sem), "ci95": _f(self.ci95),
            "intercept": _f(self.intercept), "window": self.window,
            "min_points": self.min_points, "n_seeds": self.n_seeds, "n_rows": self.n_rows,
            "converged_pct": _f(100.0 * self.converged_frac), "n_rows_total": self.n_rows_total,
        }


@dataclass
class Prefactor:
    C: float
    sem_logC: float
    n_rows: int

    def as_dict(self) -> dict:
        return {"value": _f(self.C), "sem": _f(self.sem_logC), "window": "full",
                "min_points": None, "n_rows": self.n_rows,
                "note": "geometric mean of T*eps over converged rows pooled across seeds; sem is of log C"}


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def coerce_converged(col: pd.Series) -> pd.Series:
    """Accept bool, 0/1, or 'True'/'False' strings."""
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def fit_seed(eps: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """OLS of log T on log eps; returns (alpha, intercept) with alpha = -slope."""
    A = np.vstack([np.log(eps), np.ones(len(eps))]).T
    slope, intercept = np.linalg.lstsq(A, np.log(T), rcond=None)[0]
    return -float(slope), float(intercept)


def fit_alpha(cell: pd.DataFrame, window: str) -> AlphaFit:
    """Per-seed OLS on converged rows inside the window, then mean and SEM over retained seeds."""
    eps_max, min_points = WINDOWS[window]
    conv = coerce_converged(cell["converged"])
    n_total = len(cell)
    converged_frac = float(conv.mean()) if n_total else np.nan
    sub = cell[conv & (cell["epsilon"] < eps_max)]
    per_seed: dict[int, tuple[float, float]] = {}
    n_rows = 0
    for seed, sd in sub.groupby("seed"):
        if len(sd) < min_points:
            continue
        per_seed[int(seed)] = fit_seed(sd["epsilon"].to_numpy(float), sd["total_resources"].to_numpy(float))
        n_rows += len(sd)
    alphas = np.array([a for a, _ in per_seed.values()])
    intercepts = np.array([c for _, c in per_seed.values()])
    n = len(alphas)
    if n == 0:
        alpha = sem = ci95 = intercept = np.nan
    else:
        alpha, intercept = float(alphas.mean()), float(intercepts.mean())
        sem = float(alphas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ci95 = float(sp_stats.t.ppf(0.975, df=n - 1) * sem) if n > 1 else 0.0
    return AlphaFit(window, eps_max, min_points, alpha, sem, ci95, intercept,
                    n, n_rows, converged_frac, n_total, per_seed)


def fit_both_windows(cell: pd.DataFrame) -> dict[str, AlphaFit]:
    return {w: fit_alpha(cell, w) for w in WINDOWS}


def window_shift(fits: dict[str, AlphaFit]) -> float:
    """alpha_full - alpha_asymptotic (NaN if either window is empty)."""
    return float(fits["full"].alpha - fits["asymptotic"].alpha)


def prefactor(cell: pd.DataFrame) -> Prefactor:
    """Geometric mean of T*eps over converged rows pooled across seeds (full range)."""
    sub = cell[coerce_converged(cell["converged"])]
    if sub.empty:
        return Prefactor(np.nan, np.nan, 0)
    logC = np.log(sub["total_resources"].to_numpy(float) * sub["epsilon"].to_numpy(float))
    sem = float(logC.std(ddof=1) / np.sqrt(len(logC))) if len(logC) > 1 else 0.0
    return Prefactor(float(np.exp(logC.mean())), sem, int(len(logC)))


def converged_pct(cell: pd.DataFrame) -> float:
    """Converged (%) over all rows of the cell (all targets, all seeds)."""
    return 100.0 * float(coerce_converged(cell["converged"]).mean())


@dataclass
class LogFit:
    """Seed-mean T*eps against ln(1/eps) over all converged rows (full range):
    linear model T*eps = a + b ln(1/eps) (Heisenberg with logarithmic overhead) versus a pure power law
    ln(T*eps) = c + (alpha - 1) ln(1/eps); rel_rms_* is the RMS of (data/model - 1)."""
    a: float
    b: float
    rel_rms_lin: float
    rel_rms_pow: float
    alpha_pow: float
    n_eps: int


def logfit(cell: pd.DataFrame) -> LogFit:
    sub = cell[coerce_converged(cell["converged"])]
    if sub["epsilon"].nunique() < WINDOWS["full"][1]:   # same retention rule as the full window (10 targets)
        return LogFit(np.nan, np.nan, np.nan, np.nan, np.nan, int(sub["epsilon"].nunique()))
    m = sub.groupby("epsilon")["total_resources"].mean()
    eps = m.index.to_numpy(float)
    y = m.to_numpy(float) * eps
    x = np.log(1.0 / eps)
    b, a = np.polyfit(x, y, 1)
    lin = a + b * x
    s_, c = np.polyfit(x, np.log(y), 1)
    pw = np.exp(c + s_ * x)
    return LogFit(float(a), float(b), float(np.sqrt(np.mean((y / lin - 1.0) ** 2))),
                  float(np.sqrt(np.mean((y / pw - 1.0) ** 2))), float(1.0 + s_), int(len(eps)))


def alpha_implied_by_log_law(eps: np.ndarray, window: str) -> float:
    """Exponent a power-law fit assigns to the pure law T = ln(1/eps)/eps on the given target grid and window."""
    eps_max, _ = WINDOWS[window]
    e = np.asarray(eps, float)
    e = e[e < eps_max]
    return fit_seed(e, np.log(1.0 / e) / e)[0]
