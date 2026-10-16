#!/usr/bin/env python3
"""Analyze suite CSVs: per-cell fits, numbers.json (one key per paper element), LaTeX table rows.

Usage
    uv run python scripts/analyze.py                       # results/v2
    uv run python scripts/analyze.py --results-dir DIR     # any directory of suite CSVs

Outputs (under the results directory)
    derived/cells.csv            one row per (GROUP_KEYS) cell: both fit windows, shift, prefactor, acceptance
    derived/hwindow_summary.csv  one row per (sigma_epsilon, N): T mean/median over seeds (hwindow protocol only)
    derived/fisher.csv           per-cell Fisher-saturation ratios (ED-FL and combined-FL)
    numbers.json       paper elements; keys documented in results/numbers_schema.md
    tables/*.tex       table rows in the paper's column order (paste into the tabular bodies)

Fit convention: eqsp/fit.py docstring.  Every alpha carries its window and min_points.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eqsp import fit  # noqa: E402
from eqsp.config import GROUP_KEYS, OMEGA_TRUE  # noqa: E402

V2_DIR = REPO / "results" / "v2"

PROTOCOLS = ["bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow"]   # files: <results-dir>/<protocol>.csv
TAB_PREFACTOR_ROWS = [(L, g) for L in (1, 2, 3) for g in (0.0, 0.01, 0.05, 0.10)]   # tab:prefactor-results rows; gamma = 0 is the calibration row
BARE_PREFACTOR_MODE = "bare_hamiltonian_matched"   # the bare column runs the ED depth schedule M ~ U{1..floor(1/(N eps))}; the legacy bare_hamiltonian cells drew M ~ U{1..floor(1/eps)}, N times deeper per shot
FIG1_BARE_GAMMAS = [0.01, 0.05, 0.10]
FIG3_CELLS = [(1, 0.01, 0.01), (1, 0.10, 0.01), (2, 0.10, 0.01)]
WEAK_NOISE_P = [0.01, 0.05, 0.10]
SEQ_EPS_QUOTE = 1e-2   # target at which the sequential depth is quoted (plan P5: "deepest executed round depth per eps"); the asymptotic-window edge
COMBINED_TABLE_SIGMAS = [0.01, 0.05]   # tab:combined-results cells


# --------------------------------------------------------------------------- loading
def protocol_files(results_dir: Path) -> list[Path]:
    return [results_dir / f"{p}.csv" for p in PROTOCOLS if (results_dir / f"{p}.csv").exists()]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Bool converged, int L, deduplicated on the cell keys plus (seed, epsilon) keeping the last row."""
    df["L"] = df["L"].astype(int)
    df["converged"] = fit.coerce_converged(df["converged"])
    dedupe_keys = GROUP_KEYS + ["seed", "epsilon", "N"]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    df.attrs["n_duplicates_dropped"] = before - len(df)
    return df


def load_results(results_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path).assign(source_csv=path.name) for path in protocol_files(results_dir)]
    if not frames:
        sys.exit(f"no protocol CSVs found in {results_dir}")
    return normalize(pd.concat(frames, ignore_index=True, sort=False))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no readout/prep noise and the mismatched (plain) estimator."""
    return df[(df["q"] == 0) & (df["k"] == 1) & (df["F"] == 1) & (df["estimator"] == "mismatched")]


# --------------------------------------------------------------------------- per-cell analysis
def analyze_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in df[df["protocol"] != "hwindow"].groupby(GROUP_KEYS, sort=True):
        fits = fit.fit_both_windows(cell)
        pf = fit.prefactor(cell)
        lf = fit.logfit(cell)
        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "source_csv": cell["source_csv"].iloc[0],
            "n_seeds_run": int(cell["seed"].nunique()),
            "n_rows": int(len(cell)),
            "converged_pct": fit.converged_pct(cell),
            "shift_full_minus_asym": fit.window_shift(fits),
            "C": pf.C, "sem_logC": pf.sem_logC, "n_rows_C": pf.n_rows,
            "logfit_a": lf.a, "logfit_b": lf.b, "logfit_rel_rms": lf.rel_rms_lin, "powfit_rel_rms": lf.rel_rms_pow,
            "powfit_alpha": lf.alpha_pow,
            "accept_pct": 100.0 * float(cell["acceptance_rate"].mean()) if cell["acceptance_rate"].notna().any() else np.nan,
        })
        for w, f in fits.items():
            row.update({f"alpha_{w}": f.alpha, f"sem_{w}": f.sem, f"ci95_{w}": f.ci95,
                        f"intercept_{w}": f.intercept, f"n_seeds_{w}": f.n_seeds, f"n_rows_{w}": f.n_rows})
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_hwindow(df: pd.DataFrame) -> pd.DataFrame:
    """T(N) mean and median over converged seeds per (sigma_epsilon, N); N* = 1/(2 sigma^2)."""
    hw = df[df["protocol"] == "hwindow"]
    rows = []
    for (sigma, N), g in hw.groupby(["sigma_epsilon", "N"], sort=True):
        conv = g[g["converged"]]
        rows.append({
            "sigma_epsilon": sigma, "N": int(N), "eps_tar": float(g["epsilon"].iloc[0]),
            "source_csv": g["source_csv"].iloc[0],
            "n_seeds": int(g["seed"].nunique()), "n_converged": int(len(conv)),
            "converged_pct": 100.0 * float(g["converged"].mean()),
            "T_mean": float(conv["total_resources"].mean()) if len(conv) else np.nan,
            "T_median": float(conv["total_resources"].median()) if len(conv) else np.nan,
            "T_sem": float(conv["total_resources"].std(ddof=1) / np.sqrt(len(conv))) if len(conv) > 1 else np.nan,
            "N_star": 1.0 / (2.0 * sigma**2), "N_star_half": 1.0 / (4.0 * sigma**2),
        })
    return pd.DataFrame(rows)


def fisher_saturation(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell saturation of sum_F_obs against sum_M_sq * 4 N^2 (1 - (gamma/omega)^2) on eps < 1e-2.

    Converged rows, per-seed mean of the ratio over the asymptotic window, then
    mean and SEM over seeds.
    seed_min is the smallest per-seed mean (the "dips to 0.97 for individual seeds" figure).
    """
    sub = clean(df[df["mode"].isin(["ed_full_likelihood", "combined_full_likelihood"])])
    sub = sub[sub["sum_F_obs"].notna() & sub["converged"]]
    rows = []
    for keys, g in sub.groupby(GROUP_KEYS, sort=True):
        row = dict(zip(GROUP_KEYS, keys))
        N = int(g["N_total"].iloc[0]) if row["protocol"] == "combined" else int(g["N_code"].iloc[0])
        if row["gamma"] >= OMEGA_TRUE:   # the 1 - gamma^2/omega^2 ceiling is not defined there; no suite cell reaches it
            continue
        correction = 1.0 - (row["gamma"] / OMEGA_TRUE) ** 2
        ratio = g["sum_F_obs"] / (g["sum_M_sq"] * 4.0 * N**2 * correction)
        asym = g.assign(ratio=ratio)[g["epsilon"] < fit.EPS_ASYMPTOTIC]
        seed_means = asym.groupby("seed")["ratio"].mean()
        n = len(seed_means)
        row.update({"N": N, "source_csv": g["source_csv"].iloc[0], "n_seeds": n, "n_rows": int(len(asym)),
                    "sat_mean": float(seed_means.mean()) if n else np.nan,
                    "sat_sem": float(seed_means.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                    "seed_min": float(seed_means.min()) if n else np.nan,
                    "seed_max": float(seed_means.max()) if n else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- numbers.json helpers
def gk(g): return f"g{g:.2f}"
def sk(s): return f"s{s:.2f}"
def qk(q): return f"q{q:g}"
def kk(k): return f"k{k:g}"


def fnum(x):
    return None if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else float(x)


def alpha_entry(row: pd.Series, window: str) -> dict:
    return {"value": fnum(row[f"alpha_{window}"]), "sem": fnum(row[f"sem_{window}"]),
            "ci95": fnum(row[f"ci95_{window}"]), "intercept": fnum(row[f"intercept_{window}"]),
            "window": window, "min_points": fit.WINDOWS[window][1], "source_csv": row["source_csv"],
            "n_seeds": int(row[f"n_seeds_{window}"]), "n_rows": int(row[f"n_rows_{window}"]),
            "converged_pct": fnum(row["converged_pct"]), "n_seeds_run": int(row["n_seeds_run"])}


def scalar_entry(value, source_csv, note, window=None, sem=None, n_seeds=None, n_rows=None) -> dict:
    return {"value": fnum(value), "sem": fnum(sem), "window": window, "min_points": None,
            "source_csv": source_csv, "n_seeds": n_seeds, "n_rows": n_rows, "note": note}


def cell_entries(row: pd.Series, window: str) -> dict:
    """The standard bundle for one table cell: alpha (in the table's window), the other window, shift, converged."""
    other = "asymptotic" if window == "full" else "full"
    return {
        "alpha": alpha_entry(row, window),
        f"alpha_{other}": alpha_entry(row, other),
        "window_shift": scalar_entry(row["shift_full_minus_asym"], row["source_csv"],
                                     "alpha_full minus alpha_asymptotic for this cell (replaces the paper's WLS sentence)"),
        "converged_pct": scalar_entry(row["converged_pct"], row["source_csv"],
                                      "converged fraction over all rows of the cell (all 60 targets)",
                                      n_rows=int(row["n_rows"])),
        "C": scalar_entry(row["C"], row["source_csv"], "geometric mean of T*eps over converged rows (full range); sem is of log C",
                          window="full", sem=row["sem_logC"], n_rows=int(row["n_rows_C"])),
        "logfit_b": scalar_entry(row["logfit_b"], row["source_csv"], "slope b of seed-mean T*eps = a + b ln(1/eps) over all converged targets"),
        "logfit_a": scalar_entry(row["logfit_a"], row["source_csv"], "intercept a of seed-mean T*eps = a + b ln(1/eps)"),
        "logfit_rel_rms": scalar_entry(row["logfit_rel_rms"], row["source_csv"], "RMS relative residual of the a + b ln(1/eps) model (fraction)"),
        "powfit_rel_rms": scalar_entry(row["powfit_rel_rms"], row["source_csv"], "RMS relative residual of a pure power law on the same seed-mean T*eps (fraction)"),
    }


def add_bundle(nums: dict, prefix: str, bundle: dict) -> None:
    for name, entry in bundle.items():
        nums[f"{prefix}.{name}"] = entry


def pick(cells: pd.DataFrame, **cond) -> pd.Series | None:
    mask = np.ones(len(cells), dtype=bool)
    for col, val in cond.items():
        # floats round-trip through the CSV at %.10e, so isclose (rtol 1e-5) only absorbs that formatting
        mask &= np.isclose(cells[col], val) if isinstance(val, (int, float)) else (cells[col] == val).to_numpy()
    hit = cells[mask]
    if len(hit) > 1:
        raise ValueError(f"pick: {len(hit)} cells match {cond}; the cell table must be unique on the grouping keys")
    return hit.iloc[0] if len(hit) == 1 else None


def clean_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return cells[(cells["q"] == 0) & (cells["k"] == 1) & (cells["F"] == 1) & (cells["estimator"] == "mismatched") & (cells["h"] == 0)]


# --------------------------------------------------------------------------- numbers.json builders
def numbers_bare(cells: pd.DataFrame, nums: dict) -> None:
    """Bare GHZ, adaptive N, by gamma. Window: full."""
    bare = clean_cells(cells)
    bare = bare[bare["protocol"] == "bare_adaptive"]
    for _, row in bare.iterrows():
        add_bundle(nums, f"tab_bare_within_model.{gk(row['gamma'])}", cell_entries(row, "full"))
        nums[f"fig2.alpha.bare_{gk(row['gamma'])}"] = alpha_entry(row, "asymptotic")
        nums[f"fig2.conv.bare_{gk(row['gamma'])}"] = nums[f"tab_bare_within_model.{gk(row['gamma'])}.converged_pct"]
        if any(np.isclose(row["gamma"], g) for g in FIG1_BARE_GAMMAS):
            nums[f"fig1.alpha.bare_{gk(row['gamma'])}"] = alpha_entry(row, "full")


def numbers_bare_fixed(cells: pd.DataFrame, nums: dict) -> None:
    """Bare GHZ at fixed N = 2L+1 under Hamiltonian noise (the bare column of tab:prefactor-results).
    Window: full (the text quotes full-range alpha); emitted for every bare_fixed cell, ED counterpart or not."""
    bf = clean_cells(cells[cells["protocol"] == "bare_fixed"])
    for _, row in bf[bf["mode"] == BARE_PREFACTOR_MODE].iterrows():
        p = f"tab_prefactor.L{row['L']}_{gk(row['gamma'])}"
        b = cell_entries(row, "full")
        nums[f"{p}.bare_C"] = b["C"]
        nums[f"{p}.bare_alpha_full"] = b["alpha"]
        nums[f"{p}.bare_alpha_asym"] = b["alpha_asymptotic"]
        nums[f"{p}.bare_window_shift"] = b["window_shift"]
        nums[f"{p}.bare_converged_pct"] = b["converged_pct"]


def numbers_ed(cells: pd.DataFrame, nums: dict) -> None:
    """tab:error-detected-results (asymptotic) plus fig1 ED-FL line (full) and fig2 bars (asymptotic)."""
    ed = clean_cells(cells[cells["protocol"] == "ed"])
    for _, row in ed.iterrows():
        mode = "edfl" if row["mode"] == "ed_full_likelihood" else "rej"
        prefix = f"tab_ed.L{row['L']}_{gk(row['gamma'])}.{mode}"
        add_bundle(nums, prefix, cell_entries(row, "asymptotic"))
        nums[f"{prefix}.accept_pct"] = scalar_entry(row["accept_pct"], row["source_csv"],
                                                    "mean acceptance_rate over all rows of the cell (100 for ED-FL)")
        nums[f"fig2.alpha.ed_L{row['L']}_{gk(row['gamma'])}_{mode}"] = alpha_entry(row, "asymptotic")
        nums[f"fig2.conv.ed_L{row['L']}_{gk(row['gamma'])}_{mode}"] = nums[f"{prefix}.converged_pct"]
        if mode == "edfl" and row["L"] == 1 and np.isclose(row["gamma"], 0.10):
            nums["fig1.alpha.edfl_L1_g0.10"] = alpha_entry(row, "full")


def numbers_prefactor(cells: pd.DataFrame, nums: dict) -> None:
    """tab:prefactor-results: C for the matched-depth bare probe, ED-FL and rejection at matched (L, gamma).

    All three columns draw M ~ U{1..floor(1/(N eps_tar))} and are charged N * M per shot, so the
    entries are read at matched resources. No ratios are emitted: the paper states the prefactors.
    """
    cc = clean_cells(cells)
    bare_cells = cc[(cc["protocol"] == "bare_fixed") & (cc["mode"] == BARE_PREFACTOR_MODE)]
    for _, bare in bare_cells.iterrows():
        L, g = int(bare["L"]), float(bare["gamma"])
        edfl = pick(cc, protocol="ed", mode="ed_full_likelihood", L=L, gamma=g)
        rej = pick(cc, protocol="ed", mode="ed_postselect", L=L, gamma=g)
        if edfl is None and rej is None:
            continue
        p = f"tab_prefactor.L{L}_{gk(g)}"
        for name, row in (("edfl", edfl), ("rej", rej)):
            if row is None:
                continue
            b = cell_entries(row, "asymptotic")
            nums[f"{p}.{name}_C"] = b["C"]
            nums[f"{p}.{name}_alpha_asym"] = b["alpha"]
            nums[f"{p}.{name}_window_shift"] = b["window_shift"]


def numbers_combined(cells: pd.DataFrame, nums: dict) -> None:
    """tab:combined-results (asymptotic), fig3 legend alphas, per-block acceptance. C per cell, no ratios."""
    cc = clean_cells(cells)
    comb = cc[cc["protocol"] == "combined"]
    for _, row in comb.iterrows():
        mode = {"combined_full_likelihood": "edfl", "combined_postselect": "rej",
                "combined_full_likelihood_marg": "marg", "combined_full_likelihood_marg_capped": "margcap"}[row["mode"]]
        L, g, s = int(row["L"]), float(row["gamma"]), float(row["sigma_epsilon"])
        cell = f"L{L}_{gk(g)}_{sk(s)}"
        add_bundle(nums, f"tab_combined.{cell}.{mode}", cell_entries(row, "asymptotic"))
        nums[f"tab_combined.{cell}.{mode}.accept_pct"] = scalar_entry(row["accept_pct"], row["source_csv"], "mean acceptance_rate over all rows of the cell")
        if mode == "rej":
            nums[f"text.combined.per_block_accept_pct_{cell}"] = scalar_entry(
                100.0 * (row["accept_pct"] / 100.0) ** (1.0 / 3.0), row["source_csv"],
                "cube root of the joint (three-block) acceptance rate, in percent")
        if mode == "edfl":
            if any(L == fL and np.isclose(g, fg) and np.isclose(s, fs) for fL, fg, fs in FIG3_CELLS):
                nums[f"fig3.alpha.combined_{cell}"] = alpha_entry(row, "asymptotic")



def numbers_sequential(cells: pd.DataFrame, df: pd.DataFrame, nums: dict) -> None:
    """tab:sequential-results (full window) and the deepest executed depth at eps nearest 1e-2."""
    cc = clean_cells(cells)
    for _, row in cc[cc["protocol"] == "sequential"].iterrows():
        add_bundle(nums, f"tab_sequential.L{row['L']}_{sk(row['sigma_epsilon'])}", cell_entries(row, "full"))
    seq = clean(df[df["protocol"] == "sequential"])
    if seq.empty:
        return
    eps_grid = np.sort(seq["epsilon"].unique())
    eps_q = float(eps_grid[np.argmin(np.abs(eps_grid - SEQ_EPS_QUOTE))])
    at = seq[np.isclose(seq["epsilon"], eps_q)]
    src = at["source_csv"].iloc[0]
    m_exec = float(at["M_exec_max"].max())
    nums["text.sequential.M_exec_max_eps1e-2"] = scalar_entry(
        m_exec, src,
        f"max of the M_exec_max column over seeds and cells at eps_tar = {eps_q:.6g} (nearest grid point to 1e-2)")
    nums["text.sequential.eps_quoted"] = scalar_entry(eps_q, src, "grid point used for the depth quotes")
    for s in sorted(seq["sigma_epsilon"].unique()):
        if s > 0:
            nums[f"text.sequential.M2sigma2_{sk(s)}_eps1e-2"] = scalar_entry(
                m_exec**2 * s**2, src, "M_exec_max^2 * sigma_epsilon^2 at the quoted eps")
            nums[f"text.sequential.visibility_{sk(s)}_eps1e-2"] = scalar_entry(
                math.exp(-0.5 * m_exec**2 * s**2), src, "per-qubit visibility exp(-M_exec_max^2 sigma_epsilon^2 / 2) at the quoted eps")
    # ranges over L quoted in the main text (value = [min, max] over the L cells)
    for s in sorted(cc.loc[cc["protocol"] == "sequential", "sigma_epsilon"].unique()):
        rows = cc[(cc["protocol"] == "sequential") & np.isclose(cc["sigma_epsilon"], s)]
        tag = "noiseless" if s == 0 else sk(s)
        nums[f"text.sequential.alpha_full_{tag}"] = range_entry(rows["alpha_full"], rows["source_csv"].iloc[0],
                                                                f"[min, max] over L of the full-range alpha at sigma = {s:g}", window="full")
        nums[f"text.sequential.alpha_asym_{tag}_range"] = range_entry(rows["alpha_asymptotic"], rows["source_csv"].iloc[0],
                                                                f"[min, max] over L of the asymptotic-window alpha at sigma = {s:g}", window="asymptotic")
        nums[f"text.sequential.conv_{tag}_range"] = range_entry(rows["converged_pct"], rows["source_csv"].iloc[0],
                                                                f"[min, max] over L of Converged (%) at sigma = {s:g}")

    nc = seq[np.isclose(seq["sigma_epsilon"], 0.0) & ~seq["converged"]]
    if len(nc):
        nums["text.sequential.nonconv_error_over_eps_median_s0"] = scalar_entry(
            float((nc["final_error"] / nc["epsilon"]).median()), src,
            "median |error|/eps_tar over the non-converged noiseless sequential rows: a wrong bit in some round "
            "(error many tolerances wide), not a near miss")
    # The binary-search cost schedule is deterministic: T at fixed eps_tar varies by less than
    # a fraction of a percent across seeds and not at all with sigma_epsilon, so the exponent can
    # be fitted on all 60 targets.  The sigma-dependence of the converged-row fit in
    # tab:sequential-results is the truncation of the fit to the targets that converged.
    sched_full, sched_asym, spread = [], [], []
    for (_L, _s), cell in seq.groupby(["L", "sigma_epsilon"]):
        g = cell.groupby("epsilon")["total_resources"].mean()
        e_all, T_all = g.index.to_numpy(float), g.to_numpy(float)
        sched_full.append(fit.fit_seed(e_all, T_all)[0])
        m = e_all < fit.WINDOWS["asymptotic"][0]
        if m.sum() >= fit.WINDOWS["asymptotic"][1]:
            sched_asym.append(fit.fit_seed(e_all[m], T_all[m])[0])
        per_eps = cell.groupby("epsilon")["total_resources"].agg(["min", "max"])
        spread.append(float(((per_eps["max"] - per_eps["min"]) / per_eps["max"]).max()))   # relative to the largest cost: the conservative (smallest) normaliser
    nums["text.sequential.alpha_schedule_range"] = range_entry(
        pd.Series(sched_full), src, "[min, max] over all (L, sigma_epsilon) cells of the full-range alpha fitted "
        "on the seed-mean cost at all 60 targets, converged or not (the schedule is deterministic)", window="full")
    nums["text.sequential.alpha_schedule_asym_range"] = range_entry(
        pd.Series(sched_asym), src, "[min, max] over all (L, sigma_epsilon) cells of the asymptotic-window alpha "
        "fitted on the seed-mean cost at all targets", window="asymptotic")
    nums["text.sequential.T_seed_spread_pct"] = scalar_entry(
        100.0 * max(spread), src, "largest relative spread (max - min)/max of the total cost across seeds at a "
        "fixed eps_tar, over all sequential cells")


def numbers_marg(cells: pd.DataFrame, df: pd.DataFrame, nums: dict) -> None:
    """Marginalised-inference cells: the offset noise floor, the information-optimal depth cap,
    and what each estimator does below the floor (Appendix~F, quenched vs marginalised)."""
    marg = clean(df[df["mode"].astype(str).str.startswith("combined_full_likelihood_marg")])
    if marg.empty:
        return
    src = marg["source_csv"].iloc[0]
    N_tot = float(marg["N_total"].iloc[0])
    sigma = float(marg["sigma_epsilon"].iloc[0])
    floor = sigma / math.sqrt(N_tot)
    M_cap = math.ceil(1.0 / (2.0 * sigma * math.sqrt(N_tot)))
    nums["text.marg.eps_floor"] = scalar_entry(
        floor, src, "offset noise floor sigma_epsilon / sqrt(N_total) of the marginalised cell")
    nums["text.marg.M_cap"] = scalar_entry(
        float(M_cap), src, "information-optimal depth ceil(1 / (2 sigma_epsilon sqrt(N_total))), the peak of "
        "4 M^2 N^2 exp(-4 M^2 N sigma_epsilon^2)")
    eps_min = float(np.min(marg["epsilon"]))
    nums["text.marg.eps_min"] = scalar_entry(eps_min, src, "smallest target of the eps grid")
    for mode, tag in (("combined_full_likelihood_marg", "uncapped"),
                      ("combined_full_likelihood_marg_capped", "capped")):
        cell = marg[marg["mode"] == mode]
        if cell.empty:
            continue
        at_min = cell[np.isclose(cell["epsilon"], eps_min)]
        nums[f"text.marg.conv_{tag}_min_eps"] = scalar_entry(
            100.0 * float(fit.coerce_converged(at_min["converged"]).mean()), src,
            f"Converged (%) of the {tag} marginalised cell at the smallest target")
        nums[f"text.marg.M_max_{tag}_min_eps"] = scalar_entry(
            float(at_min["M_max"].max()), src, f"depth ceiling M_max of the {tag} cell at the smallest target")
        conv = cell[fit.coerce_converged(cell["converged"])]
        below = conv[conv["epsilon"] < floor]
        g = below.groupby("epsilon")["total_resources"].mean()
        if len(g) >= fit.WINDOWS["asymptotic"][1]:
            nums[f"text.marg.alpha_below_floor_{tag}"] = scalar_entry(
                fit.fit_seed(g.index.to_numpy(float), g.to_numpy(float))[0], src,
                f"alpha of the seed-mean converged cost of the {tag} cell on the targets below the noise floor",
                window="below_floor")
        # the same fit restricted to targets where the cell still converges on most seeds
        ok = [e for e, sd in cell.groupby("epsilon") if fit.coerce_converged(sd["converged"]).mean() >= 0.9]
        gg = conv[conv["epsilon"].isin(ok) & (conv["epsilon"] < floor)].groupby("epsilon")["total_resources"].mean()
        if len(gg) >= fit.WINDOWS["asymptotic"][1]:   # same retention rule as the asymptotic window (5 targets), the floor sits inside it
            nums[f"text.marg.alpha_below_floor_converged_{tag}"] = scalar_entry(
                fit.fit_seed(gg.index.to_numpy(float), gg.to_numpy(float))[0], src,
                f"alpha of the {tag} cell below the noise floor, restricted to targets that converge on at least "
                "90% of seeds", window="below_floor")
            nums[f"text.marg.eps_reach_{tag}"] = scalar_entry(
                float(min(gg.index)), src,
                f"smallest target at which the {tag} cell still converges on at least 90% of seeds")


def numbers_realistic(cells: pd.DataFrame, nums: dict) -> None:
    """Readout-error (q > 0) and prep-infidelity (k < 1 or F < 1) cells for every protocol. Window: asymptotic."""
    real = cells[(cells["q"] > 0) | (cells["k"] < 1) | (cells["F"] < 1)]
    for _, row in real.iterrows():
        proto = {"bare_adaptive": "bare", "bare_fixed": "bare_fixed", "ed": "ed", "combined": "combined", "sequential": "sequential"}[row["protocol"]]
        if row["mode"] in ("ed_postselect", "combined_postselect"):
            proto += "_rej"
        Lpart = "" if row["protocol"] == "bare_adaptive" else f"_L{row['L']}"
        # sequential: majority vote is estimator-agnostic (no estimator suffix); prep enters through F only
        est = "" if row["protocol"] == "sequential" else f"_{row['estimator']}"
        if row["q"] > 0:
            key = f"tab_realistic.readout.{proto}{Lpart}_{qk(row['q'])}{est}"
        elif row["protocol"] == "sequential":
            key = f"tab_realistic.prep.{proto}{Lpart}_F{row['F']:g}"
        else:
            key = f"tab_realistic.prep.{proto}{Lpart}_{kk(row['k'])}{est}"
        add_bundle(nums, key, cell_entries(row, "asymptotic"))
        nums[f"{key}.F"] = scalar_entry(row["F"], row["source_csv"], "single-qubit init fidelity of the cell")
    for q in sorted(real.loc[real["q"] > 0, "q"].unique()):
        nums[f"text.readout.n_breakdown_{qk(q)}"] = scalar_entry(math.floor(-1.0 / math.log(1.0 - 2.0 * q)), None, "probe size n = floor(-1/ln(1-2q)) at which the bare parity visibility (1-2q)^n has fallen to e^-1 (about 1/(2q))")
    for k in sorted(real.loc[real["k"] < 1, "k"].unique()):
        nums[f"text.prep.n_star_{kk(k)}"] = scalar_entry(-1.0 / math.log(k), None, "optimal bare probe size n* = -1/ln k under visibility k^(n-1) F")


def numbers_hwindow(hw: pd.DataFrame, nums: dict) -> None:
    """fig4: T(N) per (sigma, N), the argmin over sampled N, and the predicted N* = 1/(2 sigma^2), N*/2."""
    if hw.empty:
        return
    for sigma, g in hw.groupby("sigma_epsilon"):
        p = f"fig4.window.{sk(sigma)}"
        src = g["source_csv"].iloc[0]
        nums[f"{p}.N_star"] = scalar_entry(1.0 / (2.0 * sigma**2), None, "Fisher-information peak N* = 1/(2 sigma^2)")
        nums[f"{p}.N_star_half"] = scalar_entry(1.0 / (4.0 * sigma**2), None, "predicted T(N) minimum N*/2 = 1/(4 sigma^2)")
        ok = g[(g["converged_pct"] >= 50.0) & g["T_mean"].notna()]   # a T minimum is read only where the median seed converged (T_median defined)
        if len(ok):
            nums[f"{p}.N_argmin"] = scalar_entry(int(ok.loc[ok["T_mean"].idxmin(), "N"]), src, "sampled N minimizing mean T over converged seeds (cells with >= 50% convergence)")
            nums[f"{p}.N_argmin_median"] = scalar_entry(int(ok.loc[ok["T_median"].idxmin(), "N"]), src, "sampled N minimizing median T over converged seeds (cells with >= 50% convergence)")
        for _, r in g.iterrows():
            N = int(r["N"])
            nums[f"{p}.T_mean_N{N}"] = scalar_entry(r["T_mean"], src, "mean total_resources over converged seeds", sem=r["T_sem"], n_seeds=int(r["n_converged"]))
            nums[f"{p}.T_median_N{N}"] = scalar_entry(r["T_median"], src, "median total_resources over converged seeds", n_seeds=int(r["n_converged"]))
            nums[f"{p}.converged_pct_N{N}"] = scalar_entry(r["converged_pct"], src, "converged fraction over all seeds at this N", n_seeds=int(r["n_seeds"]))


def numbers_fisher(fs: pd.DataFrame, nums: dict) -> None:
    if fs.empty:
        return
    for _, r in fs.iterrows():
        cell = f"edfl_L{r['L']}_{gk(r['gamma'])}" if r["protocol"] == "ed" else f"combined_L{r['L']}_{gk(r['gamma'])}_{sk(r['sigma_epsilon'])}"
        p = f"text.fisher_saturation.{cell}"
        nums[f"{p}.mean"] = scalar_entry(r["sat_mean"], r["source_csv"], "per-seed mean saturation ratio on eps < 1e-2, averaged over seeds", window="asymptotic", sem=r["sat_sem"], n_seeds=int(r["n_seeds"]), n_rows=int(r["n_rows"]))
        nums[f"{p}.seed_min"] = scalar_entry(r["seed_min"], r["source_csv"], "smallest per-seed mean saturation ratio in the cell", window="asymptotic", n_seeds=int(r["n_seeds"]))
    in_tables = fs[(fs["protocol"] == "ed") | fs["sigma_epsilon"].apply(lambda s: any(np.isclose(s, t) for t in COMBINED_TABLE_SIGMAS))]
    for proto, name in (("ed", "edfl"), ("combined", "combined")):
        sub = in_tables[in_tables["protocol"] == proto]
        if len(sub):
            nums[f"text.fisher_saturation.{name}_min"] = scalar_entry(sub["sat_mean"].min(), None, f"min over {name} table cells of the cell-mean saturation ratio", window="asymptotic")
            nums[f"text.fisher_saturation.{name}_max"] = scalar_entry(sub["sat_mean"].max(), None, f"max over {name} table cells of the cell-mean saturation ratio", window="asymptotic")
    nums["text.fisher_saturation.min"] = scalar_entry(in_tables["seed_min"].min(), None,
                                                      f"smallest per-seed mean saturation ratio over the ED-FL cells and the combined-FL cells with sigma in {COMBINED_TABLE_SIGMAS}", window="asymptotic")


def numbers_weak_noise(nums: dict) -> None:
    for p in WEAK_NOISE_P:
        nums[f"text.weak_noise_map.gamma_ham_p{p:g}"] = scalar_entry(
            math.sqrt(p) * OMEGA_TRUE / math.sin(OMEGA_TRUE), None,
            f"gamma_Ham = sqrt(p) omega / sin(omega) at omega = {OMEGA_TRUE}; leading order of p = sin^2(omega) gamma^2/omega^2")


def range_entry(values, source_csv, note, window=None) -> dict:
    vals = [fnum(v) for v in values if fnum(v) is not None]
    e = scalar_entry(None, source_csv, note, window=window)
    e["value"] = [min(vals), max(vals)] if vals else None
    return e


def numbers_logfit(cells: pd.DataFrame, df: pd.DataFrame, nums: dict) -> None:
    """Heisenberg-with-log-overhead keys: implied exponents of a pure eps^-1 ln(1/eps) law on the target grid,
    and residual ranges of the linear-in-ln(1/eps) model versus a power law for the headline cells."""
    eps = np.sort(df.loc[df["protocol"] != "hwindow", "epsilon"].unique())
    if len(eps) >= fit.WINDOWS["full"][1]:   # the implied exponent is a full-window fit, so the same retention rule applies
        for w in ("full", "asymptotic"):
            nums[f"text.logfit.alpha_implied_{w}"] = scalar_entry(
                fit.alpha_implied_by_log_law(eps, w), None,
                f"alpha a per-seed power-law fit assigns to T = ln(1/eps)/eps on the {len(eps)}-target grid ({w} window)", window=w)
    cc = clean_cells(cells)
    groups = {
        "edfl": cc[(cc["protocol"] == "ed") & (cc["mode"] == "ed_full_likelihood")],
        "rej": cc[(cc["protocol"] == "ed") & (cc["mode"] == "ed_postselect")],
        "bare_fixed": cc[cc["protocol"] == "bare_fixed"],
        "combfl": cc[(cc["protocol"] == "combined") & (cc["mode"] == "combined_full_likelihood")],
    }
    for name, g in groups.items():
        g = g[g["logfit_b"].notna()]
        if g.empty:
            continue
        src = ", ".join(sorted(g["source_csv"].unique()))
        nums[f"text.logfit.{name}_rel_rms_range"] = range_entry(g["logfit_rel_rms"], src, f"[min, max] RMS relative residual of T*eps = a + b ln(1/eps) over {name} cells")
        nums[f"text.logfit.{name}_pow_rel_rms_range"] = range_entry(g["powfit_rel_rms"], src, f"[min, max] RMS relative residual of a pure power law over {name} cells")
        nums[f"text.logfit.{name}_b_range"] = range_entry(g["logfit_b"], src, f"[min, max] slope b over {name} cells")
        nums[f"text.logfit.{name}_alpha_asym_range"] = range_entry(g["alpha_asymptotic"], src, f"[min, max] asymptotic alpha over {name} cells")
    ba = cc[(cc["protocol"] == "bare_adaptive") & np.isclose(cc["gamma"], 0.0)]
    if len(ba) == 1:
        r = ba.iloc[0]
        for k in ("logfit_a", "logfit_b", "logfit_rel_rms", "powfit_rel_rms"):
            nums[f"text.logfit.bare_adaptive_g0_{k}"] = scalar_entry(r[k], r["source_csv"], f"{k} of the noiseless adaptive bare probe")


def numbers_text_extras(cells: pd.DataFrame, nums: dict) -> None:
    """Main-text quotes that are functions of several cells: combined ED-FL asymptotic alpha maximum,
    the block-independence check (cube root of the joint acceptance vs the single-block rate), and the
    [min, max] window shift over the ED-FL and combined-FL table cells."""
    cc = clean_cells(cells)
    comb_fl = cc[(cc["protocol"] == "combined") & (cc["mode"] == "combined_full_likelihood")]
    if not comb_fl.empty:
        worst = comb_fl.loc[comb_fl["alpha_asymptotic"].idxmax()]
        nums["text.combined.alpha_asym_max_edfl"] = scalar_entry(
            worst["alpha_asymptotic"], worst["source_csv"],
            f"largest asymptotic alpha over combined ED-FL cells; attained at L={worst['L']} gamma={worst['gamma']} sigma={worst['sigma_epsilon']}",
            window="asymptotic", sem=worst["sem_asymptotic"], n_seeds=int(worst["n_seeds_asymptotic"]))
    joint = pick(cc, protocol="combined", mode="combined_postselect", L=1, gamma=0.05, sigma_epsilon=0.01)
    single = pick(cc, protocol="ed", mode="ed_postselect", L=1, gamma=0.05)
    if joint is not None and single is not None and fnum(joint["accept_pct"]) and fnum(single["accept_pct"]):
        per_block = 100.0 * (joint["accept_pct"] / 100.0) ** (1.0 / 3.0)
        nums["text.combined.per_block_accept_gap_L1_g0.05_s0.01"] = scalar_entry(
            abs(per_block - single["accept_pct"]), joint["source_csv"],
            "|cube root of the combined L=1 gamma=0.05 sigma=0.01 joint acceptance - ED L=1 gamma=0.05 acceptance| in points")
    ed_fl = cc[(cc["protocol"] == "ed") & (cc["mode"] == "ed_full_likelihood")]
    ed_rej = cc[(cc["protocol"] == "ed") & (cc["mode"] == "ed_postselect")]
    if not ed_fl.empty:
        nums["text.fit.window_shift_edfl_range"] = range_entry(ed_fl["shift_full_minus_asym"].abs(), None,
                                                              "[min, max] of |alpha_full - alpha_asymptotic| over ED-FL table cells")
        nums["text.ed.alpha_asym_edfl_range"] = range_entry(ed_fl["alpha_asymptotic"], ed_fl["source_csv"].iloc[0],
                                                           "[min, max] asymptotic alpha over the ED-FL table cells", window="asymptotic")
        nums["text.ed.conv_edfl_min"] = scalar_entry(ed_fl["converged_pct"].min(), ed_fl["source_csv"].iloc[0],
                                                     "smallest Converged (%) over the ED-FL table cells")
    if not ed_rej.empty:
        nums["text.ed.conv_rej_min"] = scalar_entry(ed_rej["converged_pct"].min(), ed_rej["source_csv"].iloc[0],
                                                     "smallest convergence over the rejection-sampling cells of tab:error-detected-results")
        nums["text.ed.alpha_asym_rej_range"] = range_entry(ed_rej["alpha_asymptotic"], ed_rej["source_csv"].iloc[0],
                                                          "[min, max] asymptotic alpha over the ED rejection-sampling table cells", window="asymptotic")
    ed_all = pd.concat([ed_fl, ed_rej])
    if not ed_all.empty:
        nums["text.ed.alpha_asym_max_dev"] = scalar_entry((ed_all["alpha_asymptotic"] - 1.0).abs().max(), None,
                                                          "largest |alpha_asymptotic - 1| over the ED-FL and rejection table cells (the 'within X of the Heisenberg limit' quote)",
                                                          window="asymptotic")
    bare_rows = cc[cc["protocol"] == "bare_fixed"]
    if not bare_rows.empty:
        nums["text.prefactor.bare_alpha_asym_range"] = range_entry(bare_rows["alpha_asymptotic"], bare_rows["source_csv"].iloc[0],
                                                                   "[min, max] asymptotic alpha over the bare fixed-N cells", window="asymptotic")
        nums["text.prefactor.bare_conv_min"] = scalar_entry(bare_rows["converged_pct"].min(), bare_rows["source_csv"].iloc[0],
                                                            "smallest Converged (%) over the bare fixed-N cells")
    if not comb_fl.empty:
        nums["text.fit.window_shift_combfl_range"] = range_entry(comb_fl["shift_full_minus_asym"].abs(), None,
                                                                "[min, max] of |alpha_full - alpha_asymptotic| over combined ED-FL table cells")


def numbers_shift(cells: pd.DataFrame, nums: dict) -> None:
    """text.fit.window_shift_max: largest |alpha_full - alpha_asymptotic| over the ED and combined table cells."""
    cc = clean_cells(cells[cells["shift_full_minus_asym"].notna()])
    if cc.empty:
        return
    edc = cc[cc["protocol"].isin(["ed", "combined"]) & ~cc["mode"].str.startswith("combined_full_likelihood_marg")]
    if not edc.empty:
        worst = edc.loc[edc["shift_full_minus_asym"].abs().idxmax()]
        nums["text.fit.window_shift_max"] = scalar_entry(
            abs(worst["shift_full_minus_asym"]), worst["source_csv"],
            f"largest |alpha_full - alpha_asymptotic| over the ED and combined (ED-FL and rejection) cells; attained at "
            f"{worst['mode']} L={worst['L']} gamma={worst['gamma']} sigma={worst['sigma_epsilon']}")
    pl = cc[~cc["mode"].str.startswith("combined_full_likelihood_marg")]   # marg cells are not power-law cells (finding 12)
    for proto, g in pl.groupby("protocol"):
        nums[f"text.fit.window_shift_max_{proto}"] = scalar_entry(g["shift_full_minus_asym"].abs().max(), None, f"largest |alpha_full - alpha_asymptotic| over {proto} cells (clean configs, marg excluded)")


def build_numbers(cells, hw, fs, df) -> dict:
    nums: dict = {}
    numbers_bare(cells, nums)
    numbers_bare_fixed(cells, nums)
    numbers_ed(cells, nums)
    numbers_prefactor(cells, nums)
    numbers_combined(cells, nums)
    numbers_sequential(cells, df, nums)
    numbers_marg(cells, df, nums)
    numbers_realistic(cells, nums)
    numbers_hwindow(hw, nums)
    numbers_fisher(fs, nums)
    numbers_weak_noise(nums)
    numbers_shift(cells, nums)
    numbers_text_extras(cells, nums)
    numbers_logfit(cells, df, nums)
    return dict(sorted(nums.items()))


# --------------------------------------------------------------------------- LaTeX rows
def pm(a, s, nd=None):
    """alpha +/- SEM.  Three decimals when the SEM is below 0.005, else two: the rule paper_numbers.fmt applies to
    inline exponents, so a table cell and the text quoting it round identically."""
    if a is None or np.isnan(a):
        return "--"
    if nd is None:
        nd = 3 if (s is not None and not np.isnan(s) and s < 0.005) else 2
    return f"${a:.{nd}f} \\pm {s:.{nd}f}$"


def pct(x, nd=0, gt99=False):
    if x is None or np.isnan(x):
        return "--"
    if gt99 and x > 99 and x < 100:
        return "$>$99"
    return f"{x:.{nd}f}"


def sig2(x):
    if x is None or np.isnan(x) or x == 0:
        return "--"
    d = -int(math.floor(math.log10(abs(x)))) + 1
    return f"{x:.{max(d, 0)}f}"


def gamma_pct_label(g):
    return f"$\\gamma = {100 * g:g}\\%$"


def write_rows(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(f"% {header}\n" + "\n".join(rows) + "\n")


def midrule_groups(rows_by_group: dict[object, list[str]]) -> list[str]:
    out = []
    for i, rows in enumerate(rows_by_group.values()):
        if i:
            out.append("\\midrule")
        out.extend(rows)
    return out


def table_prefactor(cells, outdir):
    cc = clean_cells(cells)
    groups: dict[int, list[str]] = {}
    for L, g in TAB_PREFACTOR_ROWS:   # the paper's five rows; the other (L, gamma) cells stay in derived/cells.csv
        bare = pick(cc, protocol="bare_fixed", mode=BARE_PREFACTOR_MODE, L=L, gamma=g)
        if bare is None:
            continue
        edfl = pick(cc, protocol="ed", mode="ed_full_likelihood", L=L, gamma=g)
        rej = pick(cc, protocol="ed", mode="ed_postselect", L=L, gamma=g)
        if edfl is None:
            continue
        label = f"$L{{=}}{L}$, $\\gamma = 0$" if g == 0 else f"$L{{=}}{L}$, $\\gamma = {g:.2f}$"
        rej_cell = f"${rej['C']:.1f}$" if rej is not None else "---"
        groups.setdefault(L, []).append(
            f"{label} & ${bare['C']:.1f}$ & ${edfl['C']:.1f}$ & {rej_cell} \\\\")
    write_rows(outdir / "tab_prefactor.tex", "Configuration & Bare GHZ & ED-FL & Rejection sampling  (C = geometric mean of T*eps over converged rows, full range; bare column = " + BARE_PREFACTOR_MODE + "; rows = TAB_PREFACTOR_ROWS)", midrule_groups(groups))


def table_ed(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "ed"])
    groups: dict[int, list[str]] = {}
    for (L, g), _ in cc.groupby(["L", "gamma"], sort=True):
        edfl = pick(cc, mode="ed_full_likelihood", L=int(L), gamma=float(g))
        rej = pick(cc, mode="ed_postselect", L=int(L), gamma=float(g))
        if edfl is None or rej is None:
            continue
        groups.setdefault(int(L), []).append(
            f"$L{{=}}{int(L)}$, {gamma_pct_label(g)} & {pm(edfl['alpha_asymptotic'], edfl['sem_asymptotic'])} & {pct(edfl['converged_pct'])} & "
            f"{pm(rej['alpha_asymptotic'], rej['sem_asymptotic'])} & {pct(rej['converged_pct'])} & {pct(rej['accept_pct'], gt99=True)} \\\\")
    write_rows(outdir / "tab_error_detected.tex", "Configuration & ED-FL alpha & Converged (%) & Rejection alpha & Converged (%) & Accept. (%)  (asymptotic window eps < 1e-2, min_points 5)", midrule_groups(groups))


def table_combined(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "combined"])
    groups: dict[tuple, list[str]] = {}
    for (s, L, g), _ in cc.groupby(["sigma_epsilon", "L", "gamma"], sort=True):
        edfl = pick(cc, mode="combined_full_likelihood", L=int(L), gamma=float(g), sigma_epsilon=float(s))
        rej = pick(cc, mode="combined_postselect", L=int(L), gamma=float(g), sigma_epsilon=float(s))
        if edfl is None or rej is None:
            continue
        groups.setdefault((s, L), []).append(
            f"$L{{=}}{int(L)}$, {gamma_pct_label(g)}, $\\sigma_\\epsilon = {s:g}$ & {pm(edfl['alpha_asymptotic'], edfl['sem_asymptotic'])} & {pct(edfl['converged_pct'])} & "
            f"{pm(rej['alpha_asymptotic'], rej['sem_asymptotic'])} & {pct(rej['converged_pct'])} & {pct(rej['accept_pct'], nd=1)} \\\\")
    write_rows(outdir / "tab_combined.tex", "Configuration & ED-FL alpha & Converged (%) & Rejection alpha & Converged (%) & Accept. (%)  (asymptotic window; rows grouped by sigma then L)", midrule_groups(groups))


def table_sequential(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "sequential"]).sort_values(["sigma_epsilon", "L"])
    groups: dict[float, list[str]] = {}
    for _, r in cc.iterrows():
        s = float(r["sigma_epsilon"])
        label = f"$L{{=}}{int(r['L'])}$, $\\sigma_\\epsilon = {s:g}$" + (" (noiseless)" if s == 0 and r["L"] == 1 else "")
        groups.setdefault(s, []).append(f"{label} & {pm(r['alpha_full'], r['sem_full'])} & {pct(r['converged_pct'])} & {int(r['n_seeds_full'])} \\\\")
    write_rows(outdir / "tab_sequential.tex", "Configuration & alpha & Converged (%) & Seeds  (full window, min_points 10; Seeds = seeds retained)", midrule_groups(groups))


def table_bare(cells, outdir):
    cc = clean_cells(cells)
    cc = cc[cc["protocol"] == "bare_adaptive"].sort_values("gamma")
    rows = []
    for _, r in cc.iterrows():
        label = "Noiseless" if r["gamma"] == 0 else f"$\\gamma = {100 * r['gamma']:g}\\%$"
        rows.append(f"{label} & {pm(r['alpha_full'], r['sem_full'])} & {pct(r['converged_pct'], nd=1)} & {int(r['n_seeds_full'])} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'])} \\\\")
    write_rows(outdir / "tab_bare_within_model.tex", "Configuration & alpha (full) & Converged (%) & Seeds & alpha (asymptotic)  (bare GHZ, adaptive N)", rows)


def table_bare_fixed(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "bare_fixed"]).sort_values(["L", "gamma"])
    groups: dict[int, list[str]] = {}
    for _, r in cc.iterrows():
        groups.setdefault(int(r["L"]), []).append(
            f"$L{{=}}{int(r['L'])}$, $\\gamma = {r['gamma']:.2f}$ & {pm(r['alpha_full'], r['sem_full'])} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'])} & {pct(r['converged_pct'])} & {int(r['n_seeds_full'])} \\\\")
    write_rows(outdir / "tab_bare_fixed.tex", "Configuration & alpha (full) & alpha (asymptotic) & Converged (%) & Seeds  (bare GHZ, fixed N = 2L+1)", midrule_groups(groups))


def table_realistic(cells, outdir):
    real = cells[(cells["q"] > 0) | (cells["k"] < 1) | (cells["F"] < 1)]
    for kind, mask, col in (("readout", real["q"] > 0, "q"), ("prep", real["q"] == 0, "k")):
        sub = real[mask].sort_values(["protocol", "mode", "L", col, "estimator"])
        rows = []
        for _, r in sub.iterrows():
            label = r["mode"].replace("_", "\\_") + ("" if r["protocol"] == "bare_adaptive" else f", $L{{=}}{int(r['L'])}$")
            par = f"{r['q']:g}" if kind == "readout" else f"{r['k']:g}, $F = {r['F']:g}$"
            rows.append(f"{label} & {par} & {r['estimator']} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'])} & {pct(r['converged_pct'])} & {r['C']:.0f} & {int(r['n_seeds_asymptotic'])} \\\\")
        write_rows(outdir / f"tab_realistic_{kind}.tex", f"Protocol & {col} & Estimator & alpha (asymptotic) & Converged (%) & C & Seeds", rows)


def table_window(hw, outdir):
    rows = []
    for _, r in hw.sort_values(["sigma_epsilon", "N"]).iterrows():
        rows.append(f"{r['sigma_epsilon']:g} & {int(r['N'])} & {r['T_mean']:.3g} & {r['T_median']:.3g} & {pct(r['converged_pct'])} & {r['N_star']:.0f} & {r['N_star_half']:.0f} \\\\")
    write_rows(outdir / "tab_window.tex", "sigma & N & T mean & T median & Converged (%) & N* & N*/2  (converged seeds only)", rows)


def write_tables(cells, hw, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    table_prefactor(cells, outdir)
    table_ed(cells, outdir)
    table_combined(cells, outdir)
    table_sequential(cells, outdir)
    table_bare(cells, outdir)
    table_bare_fixed(cells, outdir)
    table_realistic(cells, outdir)
    if not hw.empty:
        table_window(hw, outdir)


# --------------------------------------------------------------------------- main
def print_summary(cells: pd.DataFrame, hw: pd.DataFrame, df: pd.DataFrame) -> None:
    print(f"rows={len(df):,}  duplicates dropped={df.attrs['n_duplicates_dropped']}  cells={len(cells)}")
    show = ["protocol", "mode", "L", "gamma", "sigma_epsilon", "q", "k", "estimator",
            "alpha_full", "sem_full", "n_seeds_full", "alpha_asymptotic", "sem_asymptotic", "n_seeds_asymptotic", "shift_full_minus_asym", "converged_pct", "C", "accept_pct"]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", lambda x: f"{x:.4g}"):
        print(cells[show].to_string(index=False))
        if not hw.empty:
            print(hw[["sigma_epsilon", "N", "T_mean", "T_median", "converged_pct", "n_converged", "N_star", "N_star_half"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None, help="directory of suite CSVs (default results/v2)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results_dir = args.results_dir or V2_DIR
    df = load_results(results_dir)
    cells = analyze_cells(df)
    hw = analyze_hwindow(df)
    fs = fisher_saturation(df)
    nums = build_numbers(cells, hw, fs, df)

    derived = results_dir / "derived"
    derived.mkdir(exist_ok=True)
    cells.to_csv(derived / "cells.csv", index=False, float_format="%.6g")
    if not hw.empty:
        hw.to_csv(derived / "hwindow_summary.csv", index=False, float_format="%.6g")
    if not fs.empty:
        fs.to_csv(derived / "fisher.csv", index=False, float_format="%.6g")
    (results_dir / "numbers.json").write_text(json.dumps(nums, indent=1) + "\n")   # indent 1: one key per line for readable diffs, minimal file size
    write_tables(cells, hw, results_dir / "tables")
    if not args.quiet:
        print_summary(cells, hw, df)
    print(f"wrote {results_dir / 'numbers.json'} ({len(nums)} keys), {derived}/*.csv, {results_dir / 'tables'}/*.tex")


if __name__ == "__main__":
    main()
