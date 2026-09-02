#!/usr/bin/env python3
"""Analyze suite CSVs: per-cell fits, numbers.json (one key per paper element), LaTeX table rows.

Usage
    uv run python scripts/analyze.py                       # results/v2 (auto-detects layout)
    uv run python scripts/analyze.py --v1                  # committed scripts/results (legacy headers)
    uv run python scripts/analyze.py --compare v1          # also write results/v2/compare_v1.md
    uv run python scripts/analyze.py --results-dir DIR     # any directory in either layout

Outputs (under the results directory)
    derived/cells.csv            one row per (GROUP_KEYS) cell: both fit windows, shift, prefactor, acceptance
    derived/hwindow_summary.csv  one row per (sigma_epsilon, N): T mean/median over seeds (hwindow protocol only)
    derived/fisher.csv           per-cell Fisher-saturation ratios (ED-FL and combined-FL)
    numbers.json       paper elements; keys documented in results/numbers_schema.md
    tables/*.tex       table rows in the paper's column order (paste into the tabular bodies)
    compare_v1.md      with --compare v1

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
from eqsp.config import CSV_COLUMNS, GROUP_KEYS, OMEGA_TRUE  # noqa: E402

V1_DIR = REPO / "scripts" / "results"
V2_DIR = REPO / "results" / "v2"

# legacy file stem -> (v2 protocol, v2 mode label); mode values inside the files differ from the stems
V1_FILES = {
    "bare_ghz": ("bare_depol", "ghz"),
    "bare_hamiltonian": ("bare_fixed", "bare_hamiltonian"),
    "ed_full_likelihood": ("ed", "ed_full_likelihood"),
    "ed_postselect": ("ed", "ed_postselect"),
    "combined_full_likelihood": ("combined", "combined_full_likelihood"),
    "combined_postselect": ("combined", "combined_postselect"),
    "combined_full_likelihood_marg": ("combined", "combined_full_likelihood_marg"),
    "sequential": ("sequential", "sequential"),
}
PROTOCOLS = ["bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow"]   # v2 files: results/v2/<protocol>.csv
# sentinel for group keys that a protocol does not use (NaN keys would drop rows in groupby)
KEY_FILL = {"gamma": 0.0, "h": 0.0, "sigma_epsilon": 0.0, "L": 0, "q": 0.0, "k": 1.0, "F": 1.0,
            "estimator": "mismatched", "model": "legacy"}
TAB_PREFACTOR_ROWS = [(1, 0.01), (1, 0.05), (1, 0.10), (2, 0.10), (3, 0.10)]
FIG1_BARE_GAMMAS = [0.01, 0.05, 0.10]
FIG3_CELLS = [(1, 0.01, 0.01), (1, 0.10, 0.01), (2, 0.10, 0.01)]
WEAK_NOISE_P = [0.01, 0.05, 0.10]
SEQ_EPS_QUOTE = 1e-2
COMBINED_TABLE_SIGMAS = [0.01, 0.05]   # tab:combined-results cells; the v1 CSV also holds the cut sigma scan


# --------------------------------------------------------------------------- loading
def v2_files(results_dir: Path) -> list[Path]:
    return [results_dir / f"{p}.csv" for p in PROTOCOLS if (results_dir / f"{p}.csv").exists()]


def is_v2_layout(results_dir: Path) -> bool:
    for p in v2_files(results_dir):
        return "protocol" in p.open().readline().strip().split(",")
    return False


def load_v1(results_dir: Path) -> pd.DataFrame:
    frames = []
    for stem, (protocol, mode) in V1_FILES.items():
        path = results_dir / f"{stem}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if stem == "bare_ghz":
            df = df[df["mode"] == "ghz"]
        df = df.assign(protocol=protocol, mode=mode, source_csv=path.name)
        frames.append(df)
    if not frames:
        sys.exit(f"no legacy CSVs found in {results_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_v2(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in v2_files(results_dir):
        frames.append(pd.read_csv(path).assign(source_csv=path.name))
    if not frames:
        sys.exit(f"no v2 CSVs found in {results_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Common column set, sentinel-filled group keys, bool converged, deduplicated (keep last)."""
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col, val in KEY_FILL.items():
        df[col] = df[col].fillna(val)
    df["L"] = df["L"].astype(int)
    df["converged"] = fit.coerce_converged(df["converged"])
    dedupe_keys = GROUP_KEYS + ["seed", "epsilon"] + (["N"] if "N" in df.columns else [])
    df["N"] = df["N"].fillna(-1)
    before = len(df)
    df = df.drop_duplicates(subset=dedupe_keys, keep="last").reset_index(drop=True)
    df.attrs["n_duplicates_dropped"] = before - len(df)
    return df


def load_results(results_dir: Path, v1: bool) -> pd.DataFrame:
    df = load_v1(results_dir) if v1 or not is_v2_layout(results_dir) else load_v2(results_dir)
    df = normalize(df)
    df.attrs["layout"] = "v1" if v1 or not is_v2_layout(results_dir) else "v2"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with no readout/prep noise and the mismatched (plain) estimator."""
    return df[(df["q"] == 0) & (df["k"] == 1) & (df["F"] == 1) & (df["estimator"] == "mismatched")]


# --------------------------------------------------------------------------- per-cell analysis
def analyze_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in df[df["protocol"] != "hwindow"].groupby(GROUP_KEYS, sort=True):
        fits = fit.fit_both_windows(cell)
        pf = fit.prefactor(cell)
        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "source_csv": cell["source_csv"].iloc[0],
            "n_seeds_run": int(cell["seed"].nunique()),
            "n_rows": int(len(cell)),
            "converged_pct": fit.converged_pct(cell),
            "shift_full_minus_asym": fit.window_shift(fits),
            "C": pf.C, "sem_logC": pf.sem_logC, "n_rows_C": pf.n_rows,
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

    Recipe lifted from scripts/analyze_fisher_saturation.py:46-85 (v1-jax-suite): converged rows,
    per-seed mean of the ratio over the asymptotic window, then mean and SEM over seeds.
    seed_min is the smallest per-seed mean (the "dips to 0.97 for individual seeds" figure).
    """
    sub = clean(df[df["mode"].isin(["ed_full_likelihood", "combined_full_likelihood"])])
    sub = sub[sub["sum_F_obs"].notna() & sub["converged"]]
    rows = []
    for keys, g in sub.groupby(GROUP_KEYS, sort=True):
        row = dict(zip(GROUP_KEYS, keys))
        N = int(g["N_total"].iloc[0]) if row["protocol"] == "combined" else int(g["N_code"].iloc[0])
        correction = max(1.0 - (row["gamma"] / OMEGA_TRUE) ** 2, 1e-12)
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


def sequential_schedule_depth(eps_tar: float, w0: float = math.pi, c_m: float = math.pi / 2) -> int:
    """Deepest executed round depth of the legacy binary search (simulation_sequential.py:76-84 replayed)."""
    width, deepest = w0, 1
    for _ in range(200):
        if width <= 2.0 * eps_tar:
            break
        deepest = max(1, math.ceil(c_m / width))
        width /= 2.0
    return deepest


# --------------------------------------------------------------------------- numbers.json helpers
def gk(g): return f"g{g:.2f}"
def sk(s): return f"s{s:.2f}"
def qk(q): return f"q{q:g}"
def kk(k): return f"k{k:.3f}"


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
    }


def add_bundle(nums: dict, prefix: str, bundle: dict) -> None:
    for name, entry in bundle.items():
        nums[f"{prefix}.{name}"] = entry


def pick(cells: pd.DataFrame, **cond) -> pd.Series | None:
    mask = np.ones(len(cells), dtype=bool)
    for col, val in cond.items():
        mask &= np.isclose(cells[col], val) if isinstance(val, (int, float)) else (cells[col] == val).to_numpy()
    hit = cells[mask]
    return hit.iloc[0] if len(hit) == 1 else None


def clean_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return cells[(cells["q"] == 0) & (cells["k"] == 1) & (cells["F"] == 1) & (cells["estimator"] == "mismatched") & (cells["h"] == 0)]


# --------------------------------------------------------------------------- numbers.json builders
def numbers_bare(cells: pd.DataFrame, nums: dict) -> None:
    """Bare GHZ, adaptive N (v2: bare_adaptive; v1: depolarizing bare_ghz.csv), by gamma. Window: full."""
    bare = clean_cells(cells)
    bare = bare[bare["protocol"].isin(["bare_adaptive", "bare_depol"])]
    for _, row in bare.iterrows():
        add_bundle(nums, f"tab_bare_within_model.{gk(row['gamma'])}", cell_entries(row, "full"))
        nums[f"fig2.alpha.bare_{gk(row['gamma'])}"] = alpha_entry(row, "asymptotic")
        nums[f"fig2.conv.bare_{gk(row['gamma'])}"] = nums[f"tab_bare_within_model.{gk(row['gamma'])}.converged_pct"]
        if any(np.isclose(row["gamma"], g) for g in FIG1_BARE_GAMMAS):
            nums[f"fig1.alpha.bare_{gk(row['gamma'])}"] = alpha_entry(row, "full")


def numbers_bare_fixed(cells: pd.DataFrame, nums: dict) -> None:
    """Bare GHZ at fixed N = 2L+1 under Hamiltonian noise (the bare column of tab:prefactor-results).
    Window: full (the text quotes full-range alpha); emitted for every bare_fixed cell, ED counterpart or not."""
    for _, row in clean_cells(cells[cells["protocol"] == "bare_fixed"]).iterrows():
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
    """tab:prefactor-results: C for bare_fixed, ED-FL, rejection at matched (L, gamma) and their ratios."""
    cc = clean_cells(cells)
    ratios_edfl, ratios_rej = {}, {}
    for _, bare in cc[cc["protocol"] == "bare_fixed"].iterrows():
        L, g = int(bare["L"]), float(bare["gamma"])
        edfl = pick(cc, protocol="ed", mode="ed_full_likelihood", L=L, gamma=g)
        rej = pick(cc, protocol="ed", mode="ed_postselect", L=L, gamma=g)
        if edfl is None and rej is None:
            continue
        p = f"tab_prefactor.L{L}_{gk(g)}"
        for name, row, store in (("edfl", edfl, ratios_edfl), ("rej", rej, ratios_rej)):
            if row is None:
                continue
            b = cell_entries(row, "asymptotic")
            nums[f"{p}.{name}_C"] = b["C"]
            nums[f"{p}.{name}_alpha_asym"] = b["alpha"]
            nums[f"{p}.{name}_window_shift"] = b["window_shift"]
            ratio = row["C"] / bare["C"]
            nums[f"{p}.{name}_over_bare"] = scalar_entry(ratio, row["source_csv"], f"C_{name} / C_bare at matched (L, gamma)")
            nums[f"{p}.bare_over_{name}"] = scalar_entry(1.0 / ratio, row["source_csv"], f"C_bare / C_{name} (the 'x times' figure)")
            store[(L, g)] = 1.0 / ratio
    for name, store in (("edfl", ratios_edfl), ("rej", ratios_rej)):
        table = [v for (L, g), v in store.items() if any(L == tL and np.isclose(g, tg) for tL, tg in TAB_PREFACTOR_ROWS)]
        for suffix, vals in (("", table), ("_all", list(store.values()))):
            if vals:
                nums[f"text.prefactor.bare_over_{name}_min{suffix}"] = scalar_entry(min(vals), None, f"min of C_bare/C_{name} over {'the 5 table rows' if not suffix else 'all matched (L, gamma) cells'}")
                nums[f"text.prefactor.bare_over_{name}_max{suffix}"] = scalar_entry(max(vals), None, f"max of C_bare/C_{name} over {'the 5 table rows' if not suffix else 'all matched (L, gamma) cells'}")


def numbers_combined(cells: pd.DataFrame, nums: dict) -> None:
    """tab:combined-results (asymptotic), fig3 legend alphas, prefactor ratios, per-block acceptance."""
    cc = clean_cells(cells)
    comb = cc[cc["protocol"] == "combined"]
    ratios = {}
    for _, row in comb.iterrows():
        mode = {"combined_full_likelihood": "edfl", "combined_postselect": "rej", "combined_full_likelihood_marg": "marg"}[row["mode"]]
        L, g, s = int(row["L"]), float(row["gamma"]), float(row["sigma_epsilon"])
        cell = f"L{L}_{gk(g)}_{sk(s)}"
        add_bundle(nums, f"tab_combined.{cell}.{mode}", cell_entries(row, "asymptotic"))
        nums[f"tab_combined.{cell}.{mode}.accept_pct"] = scalar_entry(row["accept_pct"], row["source_csv"], "mean acceptance_rate over all rows of the cell")
        if mode == "rej":
            nums[f"text.combined.per_block_accept_pct_{cell}"] = scalar_entry(
                100.0 * (row["accept_pct"] / 100.0) ** (1.0 / 3.0), row["source_csv"],
                "cube root of the joint (three-block) acceptance rate, in percent")
            edfl = pick(cc, protocol="combined", mode="combined_full_likelihood", L=L, gamma=g, sigma_epsilon=s)
            if edfl is not None:
                ratios[cell] = row["C"] / edfl["C"]
                nums[f"text.combined.prefactor_ratio_{cell}"] = scalar_entry(ratios[cell], row["source_csv"], "C_rejection / C_EDFL for the combined protocol (the paper's '3 to 26x')")
        if mode == "edfl":
            bare = pick(cc, protocol="bare_fixed", L=L, gamma=g)
            if bare is not None:
                nums[f"text.combined.prefactor_ratio_edfl_over_bare_fixed_{cell}"] = scalar_entry(
                    row["C"] / bare["C"], row["source_csv"],
                    "C_EDFL(combined, N_total = 3(2L+1)) / C_bare(fixed N = 2L+1) at the same (L, gamma); probe sizes differ")
            if any(L == fL and np.isclose(g, fg) and np.isclose(s, fs) for fL, fg, fs in FIG3_CELLS):
                nums[f"fig3.alpha.combined_{cell}"] = alpha_entry(row, "asymptotic")
    if ratios:
        nums["text.combined.prefactor_ratio_min"] = scalar_entry(min(ratios.values()), None, "min over combined cells of C_rejection / C_EDFL")
        nums["text.combined.prefactor_ratio_max"] = scalar_entry(max(ratios.values()), None, "max over combined cells of C_rejection / C_EDFL")


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
    if "M_exec_max" in at.columns and at["M_exec_max"].notna().any():
        m_exec = float(at["M_exec_max"].max())
        note = f"max of the M_exec_max column over seeds and cells at eps_tar = {eps_q:.6g} (nearest grid point to 1e-2)"
    else:
        m_exec = float(sequential_schedule_depth(eps_q))
        note = (f"v1 fallback: no M_exec_max column; deepest depth of the deterministic schedule "
                f"M_r = ceil((pi/2)/W_r), W_r = pi/2^r, replayed at eps_tar = {eps_q:.6g} (simulation_sequential.py:76-84)")
    nums["text.sequential.M_exec_max_eps1e-2"] = scalar_entry(m_exec, src, note)
    if "M_max_used" in at.columns and at["M_max_used"].notna().any():
        nums["text.sequential.M_max_used_eps1e-2"] = scalar_entry(
            float(at["M_max_used"].max()), src,
            f"legacy CSV column M_max_used = ceil((pi/2)/(2 eps sqrt(N))), max over cells at eps_tar = {eps_q:.6g}; "
            "not the executed depth (simulation_sequential.py:130)")
    nums["text.sequential.eps_quoted"] = scalar_entry(eps_q, src, "grid point used for the depth quotes")
    for s in sorted(seq["sigma_epsilon"].unique()):
        if s > 0:
            nums[f"text.sequential.M2sigma2_{sk(s)}_eps1e-2"] = scalar_entry(
                m_exec**2 * s**2, src, "M_exec_max^2 * sigma_epsilon^2 at the quoted eps")


def numbers_realistic(cells: pd.DataFrame, nums: dict) -> None:
    """Readout-error (q > 0) and prep-infidelity (k < 1 or F < 1) cells for every protocol. Window: asymptotic."""
    real = cells[(cells["q"] > 0) | (cells["k"] < 1) | (cells["F"] < 1)]
    for _, row in real.iterrows():
        proto = {"bare_adaptive": "bare", "bare_fixed": "bare_fixed", "ed": "ed", "combined": "combined", "sequential": "sequential"}[row["protocol"]]
        if row["mode"] in ("ed_postselect", "combined_postselect"):
            proto += "_rej"
        Lpart = "" if row["protocol"] == "bare_adaptive" else f"_L{row['L']}"
        if row["q"] > 0:
            key = f"tab_realistic.readout.{proto}{Lpart}_{qk(row['q'])}_{row['estimator']}"
        else:
            key = f"tab_realistic.prep.{proto}{Lpart}_{kk(row['k'])}_{row['estimator']}"
        add_bundle(nums, key, cell_entries(row, "asymptotic"))
        nums[f"{key}.F"] = scalar_entry(row["F"], row["source_csv"], "single-qubit init fidelity of the cell")
    for q in sorted(real.loc[real["q"] > 0, "q"].unique()):
        nums[f"text.readout.n_breakdown_{qk(q)}"] = scalar_entry(1.0 / (2.0 * q), None, "probe size n = 1/(2q) where the bare parity visibility (1-2q)^n reaches about e^-1")
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
        ok = g[(g["converged_pct"] >= 50.0) & g["T_mean"].notna()]
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


def numbers_shift(cells: pd.DataFrame, nums: dict) -> None:
    """text.fit.window_shift_max: largest |alpha_full - alpha_asymptotic| over the ED and combined table cells."""
    cc = clean_cells(cells[cells["shift_full_minus_asym"].notna()])
    if cc.empty:
        return
    edc = cc[cc["protocol"].isin(["ed", "combined"]) & (cc["mode"] != "combined_full_likelihood_marg")]
    if not edc.empty:
        worst = edc.loc[edc["shift_full_minus_asym"].abs().idxmax()]
        nums["text.fit.window_shift_max"] = scalar_entry(
            abs(worst["shift_full_minus_asym"]), worst["source_csv"],
            f"largest |alpha_full - alpha_asymptotic| over the ED and combined (ED-FL and rejection) cells; attained at "
            f"{worst['mode']} L={worst['L']} gamma={worst['gamma']} sigma={worst['sigma_epsilon']}")
    for proto, g in cc.groupby("protocol"):
        nums[f"text.fit.window_shift_max_{proto}"] = scalar_entry(g["shift_full_minus_asym"].abs().max(), None, f"largest |alpha_full - alpha_asymptotic| over {proto} cells (clean configs)")


def build_numbers(cells, hw, fs, df) -> dict:
    nums: dict = {}
    numbers_bare(cells, nums)
    numbers_bare_fixed(cells, nums)
    numbers_ed(cells, nums)
    numbers_prefactor(cells, nums)
    numbers_combined(cells, nums)
    numbers_sequential(cells, df, nums)
    numbers_realistic(cells, nums)
    numbers_hwindow(hw, nums)
    numbers_fisher(fs, nums)
    numbers_weak_noise(nums)
    numbers_shift(cells, nums)
    return dict(sorted(nums.items()))


# --------------------------------------------------------------------------- LaTeX rows
def pm(a, s, nd):
    return "--" if a is None or np.isnan(a) else f"${a:.{nd}f} \\pm {s:.{nd}f}$"


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
    for _, bare in cc[cc["protocol"] == "bare_fixed"].sort_values(["L", "gamma"]).iterrows():
        L, g = int(bare["L"]), float(bare["gamma"])
        edfl = pick(cc, protocol="ed", mode="ed_full_likelihood", L=L, gamma=g)
        rej = pick(cc, protocol="ed", mode="ed_postselect", L=L, gamma=g)
        if edfl is None or rej is None:
            continue
        groups.setdefault(L, []).append(
            f"$L{{=}}{L}$, $\\gamma_{{\\mathrm{{Ham}}}} = {g:.2f}$ & ${bare['C']:.0f}$ & $\\mathbf{{{edfl['C']:.0f}}}$ & ${rej['C']:.0f}$ & ${sig2(edfl['C'] / bare['C'])}$ \\\\")
    write_rows(outdir / "tab_prefactor.tex", "Configuration & Bare GHZ & ED-FL & Rejection sampling & ED-FL/bare  (C = geometric mean of T*eps over converged rows, full range)", midrule_groups(groups))


def table_ed(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "ed"])
    groups: dict[int, list[str]] = {}
    for (L, g), _ in cc.groupby(["L", "gamma"], sort=True):
        edfl = pick(cc, mode="ed_full_likelihood", L=int(L), gamma=float(g))
        rej = pick(cc, mode="ed_postselect", L=int(L), gamma=float(g))
        if edfl is None or rej is None:
            continue
        groups.setdefault(int(L), []).append(
            f"$L{{=}}{int(L)}$, {gamma_pct_label(g)} & {pm(edfl['alpha_asymptotic'], edfl['sem_asymptotic'], 2)} & {pct(edfl['converged_pct'])} & "
            f"{pm(rej['alpha_asymptotic'], rej['sem_asymptotic'], 2)} & {pct(rej['converged_pct'])} & {pct(rej['accept_pct'], gt99=True)} \\\\")
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
            f"$L{{=}}{int(L)}$, {gamma_pct_label(g)}, $\\sigma_\\epsilon = {s:g}$ & {pm(edfl['alpha_asymptotic'], edfl['sem_asymptotic'], 2)} & {pct(edfl['converged_pct'])} & "
            f"{pm(rej['alpha_asymptotic'], rej['sem_asymptotic'], 2)} & {pct(rej['converged_pct'])} & {pct(rej['accept_pct'], nd=1)} \\\\")
    write_rows(outdir / "tab_combined.tex", "Configuration & ED-FL alpha & Converged (%) & Rejection alpha & Converged (%) & Accept. (%)  (asymptotic window; rows grouped by sigma then L)", midrule_groups(groups))


def table_sequential(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "sequential"]).sort_values(["sigma_epsilon", "L"])
    groups: dict[float, list[str]] = {}
    for _, r in cc.iterrows():
        s = float(r["sigma_epsilon"])
        label = f"$L{{=}}{int(r['L'])}$, $\\sigma_\\epsilon = {s:g}$" + (" (noiseless)" if s == 0 and r["L"] == 1 else "")
        groups.setdefault(s, []).append(f"{label} & {pm(r['alpha_full'], r['sem_full'], 3)} & {pct(r['converged_pct'])} & {int(r['n_seeds_full'])} \\\\")
    write_rows(outdir / "tab_sequential.tex", "Configuration & alpha & Converged (%) & Seeds  (full window, min_points 10; Seeds = seeds retained)", midrule_groups(groups))


def table_bare(cells, outdir):
    cc = clean_cells(cells)
    cc = cc[cc["protocol"].isin(["bare_adaptive", "bare_depol"])].sort_values("gamma")
    rows = []
    for _, r in cc.iterrows():
        sym = "p_{\\mathrm{dep}}" if r["protocol"] == "bare_depol" else "\\gamma_{\\mathrm{Ham}}"
        label = "Noiseless" if r["gamma"] == 0 else f"${sym} = {100 * r['gamma']:g}\\%$"
        rows.append(f"{label} & {pm(r['alpha_full'], r['sem_full'], 2)} & {pct(r['converged_pct'], nd=1)} & {int(r['n_seeds_full'])} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'], 2)} \\\\")
    write_rows(outdir / "tab_bare_within_model.tex", "Configuration & alpha (full) & Converged (%) & Seeds & alpha (asymptotic)  (bare GHZ, adaptive N)", rows)


def table_bare_fixed(cells, outdir):
    cc = clean_cells(cells[cells["protocol"] == "bare_fixed"]).sort_values(["L", "gamma"])
    groups: dict[int, list[str]] = {}
    for _, r in cc.iterrows():
        groups.setdefault(int(r["L"]), []).append(
            f"$L{{=}}{int(r['L'])}$, $\\gamma_{{\\mathrm{{Ham}}}} = {r['gamma']:.2f}$ & {pm(r['alpha_full'], r['sem_full'], 3)} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'], 3)} & {pct(r['converged_pct'])} & {int(r['n_seeds_full'])} \\\\")
    write_rows(outdir / "tab_bare_fixed.tex", "Configuration & alpha (full) & alpha (asymptotic) & Converged (%) & Seeds  (bare GHZ, fixed N = 2L+1)", midrule_groups(groups))


def table_realistic(cells, outdir):
    real = cells[(cells["q"] > 0) | (cells["k"] < 1) | (cells["F"] < 1)]
    for kind, mask, col in (("readout", real["q"] > 0, "q"), ("prep", real["q"] == 0, "k")):
        sub = real[mask].sort_values(["protocol", "mode", "L", col, "estimator"])
        rows = []
        for _, r in sub.iterrows():
            label = r["mode"].replace("_", "\\_") + ("" if r["protocol"] == "bare_adaptive" else f", $L{{=}}{int(r['L'])}$")
            par = f"{r['q']:g}" if kind == "readout" else f"{r['k']:g}, $F = {r['F']:g}$"
            rows.append(f"{label} & {par} & {r['estimator']} & {pm(r['alpha_asymptotic'], r['sem_asymptotic'], 2)} & {pct(r['converged_pct'])} & {r['C']:.0f} & {int(r['n_seeds_asymptotic'])} \\\\")
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


# --------------------------------------------------------------------------- compare v1
def v1_counterpart(row: pd.Series, v1: pd.DataFrame) -> pd.Series | None:
    """Map a clean v2 cell to its legacy cell (plan: bare_fixed<->bare_hamiltonian, ed<->ed_*, combined<->combined_*,
    sequential<->sequential, bare_adaptive gamma=0 <-> bare_ghz gamma=0)."""
    proto, mode = row["protocol"], row["mode"]
    if proto == "bare_adaptive":
        return pick(v1, protocol="bare_depol", gamma=0.0, h=0.0) if np.isclose(row["gamma"], 0.0) else None
    if proto in ("bare_fixed", "ed"):
        return pick(v1, protocol=proto, mode=mode, L=int(row["L"]), gamma=float(row["gamma"]))
    if proto == "combined":
        return pick(v1, protocol=proto, mode=mode, L=int(row["L"]), gamma=float(row["gamma"]), sigma_epsilon=float(row["sigma_epsilon"]))
    if proto == "sequential":
        return pick(v1, protocol=proto, L=int(row["L"]), sigma_epsilon=float(row["sigma_epsilon"]))
    return None


def compare_v1(cells_v2: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    v1 = clean_cells(analyze_cells(load_results(V1_DIR, v1=True)))
    rows = []
    for _, r in clean_cells(cells_v2).iterrows():
        c = v1_counterpart(r, v1)
        if c is None:
            continue
        for w in fit.WINDOWS:
            diff = r[f"alpha_{w}"] - c[f"alpha_{w}"]
            sem_diff = math.hypot(r[f"sem_{w}"], c[f"sem_{w}"])
            rows.append({"protocol": r["protocol"], "mode": r["mode"], "L": r["L"], "gamma": r["gamma"], "sigma_epsilon": r["sigma_epsilon"],
                         "window": w, "alpha_v2": r[f"alpha_{w}"], "sem_v2": r[f"sem_{w}"], "n_seeds_v2": r[f"n_seeds_{w}"],
                         "alpha_v1": c[f"alpha_{w}"], "sem_v1": c[f"sem_{w}"], "n_seeds_v1": c[f"n_seeds_{w}"],
                         "diff": diff, "sem_diff": sem_diff, "flag": bool(abs(diff) > 2.0 * sem_diff) if sem_diff > 0 else bool(abs(diff) > 0),
                         "conv_v2": r["converged_pct"], "conv_v1": c["converged_pct"], "v1_csv": c["source_csv"]})
    cmp = pd.DataFrame(rows)
    lines = ["# v2 versus committed v1 (tag v1-jax-suite) per-cell alpha", "",
             "Flag rule: |alpha_v2 - alpha_v1| > 2 sqrt(sem_v2^2 + sem_v1^2). Both windows listed per cell; ",
             "Converged (%) over all rows. v1 cells deduplicated on (cell, seed, eps) keeping the last row.", ""]
    if cmp.empty:
        lines.append("No v2 cell has a v1 counterpart.")
    else:
        lines += [f"Cells compared: {len(cmp) // 2}; flagged rows: {int(cmp['flag'].sum())} of {len(cmp)}", "",
                  "| protocol | mode | L | gamma | sigma | window | alpha_v2 | alpha_v1 | diff | sem_diff | flag | conv_v2 | conv_v1 | v1 csv |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in cmp.iterrows():
            lines.append(f"| {r['protocol']} | {r['mode']} | {r['L']} | {r['gamma']:g} | {r['sigma_epsilon']:g} | {r['window']} | "
                         f"{r['alpha_v2']:.3f} +/- {r['sem_v2']:.3f} (n={r['n_seeds_v2']}) | {r['alpha_v1']:.3f} +/- {r['sem_v1']:.3f} (n={r['n_seeds_v1']}) | "
                         f"{r['diff']:+.3f} | {r['sem_diff']:.3f} | {'**YES**' if r['flag'] else 'no'} | {r['conv_v2']:.1f} | {r['conv_v1']:.1f} | {r['v1_csv']} |")
    out_path.write_text("\n".join(lines) + "\n")
    return cmp


# --------------------------------------------------------------------------- main
def print_summary(cells: pd.DataFrame, hw: pd.DataFrame, df: pd.DataFrame) -> None:
    print(f"layout={df.attrs['layout']}  rows={len(df):,}  duplicates dropped={df.attrs['n_duplicates_dropped']}  cells={len(cells)}")
    show = ["protocol", "mode", "L", "gamma", "sigma_epsilon", "q", "k", "estimator",
            "alpha_full", "sem_full", "n_seeds_full", "alpha_asymptotic", "sem_asymptotic", "n_seeds_asymptotic", "shift_full_minus_asym", "converged_pct", "C", "accept_pct"]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", lambda x: f"{x:.4g}"):
        print(cells[show].to_string(index=False))
        if not hw.empty:
            print(hw[["sigma_epsilon", "N", "T_mean", "T_median", "converged_pct", "n_converged", "N_star", "N_star_half"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None, help="default results/v2, or scripts/results with --v1")
    ap.add_argument("--v1", action="store_true", help="read the committed legacy CSVs (scripts/results) with their old headers")
    ap.add_argument("--compare", choices=["v1"], default=None, help="write compare_v1.md against the committed CSVs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results_dir = args.results_dir or (V1_DIR if args.v1 else V2_DIR)
    df = load_results(results_dir, v1=args.v1)
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
    (results_dir / "numbers.json").write_text(json.dumps(nums, indent=1) + "\n")
    write_tables(cells, hw, results_dir / "tables")
    if not args.quiet:
        print_summary(cells, hw, df)
    print(f"wrote {results_dir / 'numbers.json'} ({len(nums)} keys), {derived}/*.csv, {results_dir / 'tables'}/*.tex")
    if args.compare == "v1":
        if df.attrs["layout"] == "v1":
            sys.exit("--compare v1 needs a v2 results directory")
        cmp = compare_v1(cells, results_dir / "compare_v1.md")
        print(f"wrote {results_dir / 'compare_v1.md'}: {int(cmp['flag'].sum()) if len(cmp) else 0} flagged of {len(cmp)} rows")


if __name__ == "__main__":
    main()
