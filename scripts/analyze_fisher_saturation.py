#!/usr/bin/env python3
"""
Per-cell Fisher information saturation analysis.

Compares empirical per-shot Fisher information accumulated during the ED-FL and
combined-FL Bayesian phase-estimation runs to the QCRB ceiling implied by
Theorem~\\ref{thm:bitflip}, F_Q = 4N^2(1 - O((gamma/omega)^2)).

For each round i with depth M_i and N_eff_i = N - d_i clean qubits the simulators
log
    sum_F_obs = sum_i 4 * M_i^2 * (N_eff_i)^2,
    sum_M_sq  = sum_i M_i^2,
which the joint estimator's per-shot Fisher information saturates at the
syndrome-conditioned operating point (4 M^2 N_eff^2 per shot, the squared
omega-derivative of phi_eff = (N-d) M omega).

The QCRB ceiling per (seed, epsilon) cell, with N = N_code (ED) or N_total
(combined) and homogeneous gamma, is
    F_QCRB = sum_M_sq * 4 * N^2 * (1 - (gamma/omega)^2),
and the saturation ratio is sum_F_obs / F_QCRB. A noiseless ceiling
F_QCRB_ideal = sum_M_sq * 4 * N^2 is reported alongside as a diagnostic for
unpacking how much of the gap is the (1 - (gamma/omega)^2) damping vs.
the (N-d_i)/N reduction from realized errors.

Inputs (auto-detected in scripts/results/):
    ed_full_likelihood.csv        -- ED-FL (Theorem 9, Algorithm 2)
    combined_full_likelihood.csv  -- combined-FL (Theorem 16)
Outputs:
    results/fisher_saturation.csv (per-config summary + per-seed ratios)

Asymptotic regime defaults to epsilon < 1e-2, matching the alpha-fit window of
analyze_combined.py.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats


OMEGA_TRUE = 0.3
EPS_ASYMPTOTIC = 1e-2


def per_cell_ratios(group: pd.DataFrame, N: int, gamma: float,
                    omega: float = OMEGA_TRUE) -> pd.DataFrame:
    """Return per-(seed, epsilon) saturation ratios for a single config slice."""
    g = group[group["converged"] == True].copy()
    if g.empty:
        return g
    correction = max(1.0 - (gamma / omega) ** 2, 1e-12)
    g["F_QCRB_ideal"] = g["sum_M_sq"] * 4.0 * (N ** 2)
    g["F_QCRB"] = g["F_QCRB_ideal"] * correction
    g["sat_ratio_ideal"] = g["sum_F_obs"] / g["F_QCRB_ideal"]
    g["sat_ratio"] = g["sum_F_obs"] / g["F_QCRB"]
    return g


def summarize_seed_means(cell_df: pd.DataFrame, eps_max: float
                         ) -> dict[str, float]:
    """Average per-seed saturation ratios over epsilon < eps_max, then over seeds."""
    sub = cell_df[cell_df["epsilon"] < eps_max]
    if sub.empty:
        return {"sat_mean": np.nan, "sat_sem": np.nan,
                "sat_ideal_mean": np.nan, "sat_ideal_sem": np.nan,
                "n_seeds": 0, "n_cells": 0}
    seed_means = sub.groupby("seed")[["sat_ratio", "sat_ratio_ideal"]].mean()
    n = len(seed_means)
    if n == 0:
        return {"sat_mean": np.nan, "sat_sem": np.nan,
                "sat_ideal_mean": np.nan, "sat_ideal_sem": np.nan,
                "n_seeds": 0, "n_cells": 0}
    sat = seed_means["sat_ratio"].to_numpy()
    sat_ideal = seed_means["sat_ratio_ideal"].to_numpy()
    sem = float(np.std(sat, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    sem_ideal = float(np.std(sat_ideal, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "sat_mean": float(np.mean(sat)),
        "sat_sem": sem,
        "sat_ideal_mean": float(np.mean(sat_ideal)),
        "sat_ideal_sem": sem_ideal,
        "n_seeds": n,
        "n_cells": int(len(sub)),
    }


def analyze_protocol(df: pd.DataFrame, protocol: str, N_col: str,
                     group_keys: list[str], eps_max: float
                     ) -> pd.DataFrame:
    """Compute per-config summaries for one protocol's CSV."""
    rows = []
    for keys, g in df.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cfg = dict(zip(group_keys, keys))
        N = int(g[N_col].iloc[0])
        gamma = float(cfg["gamma"])
        cell_df = per_cell_ratios(g, N=N, gamma=gamma)
        if cell_df.empty:
            continue
        summary = summarize_seed_means(cell_df, eps_max=eps_max)
        row = {
            "protocol": protocol,
            "N": N,
            **{k: cfg[k] for k in group_keys},
            **summary,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        return
    print(f"\n=== {title} ===")
    cols_print = ["protocol"]
    for k in ("L", "gamma", "sigma_epsilon"):
        if k in df.columns:
            cols_print.append(k)
    cols_print += ["N", "sat_mean", "sat_sem", "sat_ideal_mean",
                   "sat_ideal_sem", "n_seeds", "n_cells"]
    cols_print = [c for c in cols_print if c in df.columns]
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(df[cols_print].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ed-csv", default=None,
                        help="ED-FL CSV (default: results/ed_full_likelihood.csv)")
    parser.add_argument("--combined-csv", default=None,
                        help="Combined-FL CSV (default: results/combined_full_likelihood.csv)")
    parser.add_argument("--outdir", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--eps-max", type=float, default=EPS_ASYMPTOTIC,
                        help="Asymptotic regime cutoff (default: 1e-2)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    ed_path = Path(args.ed_csv) if args.ed_csv else script_dir / "results" / "ed_full_likelihood.csv"
    combined_path = Path(args.combined_csv) if args.combined_csv else script_dir / "results" / "combined_full_likelihood.csv"

    all_summaries = []

    if ed_path.exists() and ed_path.stat().st_size > 0:
        df_ed = pd.read_csv(ed_path)
        if "sum_F_obs" not in df_ed.columns:
            print(f"ED-FL: {ed_path} missing sum_F_obs/sum_M_sq columns; "
                  f"re-run simulation_error_detected.py to populate them.")
        else:
            df_ed = df_ed[df_ed["mode"] == "full_likelihood"]  # exclude cond-only ablation
            ed_summary = analyze_protocol(
                df_ed, protocol="ED-FL", N_col="N_code",
                group_keys=["L", "gamma"], eps_max=args.eps_max,
            )
            all_summaries.append(ed_summary)
            print_table(ed_summary, "ED-FL Fisher saturation")
    else:
        print(f"ED-FL data not found at {ed_path}; skipping.")

    if combined_path.exists() and combined_path.stat().st_size > 0:
        df_comb = pd.read_csv(combined_path)
        if "sum_F_obs" not in df_comb.columns:
            print(f"Combined-FL: {combined_path} missing sum_F_obs/sum_M_sq columns; "
                  f"re-run simulation_combined.py to populate them.")
        else:
            df_comb = df_comb[df_comb["mode"] == "combined_full_likelihood"]
            comb_summary = analyze_protocol(
                df_comb, protocol="Comb-FL", N_col="N_total",
                group_keys=["L", "gamma", "sigma_epsilon"], eps_max=args.eps_max,
            )
            all_summaries.append(comb_summary)
            print_table(comb_summary, "Combined-FL Fisher saturation")
    else:
        print(f"Combined-FL data not found at {combined_path}; skipping.")

    if all_summaries:
        merged = pd.concat(all_summaries, ignore_index=True, sort=False)
        out_csv = outdir / "fisher_saturation.csv"
        merged.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"\nSaved Fisher-saturation summary: {out_csv}")
        worst = merged.sort_values("sat_mean").head(3)
        print("\nWorst three saturation ratios (vs Theorem 9 QCRB):")
        print(worst[["protocol", "L", "gamma",
                     *(c for c in ["sigma_epsilon"] if c in merged.columns),
                     "N", "sat_mean", "sat_sem", "n_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
