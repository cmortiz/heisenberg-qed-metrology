#!/usr/bin/env python3
"""B3 verification: compare quenched vs marg alpha for the combined protocol.

Loads:
    results/combined_full_likelihood.csv      (mode == 'combined_full_likelihood', quenched)
    results/combined_full_likelihood_marg.csv (mode == 'combined_full_likelihood_marg')

For each (L, gamma, sigma_epsilon) cell, fits log(total_resources) ~ alpha * log(1/epsilon)
on epsilon < EPS_ASYMPTOTIC, averages across seeds, and reports alpha_quenched vs alpha_marg.
The lemma claim from sensing.tex Sec 9.1.3 is that the two agree to leading order when
N_total * sigma_epsilon**2 = o(1).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

EPS_ASYMPTOTIC = 1e-2


def fit_alpha_per_seed(df, eps_max=EPS_ASYMPTOTIC):
    """Per-seed alpha = -slope(log T vs log eps), averaged across seeds."""
    sub = df[(df["epsilon"] < eps_max) & (df["converged"] == True)].copy()
    if sub.empty:
        return np.nan, np.nan, 0
    alphas = []
    for seed, g in sub.groupby("seed"):
        if len(g) < 3:
            continue
        x = np.log(1.0 / g["epsilon"].to_numpy())
        y = np.log(g["total_resources"].to_numpy())
        slope, _, _, _, _ = sp_stats.linregress(x, y)
        alphas.append(slope)
    if not alphas:
        return np.nan, np.nan, 0
    a = np.array(alphas)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0, len(a)


def main():
    base = Path(__file__).parent / "results"
    df_q = pd.read_csv(base / "combined_full_likelihood.csv")
    df_m = pd.read_csv(base / "combined_full_likelihood_marg.csv")

    df_q = df_q[df_q["mode"] == "combined_full_likelihood"]
    df_m = df_m[df_m["mode"] == "combined_full_likelihood_marg"]

    keys = ["L", "gamma", "sigma_epsilon"]
    rows = []
    common_cells = sorted(set(map(tuple, df_q[keys].drop_duplicates().to_numpy())) &
                          set(map(tuple, df_m[keys].drop_duplicates().to_numpy())))

    print(f"Found {len(common_cells)} cells in both quenched and marg.")
    print("\n=== Quenched vs marg asymptotic alpha (epsilon < 1e-2) ===")
    fmt = "{:>3} {:>6} {:>6} {:>5}    a_q={:.3f}+-{:.3f}  a_m={:.3f}+-{:.3f}  N*s2={:.4f}"
    for L, g, s in common_cells:
        sub_q = df_q[(df_q["L"] == L) & (df_q["gamma"] == g) & (df_q["sigma_epsilon"] == s)]
        sub_m = df_m[(df_m["L"] == L) & (df_m["gamma"] == g) & (df_m["sigma_epsilon"] == s)]
        if sub_q.empty or sub_m.empty:
            continue
        N_total = int(sub_q["N_total"].iloc[0])
        a_q, e_q, n_q = fit_alpha_per_seed(sub_q)
        a_m, e_m, n_m = fit_alpha_per_seed(sub_m)
        Ns2 = N_total * s * s
        print(fmt.format(L, g, s, N_total, a_q, e_q, a_m, e_m, Ns2))
        rows.append({"L": L, "gamma": g, "sigma_epsilon": s, "N_total": N_total,
                     "alpha_quenched": a_q, "sem_quenched": e_q, "n_quenched": n_q,
                     "alpha_marg": a_m, "sem_marg": e_m, "n_marg": n_m,
                     "delta_alpha": a_m - a_q, "N_sigma2": Ns2})
    out = pd.DataFrame(rows)
    out_path = base / "quenched_vs_marg.csv"
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved: {out_path}")

    if not out.empty:
        in_regime = out[out["N_sigma2"] < 0.05]
        if not in_regime.empty:
            print(f"\nN*sigma2 < 0.05 cells: max |delta_alpha| = {in_regime['delta_alpha'].abs().max():.4f}")
        out_regime = out[out["N_sigma2"] >= 0.05]
        if not out_regime.empty:
            print(f"N*sigma2 >= 0.05 cells: max |delta_alpha| = {out_regime['delta_alpha'].abs().max():.4f}")


if __name__ == "__main__":
    main()
