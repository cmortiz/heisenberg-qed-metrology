#!/usr/bin/env python3
"""B3 lemma test: quenched-truth + marg-inference vs quenched (oracle) and marg.

Loads three CSVs:
    combined_full_likelihood.csv         (mode == 'combined_full_likelihood', oracle)
    combined_full_likelihood_marg.csv    (mode == 'combined_full_likelihood_marg', shot truth + marg infer)
    combined_full_likelihood_marg_qt.csv (mode == 'combined_full_likelihood_marg_qt', quenched truth + marg infer)

For matched (L, gamma, sigma_epsilon) cells, fits asymptotic alpha (epsilon < 1e-2)
per seed, averages, and compares.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

EPS_ASYMPTOTIC = 1e-2


def fit_alpha_per_seed(df, eps_max=EPS_ASYMPTOTIC):
    sub = df[(df["epsilon"] < eps_max) & (df["converged"] == True)]
    if sub.empty:
        return np.nan, np.nan, 0
    alphas = []
    for _, g in sub.groupby("seed"):
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
    df_q = df_q[df_q["mode"] == "combined_full_likelihood"]
    df_m = pd.read_csv(base / "combined_full_likelihood_marg.csv")
    df_m = df_m[df_m["mode"] == "combined_full_likelihood_marg"]
    df_qt = pd.read_csv(base / "combined_full_likelihood_marg_qt.csv")
    df_qt = df_qt[df_qt["mode"] == "combined_full_likelihood_marg_qt"]

    print(f"oracle (combined_full_likelihood):           {len(df_q):>6} rows")
    print(f"marg-shot (combined_full_likelihood_marg):    {len(df_m):>6} rows")
    print(f"marg-qt (combined_full_likelihood_marg_qt):   {len(df_qt):>6} rows")

    keys = ["L", "gamma", "sigma_epsilon"]
    cells = sorted(set(map(tuple, df_qt[keys].drop_duplicates().to_numpy())))

    print(f"\n=== Asymptotic alpha (epsilon < 1e-2) ===\n")
    print(f"{'L':>2} {'gamma':>6} {'sigma':>6} {'N*sig2':>7} | "
          f"{'a_oracle':>8}{'+-':>2}{'sem_o':>5} | "
          f"{'a_marg_qt':>9}{'+-':>2}{'sem_qt':>5} | "
          f"{'a_marg_shot':>11}{'+-':>2}{'sem_m':>5} | "
          f"{'Δ(qt-or)':>8} {'Δ(m-or)':>8}")
    print("-" * 130)
    rows = []
    for L, g, s in cells:
        sub_q = df_q[(df_q["L"] == L) & (df_q["gamma"] == g) & (df_q["sigma_epsilon"] == s)]
        sub_m = df_m[(df_m["L"] == L) & (df_m["gamma"] == g) & (df_m["sigma_epsilon"] == s)]
        sub_qt = df_qt[(df_qt["L"] == L) & (df_qt["gamma"] == g) & (df_qt["sigma_epsilon"] == s)]
        if sub_qt.empty:
            continue
        N_total = int(sub_qt["N_total"].iloc[0])
        a_q, e_q, n_q = fit_alpha_per_seed(sub_q) if not sub_q.empty else (np.nan, np.nan, 0)
        a_qt, e_qt, n_qt = fit_alpha_per_seed(sub_qt)
        a_m, e_m, n_m = fit_alpha_per_seed(sub_m) if not sub_m.empty else (np.nan, np.nan, 0)
        Ns2 = N_total * s * s
        d_qt = a_qt - a_q if not np.isnan(a_q) else np.nan
        d_m = a_m - a_q if not np.isnan(a_q) else np.nan
        print(f"{L:>2} {g:>6.2f} {s:>6.2f} {Ns2:>7.4f} | "
              f"{a_q:>8.3f} +-{e_q:>5.3f} | "
              f"{a_qt:>9.3f} +-{e_qt:>5.3f} | "
              f"{a_m:>11.3f} +-{e_m:>5.3f} | "
              f"{d_qt:>+8.3f} {d_m:>+8.3f}")
        rows.append({"L": L, "gamma": g, "sigma_epsilon": s, "N_total": N_total, "N_sigma2": Ns2,
                     "alpha_oracle": a_q, "sem_oracle": e_q, "n_oracle": n_q,
                     "alpha_marg_qt": a_qt, "sem_marg_qt": e_qt, "n_marg_qt": n_qt,
                     "alpha_marg_shot": a_m, "sem_marg_shot": e_m, "n_marg_shot": n_m,
                     "delta_qt": d_qt, "delta_marg": d_m})

    out = pd.DataFrame(rows)
    out_path = base / "lemma_test_qt.csv"
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
