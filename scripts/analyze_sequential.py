#!/usr/bin/env python3
"""
Compute scaling exponents for the sequential binary-search protocol
(Table tab:sequential-results in the paper).

Reuses analyze_results.py's exact fit convention so the sequential table is
produced identically to the bare-GHZ / error-detected tables: per-seed OLS of
log T = -alpha log eps_tar + c on converged rows, averaged over seeds (SEM).
Full-range uses min_points=10; the asymptotic (eps_tar<1e-2) window uses
min_points=5 (and is data-starved at sigma_eps=0.05, hence full-range is the
reported window for this table).

Usage: python analyze_sequential.py [--csv results/sequential.csv]
"""

import argparse
import numpy as np
import pandas as pd

from analyze_results import compute_alpha


def main():
    parser = argparse.ArgumentParser(description="Analyze sequential-protocol results")
    parser.add_argument("--csv", default="results/sequential.csv", help="Input CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows | sigma_epsilon={sorted(df['sigma_epsilon'].unique())} "
          f"| L={sorted(df['L'].unique())} | seeds={df['seed'].nunique()}\n")

    header = (f"{'Configuration':<26s}  {'alpha full (+/-SEM)':<20s}  "
              f"{'alpha asym (+/-SEM)':<20s}  {'Conv%':>6s}  {'Seeds(full)':>11s}")
    print(header)
    print("-" * len(header))

    for (sigma, L), group in df.groupby(["sigma_epsilon", "L"]):
        stats = compute_alpha(group)
        full, asym = stats["full"], stats["asymptotic"]
        label = f"L={L}, sigma_eps={sigma:g}" + (" (noiseless)" if sigma == 0 else "")
        full_str = (f"{full['mean']:.3f} +/- {full['sem']:.3f}"
                    if not np.isnan(full["mean"]) else "N/A")
        asym_str = (f"{asym['mean']:.3f} +/- {asym['sem']:.3f} (n={asym['n_seeds']})"
                    if not np.isnan(asym["mean"]) else "N/A")
        print(f"  {label:<24s}  {full_str:<20s}  {asym_str:<20s}  "
              f"{full['convergence_rate']:5.1%}  {full['n_seeds']:>11d}")


if __name__ == "__main__":
    main()
