#!/usr/bin/env python3
"""
Compute scaling exponents from bare-GHZ Hamiltonian-noise sweep.

Groups by (L, gamma, h) so within-Hamiltonian comparison against the ED
tables (error-detected post-select / full-likelihood at matched L, gamma)
can be read directly.

Usage: uv run python analyze_bare_hamiltonian.py [--csv PATH] [--outdir PATH]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from analyze_results import _fit_alphas_per_seed, _summarize_alphas


def compute_alpha_by_L(df_config: pd.DataFrame) -> dict:
    """Full-range and asymptotic α fits for one (L, gamma, h) slice."""
    n_conv = int(df_config["converged"].sum())
    n_total = len(df_config)

    full_alphas = _fit_alphas_per_seed(df_config, eps_max=np.inf, min_points=10)
    full = _summarize_alphas(full_alphas, n_conv, n_total)

    asym_mask = df_config["epsilon"] < 0.01
    n_conv_asym = int((df_config["converged"] & asym_mask).sum())
    n_total_asym = int(asym_mask.sum())
    asym_alphas = _fit_alphas_per_seed(df_config, eps_max=0.01, min_points=5)
    asym = _summarize_alphas(asym_alphas, n_conv_asym, n_total_asym)

    return {"full": full, "asymptotic": asym}


def main():
    parser = argparse.ArgumentParser(description="Analyze bare-Hamiltonian sweep")
    parser.add_argument("--csv", default="results/bare_hamiltonian.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows")
    print(f"  L values: {sorted(df['L'].unique())}")
    print(f"  Gamma values: {sorted(df['gamma'].unique())}")
    print(f"  H values: {sorted(df['h'].unique())}")
    print(f"  Seeds: {len(df['seed'].unique())}")
    print()

    results = []
    print(f"{'Config':<28s}  {'α full (±SEM)':<18s}  {'α asym (±SEM)':<18s}  {'Conv%':>6s}  {'Seeds':>5s}")
    print("-" * 90)

    for (L, gamma, h), group in sorted(df.groupby(["L", "gamma", "h"])):
        stats = compute_alpha_by_L(group)
        full, asym = stats["full"], stats["asymptotic"]

        label = f"L={L}, γ={gamma:.0%}"
        if h > 0:
            label += f"±{h:.0%}"

        results.append({
            "mode": "bare_hamiltonian",
            "L": int(L),
            "gamma": float(gamma),
            "h": float(h),
            "label": label,
            "alpha_full_mean": full["mean"],
            "alpha_full_sem": full["sem"],
            "alpha_full_ci95": full["ci95"],
            "alpha_full_n_seeds": full["n_seeds"],
            "alpha_asym_mean": asym["mean"],
            "alpha_asym_sem": asym["sem"],
            "alpha_asym_n_seeds": asym["n_seeds"],
            "convergence_rate": full["convergence_rate"],
            "per_seed_alphas_full": full["per_seed_alphas"],
        })

        full_str = f"{full['mean']:.3f} ± {full['sem']:.3f}" if not np.isnan(full["mean"]) else "N/A"
        asym_str = f"{asym['mean']:.3f} ± {asym['sem']:.3f}" if not np.isnan(asym["mean"]) else "N/A"
        print(f"  {label:<26s}  {full_str:<18s}  {asym_str:<18s}"
              f"  {full['convergence_rate']:5.1%}  {full['n_seeds']:>5d}")

    json_path = outdir / "bare_hamiltonian_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    flat = [{k: v for k, v in r.items() if k != "per_seed_alphas_full"} for r in results]
    csv_path = outdir / "bare_hamiltonian_summary.csv"
    pd.DataFrame(flat).to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
