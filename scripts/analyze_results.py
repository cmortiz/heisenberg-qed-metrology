#!/usr/bin/env python3
"""
Compute scaling exponents from bare-GHZ simulation results.

Usage: python analyze_results.py [--csv PATH] [--outdir PATH]
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path


def _fit_alphas_per_seed(df_config: pd.DataFrame, seed_col: str = "seed",
                         eps_max: float = np.inf, min_points: int = 10) -> list[float]:
    """Fit α per seed from converged data, skipping seeds with < min_points."""
    seeds = sorted(df_config[seed_col].unique())
    alphas = []

    for seed in seeds:
        mask = (
            (df_config[seed_col] == seed)
            & (df_config["converged"] == True)
            & (df_config["epsilon"] < eps_max)
        )
        sd = df_config[mask]
        if len(sd) < min_points:
            continue
        log_eps = np.log(sd["epsilon"].values)
        log_T = np.log(sd["total_resources"].values)
        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        coeffs = np.linalg.lstsq(A, log_T, rcond=None)[0]
        alphas.append(-coeffs[0])

    return alphas


def _summarize_alphas(alphas: list[float], n_conv: int, n_total: int) -> dict:
    """Mean, std, SEM, and 95% CI (t-distribution) from per-seed alphas."""
    if len(alphas) == 0:
        return {
            "mean": np.nan, "std": np.nan, "sem": np.nan, "ci95": np.nan,
            "n_seeds": 0, "per_seed_alphas": [],
            "convergence_rate": 0.0,
        }

    arr = np.array(alphas)
    n = len(arr)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0

    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "sem": sem,
        "ci95": float(t_crit * sem),
        "n_seeds": n,
        "per_seed_alphas": arr.tolist(),
        "convergence_rate": float(n_conv / n_total) if n_total > 0 else 0.0,
    }


def compute_alpha(df_config: pd.DataFrame, seed_col: str = "seed") -> dict:
    """Full-range, asymptotic (ε<0.01), and deep-asymptotic (ε<0.001) scaling fits."""
    n_conv = int(df_config["converged"].sum())
    n_total = len(df_config)

    # Full-range fit
    full_alphas = _fit_alphas_per_seed(df_config, seed_col, eps_max=np.inf, min_points=10)
    full = _summarize_alphas(full_alphas, n_conv, n_total)

    # Asymptotic fit (ε < 0.01)
    asym_mask = df_config["epsilon"] < 0.01
    n_conv_asym = int((df_config["converged"] & asym_mask).sum())
    n_total_asym = int(asym_mask.sum())
    asym_alphas = _fit_alphas_per_seed(df_config, seed_col, eps_max=0.01, min_points=5)
    asym = _summarize_alphas(asym_alphas, n_conv_asym, n_total_asym)

    # Deep-asymptotic fit (ε < 0.001)
    deep_mask = df_config["epsilon"] < 0.001
    n_conv_deep = int((df_config["converged"] & deep_mask).sum())
    n_total_deep = int(deep_mask.sum())
    deep_alphas = _fit_alphas_per_seed(df_config, seed_col, eps_max=0.001, min_points=3)
    deep = _summarize_alphas(deep_alphas, n_conv_deep, n_total_deep)

    return {"full": full, "asymptotic": asym, "deep_asymptotic": deep}


def config_label(mode: str, gamma: float, h: float) -> str:
    """Human-readable config label."""
    if gamma == 0:
        return f"Perfect ({mode})"
    if h == 0:
        return f"γ={gamma:.0%} homo ({mode})"
    return f"γ={gamma:.0%}±{h:.0%} het ({mode})"


def config_key(mode: str, gamma: float, h: float) -> str:
    """Machine-readable config key."""
    return f"{mode}_g{gamma:.4f}_h{h:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("--csv", default="results/bare_ghz.csv",
                        help="Input CSV path")
    parser.add_argument("--outdir", default="results", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} rows loaded")
    print(f"  Modes: {sorted(df['mode'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Gamma values: {sorted(df['gamma'].unique())}")
    print(f"  H values: {sorted(df['h'].unique())}")
    print()

    # Group by (mode, gamma, h)
    configs = df.groupby(["mode", "gamma", "h"])
    results = []

    print(f"{'Config':<30s}  {'α full (±SEM)':<16s}  {'α asym (±SEM)':<16s}  {'Conv%':>6s}  {'Seeds':>5s}")
    print("-" * 85)

    for (mode, gamma, h), group in sorted(configs):
        stats = compute_alpha(group)
        label = config_label(mode, gamma, h)
        key = config_key(mode, gamma, h)

        full = stats["full"]
        asym = stats["asymptotic"]
        deep = stats["deep_asymptotic"]

        results.append({
            "key": key,
            "label": label,
            "mode": mode,
            "gamma": gamma,
            "h": h,
            # Full-range
            "alpha_mean": full["mean"],
            "alpha_std": full["std"],
            "alpha_sem": full["sem"],
            "alpha_ci95": full["ci95"],
            "n_seeds": full["n_seeds"],
            "convergence_rate": full["convergence_rate"],
            "per_seed_alphas": full["per_seed_alphas"],
            # Asymptotic (ε < 0.01)
            "alpha_asymptotic_mean": asym["mean"],
            "alpha_asymptotic_sem": asym["sem"],
            "alpha_asymptotic_ci95": asym["ci95"],
            "alpha_asymptotic_n_seeds": asym["n_seeds"],
            "alpha_asymptotic_convergence_rate": asym["convergence_rate"],
            # Deep-asymptotic (ε < 0.001)
            "alpha_deep_mean": deep["mean"],
            "alpha_deep_sem": deep["sem"],
            "alpha_deep_n_seeds": deep["n_seeds"],
        })

        full_str = f"{full['mean']:.3f} ± {full['sem']:.3f}" if not np.isnan(full["mean"]) else "N/A"
        asym_str = f"{asym['mean']:.3f} ± {asym['sem']:.3f}" if not np.isnan(asym["mean"]) else "N/A"

        print(f"  {label:<28s}  {full_str:<16s}  {asym_str:<16s}"
              f"  {full['convergence_rate']:5.1%}  {full['n_seeds']:>5d}")

    print()

    # Summary by mode
    for mode in sorted(df["mode"].unique()):
        mode_results = [r for r in results if r["mode"] == mode and not np.isnan(r["alpha_mean"])]
        if mode_results:
            all_alphas = [a for r in mode_results for a in r["per_seed_alphas"]]
            print(f"  {mode.upper()} overall: α = {np.mean(all_alphas):.3f} ± {np.std(all_alphas, ddof=1):.3f} "
                  f"(n={len(all_alphas)} seed-configs)")

    # GHZ noiseless: highlight asymptotic convergence
    ghz_perfect = [r for r in results if r["mode"] == "ghz" and r["gamma"] == 0]
    if ghz_perfect:
        r = ghz_perfect[0]
        print()
        print(f"  GHZ noiseless convergence toward Heisenberg (α=1):")
        print(f"    Full range:       α = {r['alpha_mean']:.3f} ± {r['alpha_sem']:.3f}")
        print(f"    Asymptotic:       α = {r['alpha_asymptotic_mean']:.3f} ± {r['alpha_asymptotic_sem']:.3f}")
        print(f"    Deep-asymptotic:  α = {r['alpha_deep_mean']:.3f} ± {r['alpha_deep_sem']:.3f}")

    # GHZ vs Product comparison
    product_any = [r for r in results if r["mode"] == "product" and not np.isnan(r["alpha_mean"])]
    if ghz_perfect and product_any:
        print()
        prod_alphas = [a for r in product_any for a in r["per_seed_alphas"]]
        print(f"  Product avg: α = {np.mean(prod_alphas):.3f} ± {np.std(prod_alphas, ddof=1):.3f}"
              f"  (theory: 2.0; grid artifact gives ~1.3)")

    # Save JSON
    json_path = outdir / "analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON: {json_path}")

    # Save summary CSV (exclude per_seed_alphas list for flat format)
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_seed_alphas"} for r in results])
    csv_path = outdir / "analysis_summary.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
