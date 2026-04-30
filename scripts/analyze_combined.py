#!/usr/bin/env python3
"""
Unified analysis: scaling exponents and convergence sweeps for all protocols.

Outputs: summary CSV/JSON (Tables II-IV in the paper), alpha-convergence sweep.
Expects results/{bare_ghz,ed_postselect,combined_postselect}.csv.

Usage: python analyze_combined.py [--outdir PATH]
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path


# --- Core fitting ---

def fit_alpha_single_seed(df_seed: pd.DataFrame, eps_max: float = np.inf,
                          min_points: int = 5) -> float | None:
    """Fit alpha from converged data for a single seed.

    Returns alpha or None if insufficient data.
    """
    mask = (df_seed["converged"] == True) & (df_seed["epsilon"] < eps_max)
    sd = df_seed[mask]
    if len(sd) < min_points:
        return None
    log_eps = np.log(sd["epsilon"].values)
    log_T = np.log(sd["total_resources"].values)
    A = np.vstack([log_eps, np.ones(len(log_eps))]).T
    coeffs = np.linalg.lstsq(A, log_T, rcond=None)[0]
    return -coeffs[0]


def fit_alphas_per_seed(df_config: pd.DataFrame, eps_max: float = np.inf,
                        min_points: int = 5) -> list[float]:
    """Fit alpha per seed, returning list of valid alphas."""
    alphas = []
    for seed in sorted(df_config["seed"].unique()):
        sd = df_config[df_config["seed"] == seed]
        alpha = fit_alpha_single_seed(sd, eps_max, min_points)
        if alpha is not None:
            alphas.append(alpha)
    return alphas


def summarize_alphas(alphas: list[float], n_conv: int, n_total: int) -> dict:
    """Compute mean, SEM, 95% CI from per-seed alphas."""
    if len(alphas) == 0:
        return {"mean": np.nan, "sem": np.nan, "ci95": np.nan,
                "n_seeds": 0, "per_seed_alphas": [],
                "convergence_rate": 0.0}
    arr = np.array(alphas)
    n = len(arr)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "sem": sem,
        "ci95": float(t_crit * sem),
        "n_seeds": n,
        "per_seed_alphas": arr.tolist(),
        "convergence_rate": float(n_conv / n_total) if n_total > 0 else 0.0,
    }


def compute_alpha_summary(df_config: pd.DataFrame) -> dict:
    """Compute full, asymptotic, and deep-asymptotic alpha for a config."""
    n_conv = int(df_config["converged"].sum())
    n_total = len(df_config)

    full = summarize_alphas(
        fit_alphas_per_seed(df_config, np.inf, 10), n_conv, n_total)
    asym = summarize_alphas(
        fit_alphas_per_seed(df_config, 0.01, 5),
        int((df_config["converged"] & (df_config["epsilon"] < 0.01)).sum()),
        int((df_config["epsilon"] < 0.01).sum()))
    deep = summarize_alphas(
        fit_alphas_per_seed(df_config, 0.001, 3),
        int((df_config["converged"] & (df_config["epsilon"] < 0.001)).sum()),
        int((df_config["epsilon"] < 0.001).sum()))

    return {"full": full, "asymptotic": asym, "deep_asymptotic": deep}


# --- Alpha convergence sweep ---

def alpha_convergence_sweep(df_config: pd.DataFrame, label: str,
                            n_points: int = 25) -> list[dict]:
    """Sweep epsilon_max from 10^-1 down to the minimum available epsilon.

    At each cutoff, refit alpha using only data with epsilon < eps_max.
    Returns list of dicts with eps_max, alpha_mean, alpha_sem, n_seeds.
    """
    eps_values = df_config.loc[df_config["converged"], "epsilon"].values
    if len(eps_values) == 0:
        return []

    eps_min_data = eps_values.min()
    eps_max_data = eps_values.max()

    # Sweep from full range down to ~3x the minimum data point
    log_min = np.log10(max(eps_min_data * 3, 1e-4))
    log_max = np.log10(min(eps_max_data, 0.1))

    if log_min >= log_max:
        return []

    eps_maxes = np.logspace(log_max, log_min, n_points)
    sweep = []

    for em in eps_maxes:
        alphas = fit_alphas_per_seed(df_config, eps_max=em, min_points=3)
        if len(alphas) >= 2:
            arr = np.array(alphas)
            n = len(arr)
            sem = float(np.std(arr, ddof=1) / np.sqrt(n))
            sweep.append({
                "label": label,
                "eps_max": float(em),
                "alpha_mean": float(np.mean(arr)),
                "alpha_sem": sem,
                "n_seeds": n,
            })

    return sweep


# --- Config labels ---

def error_detected_label(L: int, gamma: float) -> str:
    """Label for error-detected config."""
    N = 2 * L + 1
    if gamma == 0:
        return f"ED L={L} (N={N}) noiseless"
    return f"ED L={L} (N={N}) gamma={gamma:.0%}"


def combined_label(L: int, gamma: float, sigma: float) -> str:
    """Label for combined protocol config."""
    N_total = 3 * (2 * L + 1)
    return f"Comb L={L} (N={N_total}) gamma={gamma:.0%} sigma={sigma}"


def ghz_label(gamma: float, h: float) -> str:
    """Label for bare-GHZ config."""
    if gamma == 0:
        return "GHZ noiseless"
    if h == 0:
        return f"GHZ gamma={gamma:.0%}"
    return f"GHZ gamma={gamma:.0%} h={h:.0%}"


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Unified simulation analysis")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--ghz-csv", default=None,
                        help="Bare-GHZ CSV (default: auto-detect)")
    parser.add_argument("--ed-csv", default=None,
                        help="Error-detected CSV (default: results/ed_postselect.csv)")
    parser.add_argument("--combined-csv", default=None,
                        help="Combined protocol CSV (default: results/combined_postselect.csv)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # --- Auto-detect data files ---
    ghz_path = Path(args.ghz_csv) if args.ghz_csv else None
    if ghz_path is None:
        candidates = [
            script_dir / "results" / "bare_ghz.csv",
            script_dir.parent / "sanity-check" / "scripts" / "results" / "bare_ghz.csv",
        ]
        for c in candidates:
            if c.exists():
                ghz_path = c
                break

    ed_path = Path(args.ed_csv) if args.ed_csv else script_dir / "results" / "ed_postselect.csv"
    combined_path = Path(args.combined_csv) if args.combined_csv else script_dir / "results" / "combined_postselect.csv"

    all_results = []
    convergence_sweeps = []

    # --- 1. Bare-GHZ analysis ---
    if ghz_path and ghz_path.exists():
        print(f"=== Bare-GHZ: {ghz_path} ===")
        df_ghz = pd.read_csv(ghz_path)
        print(f"  {len(df_ghz):,} rows, {len(df_ghz['seed'].unique())} seeds")

        for (mode, gamma, h), group in sorted(df_ghz.groupby(["mode", "gamma", "h"])):
            if mode != "ghz":
                continue
            stats = compute_alpha_summary(group)
            label = ghz_label(gamma, h)
            full = stats["full"]
            asym = stats["asymptotic"]

            all_results.append({
                "source": "bare_ghz", "label": label,
                "mode": "ghz", "gamma": gamma, "h": h,
                "L": None, "sigma_epsilon": None,
                "alpha_full": full["mean"], "alpha_full_sem": full["sem"],
                "alpha_asym": asym["mean"], "alpha_asym_sem": asym["sem"],
                "convergence_rate": full["convergence_rate"],
                "n_seeds": full["n_seeds"],
                "acceptance_rate": None,
            })

            full_str = f"{full['mean']:.3f} +/- {full['sem']:.3f}" if not np.isnan(full["mean"]) else "N/A"
            asym_str = f"{asym['mean']:.3f} +/- {asym['sem']:.3f}" if not np.isnan(asym["mean"]) else "N/A"
            print(f"  {label:<35s}  full: {full_str:<16s}  asym: {asym_str}")

            # Alpha convergence for noiseless GHZ
            if gamma == 0:
                sweep = alpha_convergence_sweep(group, label)
                convergence_sweeps.extend(sweep)

        print()
    else:
        print("  Bare-GHZ data not found, skipping.\n")

    # --- 2. Error-detected analysis ---
    if ed_path.exists():
        print(f"=== Error-detected: {ed_path} ===")
        df_ed = pd.read_csv(ed_path)
        print(f"  {len(df_ed):,} rows, {len(df_ed['seed'].unique())} seeds")

        for (L, gamma), group in sorted(df_ed.groupby(["L", "gamma"])):
            stats = compute_alpha_summary(group)
            label = error_detected_label(int(L), gamma)
            full = stats["full"]
            asym = stats["asymptotic"]
            avg_accept = group["acceptance_rate"].mean()

            all_results.append({
                "source": "error_detected", "label": label,
                "mode": "error_detected", "gamma": gamma, "h": 0.0,
                "L": int(L), "sigma_epsilon": None,
                "alpha_full": full["mean"], "alpha_full_sem": full["sem"],
                "alpha_asym": asym["mean"], "alpha_asym_sem": asym["sem"],
                "convergence_rate": full["convergence_rate"],
                "n_seeds": full["n_seeds"],
                "acceptance_rate": float(avg_accept),
            })

            full_str = f"{full['mean']:.3f} +/- {full['sem']:.3f}" if not np.isnan(full["mean"]) else "N/A"
            asym_str = f"{asym['mean']:.3f} +/- {asym['sem']:.3f}" if not np.isnan(asym["mean"]) else "N/A"
            print(f"  {label:<35s}  full: {full_str:<16s}  asym: {asym_str}  accept: {avg_accept:.1%}")

            # Alpha convergence for all error-detected configs
            sweep = alpha_convergence_sweep(group, label)
            convergence_sweeps.extend(sweep)

        print()
    else:
        print("  Error-detected data not found, skipping.\n")

    # --- 3. Combined protocol analysis ---
    if combined_path.exists():
        print(f"=== Combined protocol: {combined_path} ===")
        df_comb = pd.read_csv(combined_path)
        print(f"  {len(df_comb):,} rows, {len(df_comb['seed'].unique())} seeds")

        for (L, gamma, sigma), group in sorted(df_comb.groupby(["L", "gamma", "sigma_epsilon"])):
            stats = compute_alpha_summary(group)
            label = combined_label(int(L), gamma, sigma)
            full = stats["full"]
            asym = stats["asymptotic"]
            avg_accept = group["acceptance_rate"].mean()

            all_results.append({
                "source": "combined", "label": label,
                "mode": "combined", "gamma": gamma, "h": 0.0,
                "L": int(L), "sigma_epsilon": sigma,
                "alpha_full": full["mean"], "alpha_full_sem": full["sem"],
                "alpha_asym": asym["mean"], "alpha_asym_sem": asym["sem"],
                "convergence_rate": full["convergence_rate"],
                "n_seeds": full["n_seeds"],
                "acceptance_rate": float(avg_accept),
            })

            full_str = f"{full['mean']:.3f} +/- {full['sem']:.3f}" if not np.isnan(full["mean"]) else "N/A"
            asym_str = f"{asym['mean']:.3f} +/- {asym['sem']:.3f}" if not np.isnan(asym["mean"]) else "N/A"
            print(f"  {label:<40s}  full: {full_str:<16s}  asym: {asym_str}  accept: {avg_accept:.1%}")

            # Alpha convergence for combined configs
            sweep = alpha_convergence_sweep(group, label)
            convergence_sweeps.extend(sweep)

        print()
    else:
        print("  Combined data not found, skipping.\n")

    # --- Save outputs ---
    if all_results:
        summary_df = pd.DataFrame(all_results)
        csv_path = outdir / "analysis_unified.csv"
        summary_df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"Saved unified summary: {csv_path}")

        json_path = outdir / "analysis_unified.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved unified JSON:    {json_path}")

    if convergence_sweeps:
        sweep_df = pd.DataFrame(convergence_sweeps)
        sweep_path = outdir / "alpha_convergence_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False, float_format="%.6f")
        print(f"Saved convergence sweep: {sweep_path}")

        sweep_json_path = outdir / "alpha_convergence_sweep.json"
        with open(sweep_json_path, "w") as f:
            json.dump(convergence_sweeps, f, indent=2)
        print(f"Saved convergence JSON:  {sweep_json_path}")


if __name__ == "__main__":
    main()
