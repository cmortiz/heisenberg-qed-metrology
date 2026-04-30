"""Reanalysis of h=0 CSVs to find the cleanest bare-vs-ED separation axis.

Reads:
- bare_hamiltonian.csv
- ed_postselect.csv
- ed_full_likelihood.csv

All at h=0, fixed N=2L+1, γ_Ham ∈ {0.01, 0.05, 0.10}.

Computes per (protocol, L, γ):
- α scaling exponent (OLS on log T vs log ε, converged subset)
- T(ε) prefactor C where T = C · ε^(-α); report C at α=1 canonical scaling
- Convergence rate (overall and at small-ε tail)
- Precision floor (smallest ε with ≥ 50% seeds converging)
- Final-error distribution at unconverged ε

Output: reports/reanalysis_within_hamiltonian_20260423.md
"""

import os
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REPORT_PATH = os.path.join(REPO_ROOT, 'reports', 'reanalysis_within_hamiltonian_20260423.md')


def fit_alpha_per_seed(df):
    """Fit α = -slope(log T vs log ε) per seed. Returns mean α, SEM, and list per seed."""
    alphas = []
    intercepts = []
    for seed, sub in df.groupby('seed'):
        sub = sub[sub['converged']]
        if len(sub) < 4:
            continue
        log_eps = np.log(sub['epsilon'].values)
        log_T = np.log(sub['total_resources'].values)
        slope, intercept = np.polyfit(log_eps, log_T, 1)
        alphas.append(-slope)
        intercepts.append(intercept)
    if not alphas:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean_a = np.mean(alphas)
    sem_a = np.std(alphas, ddof=1) / np.sqrt(len(alphas)) if len(alphas) > 1 else np.nan
    mean_c = np.mean(intercepts)
    sem_c = np.std(intercepts, ddof=1) / np.sqrt(len(intercepts)) if len(intercepts) > 1 else np.nan
    return mean_a, sem_a, mean_c, sem_c, len(alphas)


def compute_prefactor_at_alpha_one(df):
    """
    Assume α ≈ 1 (Heisenberg). Fit T = C / ε with α locked at 1, compute geometric-mean C.
    C is the canonical Heisenberg prefactor: resources per precision unit.
    """
    sub = df[df['converged']].copy()
    if len(sub) == 0:
        return np.nan, np.nan, 0
    C_per_row = sub['total_resources'].values * sub['epsilon'].values
    log_C = np.log(C_per_row)
    mean_logC = np.mean(log_C)
    sem_logC = np.std(log_C, ddof=1) / np.sqrt(len(log_C))
    return np.exp(mean_logC), sem_logC, len(log_C)


def precision_floor(df, threshold=0.5):
    """Smallest ε where ≥ threshold fraction of seeds converged."""
    per_eps = df.groupby('epsilon')['converged'].mean()
    below = per_eps[per_eps >= threshold]
    if len(below) == 0:
        return np.nan
    return below.index.min()


def summarize(df_bare, df_ps, df_fl):
    protocols = [
        ('bare_hamiltonian', df_bare),
        ('ed_postselect', df_ps),
        ('ed_full_likelihood', df_fl),
    ]
    rows = []
    for proto_name, df in protocols:
        for L in sorted(df['L'].unique()):
            for gamma in sorted(df['gamma'].unique()):
                cut = df[(df['L'] == L) & (df['gamma'] == gamma)]
                if len(cut) == 0:
                    continue
                alpha, alpha_sem, log_c, log_c_sem, n_seeds = fit_alpha_per_seed(cut)
                C1, logC_sem, n_conv = compute_prefactor_at_alpha_one(cut)
                conv_rate = cut['converged'].mean()
                tail_cut = cut[cut['epsilon'] < 1e-3]
                tail_conv = tail_cut['converged'].mean() if len(tail_cut) else np.nan
                floor = precision_floor(cut)

                acc = cut['acceptance_rate'].mean() if 'acceptance_rate' in cut.columns else np.nan

                rows.append({
                    'protocol': proto_name,
                    'L': int(L),
                    'gamma': gamma,
                    'alpha': alpha,
                    'alpha_sem': alpha_sem,
                    'prefactor_C': C1,
                    'prefactor_logC_sem': logC_sem,
                    'convergence_rate': conv_rate,
                    'tail_convergence_rate_eps_lt_1e3': tail_conv,
                    'precision_floor': floor,
                    'acceptance_rate': acc,
                    'n_seeds_fitted': n_seeds,
                })
    return pd.DataFrame(rows)


def render_markdown(summary):
    lines = []
    lines.append("# Within-Hamiltonian Reanalysis (h=0)\n")
    lines.append("Date: 2026-04-23\n")
    lines.append("Source: existing CSVs `bare_hamiltonian.csv`, `ed_postselect.csv`, `ed_full_likelihood.csv`. ")
    lines.append("All at h=0, fixed N=2L+1, γ_Ham ∈ {0.01, 0.05, 0.10}, 40 seeds, 60 ε points.\n")
    lines.append("Purpose: identify the cleanest bare-vs-ED separation axis for the §8 supporting comparison, ")
    lines.append("given that α ≈ 1 for all protocols at these matched parameters (per verification pilot).\n")

    lines.append("## Summary: where does bare-vs-ED separate?\n")

    for gamma in sorted(summary['gamma'].unique()):
        lines.append(f"\n### γ_Ham = {gamma}\n")
        sub = summary[summary['gamma'] == gamma].copy()
        sub = sub.sort_values(['L', 'protocol'])
        lines.append("| Protocol | L | α | T·ε prefactor | Conv. | Conv. @ ε<1e-3 | Prec. floor | Accept. |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            a = f"{r['alpha']:.3f}±{r['alpha_sem']:.3f}" if not np.isnan(r['alpha_sem']) else f"{r['alpha']:.3f}"
            C = f"{r['prefactor_C']:.2e}" if not np.isnan(r['prefactor_C']) else "—"
            cr = f"{r['convergence_rate']*100:.0f}%"
            tc = f"{r['tail_convergence_rate_eps_lt_1e3']*100:.0f}%" if not np.isnan(r['tail_convergence_rate_eps_lt_1e3']) else "—"
            pf = f"{r['precision_floor']:.1e}" if not np.isnan(r['precision_floor']) else "—"
            ar = f"{r['acceptance_rate']*100:.0f}%" if not np.isnan(r['acceptance_rate']) else "—"
            lines.append(f"| {r['protocol']} | {r['L']} | {a} | {C} | {cr} | {tc} | {pf} | {ar} |")

    lines.append("\n## Reading the columns\n")
    lines.append("- **α**: scaling exponent from OLS fit of log T vs log ε per seed, averaged over seeds (SEM reported).\n")
    lines.append("- **T·ε prefactor**: geometric mean of (total_resources × ε) over converged rows. ")
    lines.append("Smaller is better; this is the 'resources per precision unit' at Heisenberg scaling.\n")
    lines.append("- **Conv.**: fraction of (seed, ε) cells that converged (reached |ω̂-ω| < 1.2ε in budget).\n")
    lines.append("- **Conv. @ ε<1e-3**: tail convergence rate at the hardest precision targets. Differentiates ")
    lines.append("protocols that hit a precision floor vs. those that succeed throughout.\n")
    lines.append("- **Prec. floor**: smallest ε where ≥50% of seeds converged. Protocols with ε_floor ≈ 10⁻⁴ ")
    lines.append("reach the full tested range; those with ε_floor > 10⁻³ hit a ceiling.\n")
    lines.append("- **Accept.**: acceptance rate for post-selection (post-select only; full-likelihood keeps all rounds).\n")

    lines.append("\n## Key reading\n")
    lines.append("To identify the separation axis, compare bare_hamiltonian to ed_full_likelihood at matched (L, γ):\n")
    lines.append("- If α is similar: α is NOT the separation axis (confirmed by pilot).\n")
    lines.append("- If T·ε prefactor differs by a constant factor > 2: prefactor IS a separation axis.\n")
    lines.append("- If tail convergence rate differs: convergence-at-small-ε IS a separation axis.\n")
    lines.append("- If precision floor differs: precision reach IS a separation axis.\n")
    lines.append("Feature whichever axis shows the cleanest and physically-interpretable gap.\n")

    return "\n".join(lines)


def main():
    bare = pd.read_csv(os.path.join(RESULTS_DIR, 'bare_hamiltonian.csv'))
    ps = pd.read_csv(os.path.join(RESULTS_DIR, 'ed_postselect.csv'))
    fl = pd.read_csv(os.path.join(RESULTS_DIR, 'ed_full_likelihood.csv'))

    print(f"Loaded: bare={len(bare)} rows, ed_postselect={len(ps)} rows, ed_full_likelihood={len(fl)} rows")

    for name, df in [('bare', bare), ('ed_postselect', ps), ('ed_full_likelihood', fl)]:
        print(f"  {name}: seeds={df['seed'].nunique()}, L={sorted(df['L'].unique())}, γ={sorted(df['gamma'].unique())}, "
              f"ε range=[{df['epsilon'].min():.1e}, {df['epsilon'].max():.1e}], "
              f"converged={df['converged'].mean()*100:.0f}%")

    summary = summarize(bare, ps, fl)

    csv_out = os.path.join(RESULTS_DIR, 'reanalysis_within_hamiltonian_summary.csv')
    summary.to_csv(csv_out, index=False, float_format='%.5g')
    print(f"\nSummary CSV: {csv_out}")

    md = render_markdown(summary)
    with open(REPORT_PATH, 'w') as f:
        f.write(md)
    print(f"Markdown report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
