"""Verification pilot: bare GHZ under Hamiltonian noise at high γ_Ham.

Tests whether α stays near 1 (Heisenberg) or shifts toward 2 (SQL) at γ_Ham ∈ {0.10, 0.30, 0.45}.
If α stays ≈ 1 at γ_Ham=0.45 (γ/ω = 1.5), the peer-review Option A cannot produce the bare→SQL
transition within-model, and we commit to Option B/D instead.
"""

import os
import sys
import numpy as np
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from simulation_bare_hamiltonian import simulate_for_epsilon, compute_grid_resolution


def run_pilot():
    omega_true = 0.3
    L = 2
    N_code = 2 * L + 1
    h = 0.0
    gammas = [0.10, 0.30, 0.45]
    seeds = [0, 1, 2]
    epsilons = np.geomspace(1e-3, 1e-1, 15)
    eps_min = float(epsilons.min())
    m = compute_grid_resolution(N_code, eps_min)

    print(f"Pilot: L={L} (N={N_code}), h={h}, ω={omega_true}")
    print(f"  γ_Ham ∈ {gammas}   (γ/ω ∈ {[g/omega_true for g in gammas]})")
    print(f"  seeds: {seeds}")
    print(f"  ε: {len(epsilons)} log-spaced in [{eps_min:.1e}, {epsilons.max():.1e}]")
    print(f"  grid: 2^{m} = {2**m} points")
    print()

    results = []
    t0 = time.time()
    for gamma in gammas:
        for seed in seeds:
            print(f"  running γ={gamma}, seed={seed}... ", end="", flush=True)
            t_cfg = time.time()
            for eps in epsilons:
                eps_rng = np.random.default_rng(seed + int(eps * 1e8) % (2**31))
                total_resources, converged, final_error, n_code, n_rounds = simulate_for_epsilon(
                    eps, omega_true, gamma, h, L, m, eps_rng
                )
                results.append({
                    'seed': seed,
                    'gamma': gamma,
                    'h': h,
                    'L': L,
                    'N_code': n_code,
                    'epsilon': eps,
                    'total_resources': total_resources,
                    'converged': converged,
                    'final_error': final_error,
                    'n_rounds': n_rounds,
                })
            print(f"done in {time.time()-t_cfg:.0f}s")

    print(f"\nTotal wall clock: {(time.time()-t0)/60:.1f} min")

    df = pd.DataFrame(results)
    out_path = os.path.join(script_dir, 'results', 'pilot_high_gamma.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    print("\n=== Scaling exponent α per (γ, seed) ===")
    for gamma in gammas:
        alphas = []
        for seed in seeds:
            sub = df[(df['gamma'] == gamma) & (df['seed'] == seed) & df['converged']]
            if len(sub) < 3:
                alphas.append(np.nan)
                continue
            log_eps = np.log(sub['epsilon'].values)
            log_T = np.log(sub['total_resources'].values)
            slope = np.polyfit(log_eps, log_T, 1)[0]
            alphas.append(-slope)
        mean_a = np.nanmean(alphas)
        std_a = np.nanstd(alphas)
        conv_rate = df[(df['gamma'] == gamma) & df['converged']].shape[0] / (len(seeds) * len(epsilons))
        print(f"  γ_Ham = {gamma:.2f} (γ/ω = {gamma/omega_true:.2f}): "
              f"α = {mean_a:.3f} ± {std_a:.3f} (n={len(alphas)} seeds), "
              f"convergence = {conv_rate*100:.0f}%")


if __name__ == "__main__":
    run_pilot()
