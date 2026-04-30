#!/usr/bin/env python3
"""
Bayesian phase estimation with bit-flip repetition code error detection.

Algorithm 2: encode |+⟩ into N=2L+1 qubits, apply U_k^M, measure stabilizers,
post-select on trivial syndrome (d=0), Bayesian update.

Theorem 9: phi_k = arctan(tan(M*Omega_k)*omega/Omega_k), p_k = sin^2(M*Omega_k)*gamma_k^2/Omega_k^2.
Post-selected phi_eff = sum_k phi_k with visibility 1 (cost: acceptance rate prod_k(1-p_k)).

Full-likelihood mode (--full-likelihood, h=0 only): uses ALL rounds with
N_eff = N_code - d (Theorem 9: errored qubits contribute zero phase).
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
import time
from tqdm import tqdm
import os
import argparse

jax.config.update("jax_enable_x64", False)

try:
    jax.devices('gpu')
    USING_GPU = True
    print("GPU detected. Using GPU acceleration.")
except Exception:
    USING_GPU = False
    print("No GPU detected. Using CPU with JAX optimization.")

MIN_M_EVOL = 1
MAX_EXPERIMENTS = 50000
CHECK_FREQUENCY = 100
CHECK_FREQUENCY_FL = 10       # Lower frequency avoids floor artifact
CONVERGENCE_TOLERANCE = 1.2
MAX_PHI_EFF_CACHE_MB = 2048


def compute_grid_resolution(N_effective, eps_min):
    """Grid exponent m s.t. N_effective * M_max < 2^(m-2) (Nyquist)."""
    import math
    M_max_needed = max(1, int(1.0 / eps_min))
    return max(16, math.ceil(math.log2(4 * N_effective * M_max_needed)))


def compute_hamiltonian_params_np(omega, gamma_k, M_evol):
    """Compute per-qubit error prob p_k and phase phi_k (Theorem 9, numpy)."""
    gamma_k = np.asarray(gamma_k, dtype=np.float64)
    Omega_k = np.sqrt(omega**2 + gamma_k**2)
    M_Omega_k = M_evol * Omega_k
    sin2 = np.sin(M_Omega_k)**2
    denom = omega**2 + gamma_k**2 + 1e-30
    ratio2 = omega**2 / denom
    p_k = sin2 * (1.0 - ratio2)
    # sin/cos form avoids tan overflow near poles
    c = np.cos(M_Omega_k)
    s = np.sin(M_Omega_k)
    sign_c = np.where(c >= 0, 1.0, -1.0)
    phi_k = np.arctan2(sign_c * s * omega, sign_c * Omega_k * c)
    return p_k, phi_k


@jit
def bayesian_update(logpk, phi_eff_grid, outcome):
    """Log-space Bayesian update: P(+1|omega) = (1 + cos(2*phi_eff))/2 (Remark 6)."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    cos_val = jnp.cos(2.0 * phi_eff_grid)
    likelihood = (1.0 + sign * cos_val) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def precompute_phi_eff_grid_jax(omega_grid, gammas, M_evol):
    """Compute phi_eff = sum_k phi_k over grid (d=0 post-selection)."""
    Omega_k = jnp.sqrt(omega_grid[:, None]**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    # sin/cos form avoids tan overflow near poles
    c = jnp.cos(M_Omega_k)
    s = jnp.sin(M_Omega_k)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    phi_k = jnp.arctan2(sign_c * s * omega_grid[:, None], sign_c * Omega_k * c)
    return jnp.sum(phi_k, axis=1)


@jit
def precompute_phi_single_grid_jax(omega_grid, gamma, M_evol):
    """Single-qubit phi over grid (homogeneous, for full-likelihood N_eff scaling)."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    c = jnp.cos(M_Omega)
    s = jnp.sin(M_Omega)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    return jnp.arctan2(sign_c * s * omega_grid, sign_c * Omega * c)


@jit
def circular_distance(phi, theta):
    """Circular phase distance on [0, 2*pi)."""
    diff = jnp.abs(phi - theta)
    return jnp.minimum(diff, 2.0 * jnp.pi - diff)


@jit
def compute_estimate_and_error(logpk, xs, phi):
    """MAP estimate and circular error."""
    estimate = xs[jnp.argmax(logpk)]
    error = circular_distance(phi, estimate)
    return estimate, error


def simulate_for_epsilon(epsilon, omega_true, gamma_mean, h, L, m, rng,
                         full_likelihood=False, save_rounds_file=None):
    """Run error-detected Bayesian estimation for a single epsilon target."""
    if full_likelihood:
        assert h == 0, "Full-likelihood mode requires homogeneous noise (h=0)"

    N_code = 2 * L + 1

    M_max = max(1, int(1.0 / epsilon))
    # Nyquist safety: N_code * M must fit in grid
    nyquist_limit = (2 ** (m - 2)) // N_code
    M_max = min(M_max, max(1, nyquist_limit))

    if h > 0:
        gammas_np = rng.normal(gamma_mean, gamma_mean * h, N_code)
        gammas_np = np.clip(gammas_np, 0.0, 10.0)
    else:
        gammas_np = np.full(N_code, gamma_mean)

    grid_size = 2 ** m
    xs_np = np.linspace(0, 2 * np.pi, grid_size, endpoint=False, dtype=np.float64)
    xs = jnp.array(xs_np, dtype=jnp.float32)
    omega_grid_jax = jnp.array(xs_np, dtype=jnp.float32)
    gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)

    logpk = jnp.full(grid_size, -m * jnp.log(2.0), dtype=jnp.float32)

    M_evols = rng.integers(MIN_M_EVOL, M_max + 1, MAX_EXPERIMENTS)
    uniforms_outcome = rng.random(MAX_EXPERIMENTS)
    uniforms_syndrome = rng.random((MAX_EXPERIMENTS, N_code))

    # Batch Hamiltonian params for all M_evol values
    M_evols_col = M_evols[:, None].astype(np.float64)
    gammas_row = gammas_np[None, :].astype(np.float64)
    Omega_k_all = np.sqrt(omega_true**2 + gammas_row**2)
    M_Omega_k_all = M_evols_col * Omega_k_all
    sin2_all = np.sin(M_Omega_k_all)**2
    denom_all = omega_true**2 + gammas_row**2 + 1e-30
    ratio2_all = omega_true**2 / denom_all
    p_k_all = sin2_all * (1.0 - ratio2_all)
    # sin/cos form avoids tan overflow near poles
    c_all = np.cos(M_Omega_k_all)
    s_all = np.sin(M_Omega_k_all)
    sign_c_all = np.where(c_all >= 0, 1.0, -1.0)
    phi_k_all = np.arctan2(
        sign_c_all * s_all * omega_true, sign_c_all * Omega_k_all * c_all
    )
    phi_eff_true_all = np.sum(phi_k_all, axis=1)

    phi_eff_cache = {}
    phi_single_cache = {}
    max_cache_entries = (MAX_PHI_EFF_CACHE_MB * 1024 * 1024) // (grid_size * 4)

    total_physical_resources = 0
    n_accepted = 0
    n_rejected = 0
    n_updates = 0
    n_syndrome_used = 0
    sum_d = 0
    omega_jax = jnp.float32(omega_true)

    for i in range(MAX_EXPERIMENTS):
        M_evol = int(M_evols[i])

        p_k = p_k_all[i]
        errors_k = uniforms_syndrome[i, :N_code] < p_k
        d = int(np.sum(errors_k))

        total_physical_resources += N_code * M_evol

        if full_likelihood:
            n_updates += 1
            N_eff = N_code - d

            if d > 0:
                n_syndrome_used += 1
            sum_d += d

            phi_single_true = float(phi_k_all[i, 0])
            phi_eff_true = N_eff * phi_single_true

            p_plus = (1.0 + np.cos(2.0 * phi_eff_true)) / 2.0
            outcome = 0 if uniforms_outcome[i] < p_plus else 1

            if save_rounds_file is not None:
                save_rounds_file.write(
                    f"{epsilon:.10e},{i},{M_evol},{d},{N_eff},{outcome}\n"
                )

            if M_evol not in phi_single_cache:
                if len(phi_single_cache) >= max_cache_entries:
                    phi_single_cache.clear()
                phi_single_cache[M_evol] = precompute_phi_single_grid_jax(
                    omega_grid_jax, gammas_jax[0], jnp.float32(M_evol)
                )
            phi_eff_grid = N_eff * phi_single_cache[M_evol]

            logpk = bayesian_update(logpk, phi_eff_grid, outcome)

            if n_updates % CHECK_FREQUENCY_FL == 0 or i == MAX_EXPERIMENTS - 1:
                _, error = compute_estimate_and_error(logpk, xs, omega_jax)
                error_val = float(error)
                if error_val < epsilon * CONVERGENCE_TOLERANCE:
                    mean_d = sum_d / n_updates if n_updates > 0 else 0.0
                    return (total_physical_resources, True, error_val, N_code,
                            n_updates, 0, n_syndrome_used, mean_d)
        else:
            if d > 0:
                n_rejected += 1
                continue

            n_accepted += 1
            phi_eff_true = float(phi_eff_true_all[i])
            p_plus = (1.0 + np.cos(2.0 * phi_eff_true)) / 2.0
            outcome = 0 if uniforms_outcome[i] < p_plus else 1

            if M_evol not in phi_eff_cache:
                if len(phi_eff_cache) >= max_cache_entries:
                    phi_eff_cache.clear()
                phi_eff_cache[M_evol] = precompute_phi_eff_grid_jax(
                    omega_grid_jax, gammas_jax, jnp.float32(M_evol)
                )
            phi_eff_grid = phi_eff_cache[M_evol]

            logpk = bayesian_update(logpk, phi_eff_grid, outcome)

            if n_accepted % CHECK_FREQUENCY == 0 or i == MAX_EXPERIMENTS - 1:
                _, error = compute_estimate_and_error(logpk, xs, omega_jax)
                error_val = float(error)
                if error_val < epsilon * CONVERGENCE_TOLERANCE:
                    return (total_physical_resources, True, error_val, N_code,
                            n_accepted, n_rejected, 0, 0.0)

    _, error = compute_estimate_and_error(logpk, xs, omega_jax)
    if full_likelihood:
        mean_d = sum_d / n_updates if n_updates > 0 else 0.0
        return (total_physical_resources, False, float(error), N_code,
                n_updates, 0, n_syndrome_used, mean_d)
    return (total_physical_resources, False, float(error), N_code,
            n_accepted, n_rejected, 0, 0.0)


def run_simulation(seed, gamma_mean, h, L, extended, full_likelihood=False,
                    save_rounds=False):
    """Run error-detected estimation across all epsilon values."""
    omega_true = 0.3
    N_code = 2 * L + 1
    eps_min = 1e-6 if extended else 1e-4
    m = compute_grid_resolution(N_code, eps_min)

    if extended:
        epsilons = np.geomspace(1e-6, 1e-4, 40)
        range_str = "10^{-6} to 10^{-4} (40 points, extended)"
    else:
        epsilons = np.geomspace(1e-4, 1e-1, 60)
        range_str = "10^{-4} to 10^{-1} (60 points, standard)"

    p_k_test, _ = compute_hamiltonian_params_np(omega_true, np.full(N_code, gamma_mean), 1)
    accept_prob = float(np.prod(1.0 - p_k_test))

    mode_str = "Full-Likelihood" if full_likelihood else "Post-Selection"
    print(f"\n{'='*60}")
    print(f"Error-Detected Bayesian Phase Estimation ({mode_str})")
    print(f"{'='*60}")
    print(f"  Seed:              {seed}")
    print(f"  Code:              [[{N_code}, 1]] repetition code (L={L})")
    print(f"  gamma_mean:        {gamma_mean:.4f}")
    print(f"  h:                 {h:.4f}")
    print(f"  omega_true:        {omega_true}")
    print(f"  Grid:              2^{m} = {2**m} points")
    print(f"  Epsilon:           {range_str}")
    print(f"  Max experiments:   {MAX_EXPERIMENTS}")
    print(f"  Accept prob (M=1): {accept_prob:.4f}")
    print(f"  Mode:              {mode_str}")
    if USING_GPU:
        print(f"  GPU:               yes")
    print(f"{'='*60}\n")

    rounds_dir = None
    if save_rounds:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rounds_dir = os.path.join(script_dir, 'results', 'rounds')
        os.makedirs(rounds_dir, exist_ok=True)

    results = []
    start_time = time.time()

    for eps in tqdm(epsilons, desc=f"  seed={seed} L={L} gamma={gamma_mean}"):
        eps_rng = np.random.default_rng(seed + int(eps * 1e8) % (2**31))

        rounds_file = None
        if rounds_dir is not None:
            rounds_path = os.path.join(
                rounds_dir,
                f"ed_rounds_seed{seed}_L{L}_gamma{gamma_mean}.csv"
            )
            is_new = not os.path.exists(rounds_path)
            rounds_file = open(rounds_path, 'a')
            if is_new:
                rounds_file.write("epsilon,round,M_evol,d,N_eff,outcome\n")

        try:
            (total_resources, converged, final_error, n_code,
             n_accepted, n_rejected, n_synd, mean_d) = simulate_for_epsilon(
                eps, omega_true, gamma_mean, h, L, m, eps_rng,
                full_likelihood=full_likelihood,
                save_rounds_file=rounds_file
            )
        finally:
            if rounds_file is not None:
                rounds_file.close()

        mode_label = 'full_likelihood' if full_likelihood else 'error_detected'
        results.append({
            'seed': seed,
            'gamma': gamma_mean,
            'h': h,
            'L': L,
            'N_code': n_code,
            'epsilon': eps,
            'mode': mode_label,
            'total_resources': total_resources,
            'converged': converged,
            'final_error': final_error,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'acceptance_rate': n_accepted / max(1, n_accepted + n_rejected),
            'n_syndrome_used': n_synd,
            'mean_d': mean_d,
            'timestamp': pd.Timestamp.now().isoformat(),
        })

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    n_converged = sum(1 for r in results if r['converged'])
    print(f"Converged: {n_converged}/{len(results)} ({100*n_converged/len(results):.0f}%)")

    if not full_likelihood:
        avg_accept = np.mean([r['acceptance_rate'] for r in results])
        print(f"Mean acceptance rate: {avg_accept:.3f}")
    else:
        avg_synd = np.mean([r['n_syndrome_used'] for r in results])
        avg_d = np.mean([r['mean_d'] for r in results])
        print(f"Mean syndrome-used rounds: {avg_synd:.0f}")
        print(f"Mean d per round: {avg_d:.3f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Error-detected Bayesian phase estimation with bit-flip repetition code"
    )
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("gamma", type=float, help="Mean transverse field strength")
    parser.add_argument("h", type=float, nargs='?', default=0.0,
                        help="Noise heterogeneity (relative std dev, default: 0)")
    parser.add_argument("--L", type=int, default=1,
                        help="Code parameter: N = 2L+1 qubits (default: 1, i.e., 3-qubit code)")
    parser.add_argument("--extended", action='store_true',
                        help="Extended epsilon range (10^-6 to 10^-4, 40 pts). "
                             "Run after default to add deeper data points.")
    parser.add_argument("--full-likelihood", action='store_true',
                        help="Use full-likelihood inference (all rounds, syndrome-adjusted)")
    parser.add_argument("--save-rounds", action='store_true',
                        help="Save per-round data to results/rounds/")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df = run_simulation(args.seed, args.gamma, args.h, args.L, args.extended,
                        full_likelihood=args.full_likelihood,
                        save_rounds=args.save_rounds)

    if args.full_likelihood:
        output_file = os.path.join(output_dir, 'ed_full_likelihood.csv')
    else:
        output_file = os.path.join(output_dir, 'ed_postselect.csv')
    import fcntl
    with open(output_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, 2)  # seek to end
            needs_header = f.tell() == 0
            df.to_csv(f, header=needs_header, index=False, float_format='%.10e')
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
