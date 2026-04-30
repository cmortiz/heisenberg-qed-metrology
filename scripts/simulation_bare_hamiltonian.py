#!/usr/bin/env python3
"""
Bayesian phase estimation with bare GHZ state under Hamiltonian noise.

Protocol: prepare |+>_L = (|0^N> + |1^N>)/sqrt(2), apply U = prod_k exp(-i M (omega Z_k + gamma_k X_k)),
measure X^{otimes N} (no syndrome, no post-selection), Bayesian update.

Observable (derived from Heisenberg-picture rotation of X_k around axis (gamma_k, 0, omega)/Omega_k
by angle 2 M Omega_k, with Omega_k = sqrt(omega^2 + gamma_k^2)):

    <X^{otimes N}>_GHZ = Re prod_k (alpha_k - i beta_k)    (odd N)

where
    alpha_k = 1 - 2 (omega/Omega_k)^2 sin^2(M Omega_k)
    beta_k  = (omega/Omega_k) sin(2 M Omega_k)

For even N, an additional Re prod_k chi_k term appears with chi_k = 2 (gamma_k omega / Omega_k^2) sin^2(M Omega_k),
but this script targets odd N = 2L+1 (L in {1,2,3}) matching the paper's bit-flip code parameters.

Likelihood: P(+1 | omega, M) = (1 + <X^N>_GHZ) / 2. Reduces to cos(2 N M omega) in the noiseless limit gamma_k -> 0.
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
CONVERGENCE_TOLERANCE = 1.2
MAX_CACHE_MB = 2048


def compute_grid_resolution(N_code, eps_min):
    """Grid exponent m s.t. N_code * M_max fits in 2^(m-2) (Nyquist safety)."""
    import math
    M_max_needed = max(1, int(1.0 / eps_min))
    return max(16, math.ceil(math.log2(4 * N_code * M_max_needed)))


@jit
def ghz_x_parity_grid(omega_grid, gammas, M_evol):
    """<X^N>_GHZ as a function of omega over a grid, for odd N.

    Returns an array of shape (grid_size,) giving Re prod_k (alpha_k - i beta_k).
    """
    # Shape: (grid_size, N)
    Omega_k = jnp.sqrt(omega_grid[:, None]**2 + gammas[None, :]**2 + 1e-30)
    MO = M_evol * Omega_k
    ratio = omega_grid[:, None] / Omega_k
    s = jnp.sin(MO)
    alpha = 1.0 - 2.0 * ratio**2 * s**2
    beta = 2.0 * ratio * s * jnp.cos(MO)  # sin(2 MO) * ratio / 1 = 2 sin(MO) cos(MO) ratio
    # Complex product over qubits: prod_k (alpha_k - i beta_k)
    re = alpha
    im = -beta
    # Accumulate product via log is numerically unstable (magnitudes near 1); do direct product.
    # For N up to a few tens, direct complex product is fine.
    def cplx_mul(carry, pair):
        cr, ci = carry
        pr, pi = pair
        return (cr * pr - ci * pi, cr * pi + ci * pr)
    init = (jnp.ones_like(omega_grid), jnp.zeros_like(omega_grid))
    # scan over the qubit axis
    pairs = (re.T, im.T)  # each shape (N, grid_size)
    final, _ = jax.lax.scan(
        lambda carry, p: (cplx_mul(carry, p), None),
        init,
        pairs,
    )
    return final[0]  # Re part


@jit
def ghz_x_parity_true(omega, gammas, M_evol):
    """Scalar <X^N>_GHZ at the true omega (for sampling outcomes)."""
    Omega_k = jnp.sqrt(omega**2 + gammas**2 + 1e-30)
    MO = M_evol * Omega_k
    ratio = omega / Omega_k
    s = jnp.sin(MO)
    alpha = 1.0 - 2.0 * ratio**2 * s**2
    beta = 2.0 * ratio * s * jnp.cos(MO)
    re = alpha
    im = -beta
    # Product over qubits
    def cplx_mul(carry, pair):
        cr, ci = carry
        pr, pi = pair
        return (cr * pr - ci * pi, cr * pi + ci * pr)
    init = (jnp.float32(1.0), jnp.float32(0.0))
    final, _ = jax.lax.scan(
        lambda carry, p: (cplx_mul(carry, p), None),
        init,
        (re, im),
    )
    return final[0]


@jit
def bayesian_update(logpk, x_parity_grid, outcome):
    """Log-space Bayesian update. P(+1 | omega) = (1 + <X^N>)/2."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * x_parity_grid) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def circular_distance(phi, theta):
    diff = jnp.abs(phi - theta)
    return jnp.minimum(diff, 2.0 * jnp.pi - diff)


@jit
def compute_estimate_and_error(logpk, xs, phi):
    estimate = xs[jnp.argmax(logpk)]
    error = circular_distance(phi, estimate)
    return estimate, error


def simulate_for_epsilon(epsilon, omega_true, gamma_mean, h, L, m, rng):
    """Run bare-GHZ Hamiltonian Bayesian estimation for a single epsilon target."""
    N_code = 2 * L + 1
    assert N_code % 2 == 1, "This script assumes odd N = 2L+1"

    M_max = max(1, int(1.0 / epsilon))
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

    parity_cache = {}
    max_cache_entries = (MAX_CACHE_MB * 1024 * 1024) // (grid_size * 4)

    total_physical_resources = 0
    omega_jax = jnp.float32(omega_true)

    for i in range(MAX_EXPERIMENTS):
        M_evol = int(M_evols[i])
        total_physical_resources += N_code * M_evol

        # Sample outcome from true distribution.
        x_true = float(ghz_x_parity_true(omega_jax, gammas_jax, jnp.float32(M_evol)))
        p_plus = (1.0 + x_true) / 2.0
        outcome = 0 if uniforms_outcome[i] < p_plus else 1

        # Bayesian update.
        if M_evol not in parity_cache:
            if len(parity_cache) >= max_cache_entries:
                parity_cache.clear()
            parity_cache[M_evol] = ghz_x_parity_grid(
                omega_grid_jax, gammas_jax, jnp.float32(M_evol)
            )
        parity_grid = parity_cache[M_evol]

        logpk = bayesian_update(logpk, parity_grid, outcome)

        if (i + 1) % CHECK_FREQUENCY == 0 or i == MAX_EXPERIMENTS - 1:
            _, error = compute_estimate_and_error(logpk, xs, omega_jax)
            error_val = float(error)
            if error_val < epsilon * CONVERGENCE_TOLERANCE:
                return total_physical_resources, True, error_val, N_code, i + 1

    _, error = compute_estimate_and_error(logpk, xs, omega_jax)
    return total_physical_resources, False, float(error), N_code, MAX_EXPERIMENTS


def run_simulation(seed, gamma_mean, h, L, extended):
    """Run bare-GHZ Hamiltonian estimation across all epsilon values."""
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

    print(f"\n{'='*60}")
    print(f"Bare-GHZ Bayesian Phase Estimation (Hamiltonian Noise)")
    print(f"{'='*60}")
    print(f"  Seed:              {seed}")
    print(f"  N = 2L+1:          {N_code} (L={L})")
    print(f"  gamma_mean:        {gamma_mean:.4f}")
    print(f"  h:                 {h:.4f}")
    print(f"  omega_true:        {omega_true}")
    print(f"  Grid:              2^{m} = {2**m} points")
    print(f"  Epsilon:           {range_str}")
    print(f"  Max experiments:   {MAX_EXPERIMENTS}")
    if USING_GPU:
        print(f"  GPU:               yes")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for eps in tqdm(epsilons, desc=f"  seed={seed} L={L} gamma={gamma_mean}"):
        eps_rng = np.random.default_rng(seed + int(eps * 1e8) % (2**31))

        total_resources, converged, final_error, n_code, n_rounds = simulate_for_epsilon(
            eps, omega_true, gamma_mean, h, L, m, eps_rng
        )

        results.append({
            'seed': seed,
            'gamma': gamma_mean,
            'h': h,
            'L': L,
            'N_code': n_code,
            'epsilon': eps,
            'mode': 'bare_hamiltonian',
            'total_resources': total_resources,
            'converged': converged,
            'final_error': final_error,
            'n_rounds': n_rounds,
            'timestamp': pd.Timestamp.now().isoformat(),
        })

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    n_converged = sum(1 for r in results if r['converged'])
    print(f"Converged: {n_converged}/{len(results)} ({100*n_converged/len(results):.0f}%)")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Bare-GHZ Bayesian phase estimation with Hamiltonian noise"
    )
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("gamma", type=float, help="Mean transverse field strength")
    parser.add_argument("h", type=float, nargs='?', default=0.0,
                        help="Noise heterogeneity (relative std dev, default: 0)")
    parser.add_argument("--L", type=int, default=1,
                        help="Code parameter: N = 2L+1 qubits (default: 1 = 3 qubits)")
    parser.add_argument("--extended", action='store_true',
                        help="Extended epsilon range (10^-6 to 10^-4)")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df = run_simulation(args.seed, args.gamma, args.h, args.L, args.extended)

    output_file = os.path.join(output_dir, 'bare_hamiltonian.csv')
    import fcntl
    with open(output_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            needs_header = f.tell() == 0
            df.to_csv(f, header=needs_header, index=False, float_format='%.10e')
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
