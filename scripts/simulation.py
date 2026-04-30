#!/usr/bin/env python3
"""
Bayesian phase estimation with GHZ states under depolarizing noise.

GHZ: P(+1|omega,n) = (1 + V_n cos(2n*omega))/2, V_n = prod(1-gamma_k).
Product: P_k(+1|omega) = (1 + (1-gamma_k)cos(2*omega))/2.
Noise: gamma_k ~ N(gamma, (gamma*h)^2). Convergence: |omega_hat-omega|_circ < 1.2*eps (Theorem 7).
Multi-frequency n resolves ambiguity; no random theta needed (Remark 6).
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

M = 16                    # 2^M grid points
MAX_N = 16384             # 2^(M-2): Nyquist limit for cos(2n*omega) on 2^M grid
MIN_N = 10
MAX_EXPERIMENTS = 10000
CHECK_FREQUENCY = 100
CONVERGENCE_TOLERANCE = 1.2


@jit
def ghz_log_visibility(gammas, n):
    """log(V_n) = sum log(1 - gamma_k) for k < n."""
    mask = jnp.arange(gammas.shape[0]) < n
    log_terms = jnp.where(mask, jnp.log(jnp.maximum(1.0 - gammas, 1e-30)), 0.0)
    return jnp.sum(log_terms)


@jit
def ghz_simulate_outcome(gammas, n, omega, uniform):
    """Sample GHZ parity: P(+1) = (1 + V_n cos(2n*omega))/2."""
    log_vn = ghz_log_visibility(gammas, n)
    vn = jnp.exp(log_vn)
    cos_2n = jnp.cos(2.0 * n * omega)
    p_plus = (1.0 + vn * cos_2n) / 2.0
    return jnp.where(uniform < p_plus, 0, 1)


@jit
def ghz_bayesian_update(logpk, j_grid, gammas, n, outcome, grid_size):
    """Bayesian update for one GHZ measurement using integer-mod phase indexing."""
    log_vn = ghz_log_visibility(gammas, n)
    vn = jnp.exp(log_vn)
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    # Bitwise AND = exact mod for power-of-2 grid, safe even if 2*n*j overflows int32
    phase_idx = (2 * n * j_grid) & (grid_size - 1)
    cos_2n = jnp.cos(2.0 * jnp.pi * phase_idx / grid_size)
    likelihood = (1.0 + sign * vn * cos_2n) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def product_simulate_outcomes(gammas, n, omega, uniforms):
    """Sample n independent qubits. Padded to MAX_N to avoid XLA recompilation."""
    mask = jnp.arange(gammas.shape[0]) < n
    vis = (1.0 - gammas) * mask
    p_plus = (1.0 + vis * jnp.cos(2.0 * omega)) / 2.0
    outcomes = jnp.where(uniforms < p_plus, 0, 1)
    return outcomes


@jit
def product_bayesian_update_homogeneous(logpk, xs, v, n_plus, n_minus):
    """Homogeneous update using sufficient statistic (n_plus, n_minus). O(grid)."""
    cos_2w = jnp.cos(2.0 * xs)
    log_lik_plus = jnp.log(jnp.maximum((1.0 + v * cos_2w) / 2.0, 1e-30))
    log_lik_minus = jnp.log(jnp.maximum((1.0 - v * cos_2w) / 2.0, 1e-30))
    logpk = logpk + n_plus * log_lik_plus + n_minus * log_lik_minus
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def product_bayesian_update_heterogeneous(logpk, xs, gammas, n, outcomes):
    """Heterogeneous update via fori_loop over qubits. O(grid) memory."""
    cos_2w = jnp.cos(2.0 * xs)

    def body_fn(k, acc):
        v_k = 1.0 - gammas[k]
        s_k = jnp.where(outcomes[k] == 0, 1.0, -1.0)
        log_lik_k = jnp.log(jnp.maximum((1.0 + s_k * v_k * cos_2w) / 2.0, 1e-30))
        return acc + log_lik_k

    total_log_lik = jax.lax.fori_loop(0, n, body_fn, jnp.zeros_like(logpk))
    logpk = logpk + total_log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def circular_distance(phi, theta):
    """Circular phase distance on [0, 2*pi) (Theorem 7)."""
    diff = jnp.abs(phi - theta)
    return jnp.minimum(diff, 2.0 * jnp.pi - diff)


@jit
def compute_estimate_and_error(logpk, xs, phi):
    """MAP estimate and circular phase error."""
    estimate = xs[jnp.argmax(logpk)]
    error = circular_distance(phi, estimate)
    return estimate, error


def simulate_for_epsilon(epsilon, phi, gamma, h, mode, m, rng):
    """Run Bayesian estimation for a single epsilon. Returns (resources, converged, error, N)."""
    N = int(1.0 / epsilon)
    N = max(MIN_N, min(N, MAX_N))

    grid_size = 2 ** m
    xs = jnp.linspace(0, 2 * jnp.pi, grid_size, endpoint=False, dtype=jnp.float32)
    j_grid = jnp.arange(grid_size, dtype=jnp.int32)

    logpk = jnp.full(grid_size, -m * jnp.log(2.0), dtype=jnp.float32)

    # Padded to MAX_N for fixed array shapes (avoids XLA recompilation)
    is_homogeneous = (h == 0)
    if h > 0:
        gammas_np = np.full(MAX_N, gamma, dtype=np.float32)
        gammas_np[:N] = np.clip(rng.normal(gamma, gamma * h, N), 0.0, 0.99)
    else:
        gammas_np = np.full(MAX_N, gamma, dtype=np.float32)

    gammas = jnp.array(gammas_np, dtype=jnp.float32)
    v_homogeneous = jnp.float32(1.0 - gamma)

    # GHZ: one uniform/experiment. Product: JAX PRNG per experiment (avoids ~640 MB pre-alloc).
    if mode == 'ghz':
        ns = rng.integers(1, N + 1, MAX_EXPERIMENTS)
        uniforms = jnp.array(rng.random(MAX_EXPERIMENTS), dtype=jnp.float32)
    else:
        ns = np.full(MAX_EXPERIMENTS, N, dtype=np.int32)
        jax_key = jax.random.PRNGKey(rng.integers(0, 2**31))

    total_resources = 0
    best_error = float('inf')
    phi_jax = jnp.float32(phi)
    n_jax = jnp.int32(N)

    for i in range(MAX_EXPERIMENTS):
        if mode == 'ghz':
            n = int(ns[i])
            outcome = int(ghz_simulate_outcome(gammas, n, phi_jax, uniforms[i]))
            logpk = ghz_bayesian_update(logpk, j_grid, gammas, n, outcome, grid_size)
            total_resources += n
        else:
            jax_key, subkey = jax.random.split(jax_key)
            exp_uniforms = jax.random.uniform(subkey, shape=(MAX_N,), dtype=jnp.float32)
            outcomes = product_simulate_outcomes(gammas, n_jax, phi_jax, exp_uniforms)
            if is_homogeneous:
                n_plus = jnp.sum(jnp.where(
                    (jnp.arange(MAX_N) < n_jax) & (outcomes == 0), 1.0, 0.0
                ))
                n_minus = jnp.float32(N) - n_plus
                logpk = product_bayesian_update_homogeneous(
                    logpk, xs, v_homogeneous, n_plus, n_minus
                )
            else:
                logpk = product_bayesian_update_heterogeneous(
                    logpk, xs, gammas, n_jax, outcomes
                )
            total_resources += N

        if i % CHECK_FREQUENCY == 0 or i == MAX_EXPERIMENTS - 1:
            _, error = compute_estimate_and_error(logpk, xs, phi_jax)
            error_val = float(error)
            if error_val < best_error:
                best_error = error_val
            if error_val < epsilon * CONVERGENCE_TOLERANCE:
                return total_resources, True, error_val, N

    return total_resources, False, best_error, N


def run_simulation(seed, gamma, h, mode, extended):
    """Run simulation across all epsilon values for one seed."""
    phi = 0.3
    m = M

    if extended:
        epsilons = np.geomspace(10**(-4.5), 1e-1, 70)
        range_str = "10^{-4.5} to 10^{-1} (70 points, extended)"
    else:
        epsilons = np.geomspace(1e-4, 1e-1, 60)
        range_str = "10^{-4} to 10^{-1} (60 points, standard)"

    print(f"\n{'='*60}")
    print(f"Bayesian Phase Estimation Simulation")
    print(f"{'='*60}")
    print(f"  Seed:       {seed}")
    print(f"  Mode:       {mode}")
    print(f"  gamma:      {gamma:.4f}")
    print(f"  h:          {h:.4f}")
    print(f"  phi:        {phi}")
    print(f"  m:          {m} ({2**m} grid points)")
    print(f"  Epsilon:    {range_str}")
    print(f"  MAX_N:      {MAX_N}")
    print(f"  Max expts:  {MAX_EXPERIMENTS}")
    if USING_GPU:
        print(f"  GPU:        yes")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for eps in tqdm(epsilons, desc=f"  seed={seed} gamma={gamma} mode={mode}"):
        eps_rng = np.random.default_rng(seed + int(eps * 1e8) % (2**31))

        total_resources, converged, final_error, N = simulate_for_epsilon(
            eps, phi, gamma, h, mode, m, eps_rng
        )

        results.append({
            'seed': seed,
            'gamma': gamma,
            'h': h,
            'epsilon': eps,
            'mode': mode,
            'N': N,
            'total_resources': total_resources,
            'converged': converged,
            'final_error': final_error,
            'timestamp': pd.Timestamp.now().isoformat(),
        })

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")

    n_converged = sum(1 for r in results if r['converged'])
    print(f"Converged: {n_converged}/{len(results)} ({100*n_converged/len(results):.0f}%)")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Bayesian phase estimation simulation")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("gamma", type=float, nargs='?', default=0.0,
                        help="Mean depolarizing error rate (default: 0)")
    parser.add_argument("h", type=float, nargs='?', default=0.0,
                        help="Noise heterogeneity (relative std dev, default: 0)")
    parser.add_argument("--mode", choices=['ghz', 'product'], default='ghz',
                        help="Measurement mode (default: ghz)")
    parser.add_argument("--extended", action='store_true',
                        help="Use extended epsilon range (10^{-4.5} to 10^{-1})")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df = run_simulation(args.seed, args.gamma, args.h, args.mode, args.extended)

    output_file = os.path.join(output_dir, 'bare_ghz.csv')
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
