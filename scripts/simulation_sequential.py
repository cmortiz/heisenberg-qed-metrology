#!/usr/bin/env python3
"""
Sequential signal-amplification protocol (Theorem 22 / Theorem 24).

Product-state probe: |0>^N (N = 2L+1, no entanglement). Per qubit, apply the
signal U(phi) = exp(-i phi X) M times, then measure in Z. Per-qubit phi_k =
phi + epsilon_k with epsilon_k ~ N(0, sigma_eps^2) quenched per qubit (per
seed). Theorem 24(b): majority vote over N independent qubits decides one bit
per round; total cost M_total = O(log(1/delta)/(epsilon * sqrt(N))).

Adaptive binary search (this implementation):
  - Maintain interval [lo, hi]; mid = (lo+hi)/2.
  - Round depth M_r = ceil(C_M / ((hi-lo) * sqrt(N))), with C_M absorbing
    the noiseless guard-band constant of Theorem 22.
  - Apply rotation U(M_r * (phi - mid)) to each qubit and shift by pi/(4 M_r)
    so the operating point lands at the threshold sin^2 = 1/2.
    Single-qubit P(|1>) = sin^2(pi/4 + M_r (phi + epsilon_k - mid)).
  - Sample N i.i.d. outcomes, take majority. If majority == 1 then phi > mid,
    else phi < mid. Halve the interval.
  - Iterate until (hi - lo) <= 2 * epsilon_target.
  - Total resource cost: sum_r N * M_r qubit-time units.

CSV output: scripts/results/sequential.csv
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
import argparse

OMEGA_TRUE_DEFAULT = 0.3
W0_DEFAULT = np.pi   # initial interval width (matches GHZ grid range)
# Operating-point depth: M_r = ceil(C_M / W) sends the bin-boundary rotation
# argument to pi/4 + C_M/2. C_M = pi/2 puts the boundary at max signal
# sin^2(pi/2) = 1 (no wrap-around) while keeping the threshold-center at the
# 50/50 operating point sin^2(pi/4) = 1/2.
C_M_DEFAULT = np.pi / 2
SHOTS_PER_ROUND_DEFAULT = 30  # K shots per round; majority over N*K outcomes per
                              # Theorem 24(b). K=30 gives ~95% noiseless convergence
                              # over R~14 rounds at small N (rounds where true omega
                              # lies near the bin midpoint dominate the failure rate).
MAX_ROUNDS = 200     # safety cap
CONVERGENCE_TOLERANCE = 1.2  # match other sims: |omega_hat - omega| < 1.2 * eps


def run_one_seed(seed, gamma_mean, sigma_epsilon, L, omega_true, epsilons,
                 c_m=C_M_DEFAULT, w0=W0_DEFAULT,
                 shots_per_round=SHOTS_PER_ROUND_DEFAULT,
                 eps_targets=None):
    """Run sequential binary search across all epsilon targets for one seed.

    epsilons: per-qubit calibration offsets (length N), drawn ONCE per seed
              (quenched device realization); known to estimator (analogous to
              the combined-sim convention).
    """
    if eps_targets is None:
        eps_targets = np.geomspace(1e-4, 1e-1, 60)
    N = 2 * L + 1
    rng = np.random.default_rng(seed)

    results = []
    for eps_target in eps_targets:
        # Per-target RNG (reproducible per (seed, eps_target))
        target_rng = np.random.default_rng(seed + int(eps_target * 1e8) % (2**31))

        # Wrap omega into the canonical interval [0, w0)
        # (Algorithm assumes omega lies in the initial interval.)
        omega_wrapped = omega_true % w0

        lo, hi = 0.0, w0
        total_resources = 0
        n_rounds = 0

        for r in range(MAX_ROUNDS):
            width = hi - lo
            if width <= 2.0 * eps_target:
                break

            mid = 0.5 * (lo + hi)
            # Depth: M_r = ceil(c_m / width). With c_m = pi/2 the bin boundary
            # rotation argument is pi/4 + c_m/2 = pi/2, max signal, no wrap.
            # Note this depth does NOT depend on N: the sqrt(N) speedup of
            # Theorem 22 enters via majority-vote concentration over N*K
            # outcomes (Theorem 24(b)), not via M_r directly.
            M_r = max(1, int(np.ceil(c_m / width)))

            # Per-qubit signal: rotation argument = M_r*(phi - mid + eps_k) + pi/4
            arg = M_r * (omega_wrapped - mid + epsilons) + 0.25 * np.pi
            p_one = np.sin(arg) ** 2  # per-qubit P(|1>)

            # K shots per round, N independent qubits per shot
            uniforms = target_rng.random((shots_per_round, N))
            outcomes_one = (uniforms < p_one[None, :]).astype(np.int32)
            n_one = int(np.sum(outcomes_one))
            n_total = shots_per_round * N

            # Decision: majority of N*K outcomes
            if n_one > n_total // 2:
                lo = mid
            else:
                hi = mid

            total_resources += shots_per_round * N * M_r
            n_rounds = r + 1

        # Final estimate: midpoint of final interval
        omega_hat = 0.5 * (lo + hi)
        # Circular distance on [0, w0)
        diff = abs(omega_hat - omega_wrapped)
        final_error = min(diff, w0 - diff)
        # Convergence criterion: same as other sims (final_error < 1.2 * eps_target).
        # Width-only convergence is misleading for binary search because per-round
        # majority-vote failures can put the true omega outside the final bin.
        converged = bool(final_error < CONVERGENCE_TOLERANCE * eps_target)

        results.append({
            'seed': seed,
            'gamma': gamma_mean,
            'sigma_epsilon': sigma_epsilon,
            'L': L,
            'N_code': N,
            'epsilon': eps_target,
            'mode': 'sequential',
            'total_resources': total_resources,
            'converged': converged,
            'final_error': final_error,
            'n_rounds': n_rounds,
            'M_max_used': max(1, int(np.ceil(c_m / (eps_target * 2.0 * np.sqrt(N))))),
            'timestamp': pd.Timestamp.now().isoformat(),
        })
    return pd.DataFrame(results)


def run_simulation(seed, gamma_mean, sigma_epsilon, L, c_m, w0,
                    shots_per_round, extended):
    omega_true = OMEGA_TRUE_DEFAULT
    N = 2 * L + 1

    seed_rng = np.random.default_rng(seed)
    if sigma_epsilon > 0:
        epsilons = seed_rng.normal(0.0, sigma_epsilon, N)
    else:
        epsilons = np.zeros(N)

    if extended:
        eps_targets = np.geomspace(1e-6, 1e-4, 40)
        range_str = "10^{-6} to 10^{-4} (40 points, extended)"
    else:
        eps_targets = np.geomspace(1e-4, 1e-1, 60)
        range_str = "10^{-4} to 10^{-1} (60 points, standard)"

    print(f"\n{'='*60}")
    print(f"Sequential Binary Search (Theorem 22 / 24)")
    print(f"{'='*60}")
    print(f"  Seed:              {seed}")
    print(f"  N = 2L+1:          {N} (L={L})")
    print(f"  gamma_mean:        {gamma_mean:.4f} (placeholder; not used in this sim)")
    print(f"  sigma_epsilon:     {sigma_epsilon:.4f}")
    print(f"  omega_true:        {omega_true}")
    print(f"  C_M (depth prefac):{c_m:.4f} (M_r = ceil(C_M/W))")
    print(f"  Shots per round K: {shots_per_round}")
    print(f"  W_0:               {w0:.4f}")
    print(f"  Epsilon:           {range_str}")
    if sigma_epsilon > 0:
        print(f"  epsilon_k range:   [{epsilons.min():.6f}, {epsilons.max():.6f}]")
    print(f"{'='*60}\n")

    start_time = time.time()
    df = run_one_seed(seed, gamma_mean, sigma_epsilon, L, omega_true,
                      epsilons, c_m=c_m, w0=w0,
                      shots_per_round=shots_per_round,
                      eps_targets=eps_targets)
    elapsed = time.time() - start_time
    n_converged = int(df['converged'].sum())
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Converged: {n_converged}/{len(df)} ({100*n_converged/len(df):.0f}%)")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Sequential binary search (Theorem 22 / 24): product-state, "
                    "M repeated signal applications, majority-vote per round."
    )
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("gamma", type=float, nargs='?', default=0.0,
                        help="Mean transverse field strength (recorded; "
                             "Theorem 24 noise model only uses sigma_epsilon)")
    parser.add_argument("sigma_epsilon", type=float, nargs='?', default=0.0,
                        help="Std dev of per-qubit longitudinal offsets "
                             "(quenched per seed)")
    parser.add_argument("--L", type=int, default=1,
                        help="Code parameter: N = 2L+1 qubits (default: 1)")
    parser.add_argument("--c-m", type=float, default=C_M_DEFAULT,
                        help=f"Depth prefactor M_r=ceil(C_M/W) (default: pi/2)")
    parser.add_argument("--shots-per-round", type=int, default=SHOTS_PER_ROUND_DEFAULT,
                        help=f"K shots per binary-search round (default: {SHOTS_PER_ROUND_DEFAULT})")
    parser.add_argument("--w0", type=float, default=W0_DEFAULT,
                        help="Initial search-interval width (default: pi)")
    parser.add_argument("--extended", action='store_true',
                        help="Extended epsilon range (10^-6 to 10^-4, 40 pts)")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    df = run_simulation(args.seed, args.gamma, args.sigma_epsilon,
                        args.L, args.c_m, args.w0,
                        args.shots_per_round, args.extended)

    output_file = os.path.join(output_dir, 'sequential.csv')
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
