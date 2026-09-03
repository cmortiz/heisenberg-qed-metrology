"""Engine configuration and run specification (shared contract; no jax import)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

OMEGA_TRUE = 0.3
MAX_N_ADAPTIVE = 16384          # Nyquist limit of the 2^16 grid (legacy simulation.py:43)
MIN_N_ADAPTIVE = 10             # legacy simulation.py:138


@dataclass(frozen=True)
class EngineConfig:
    """Per-protocol loop constants (legacy anchors in the plan, 'Engine contract')."""
    max_experiments: int                      # 10000 bare_adaptive; 50000 bare_fixed/ed/combined; 2000000 hwindow
    check_every: int                          # 100 (bare, ed-ps) or 10 (ed-fl, combined-fl)
    check_counter: Literal["round0", "round1", "update"]
    tolerance: float = 1.2                    # converge when |omega_hat - omega|_circ < tolerance * eps
    track_best_error: bool = False            # True only for bare_adaptive
    cache_mb: int = 2048
    stop: Literal["posterior", "legacy"] = "posterior"
        # posterior: truth-free stop when the posterior 95% half-width (2 std) <= eps, checked
        #   every update; converged iff |MAP - omega|_circ < tolerance * eps at the stop.
        # legacy: stop at the first check (legacy cadence) with |MAP - omega|_circ < tolerance * eps.


@dataclass(frozen=True)
class RunSpec:
    """Everything that identifies one simulated cell (the CSV grouping keys)."""
    protocol: Literal["bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow"]
    gamma: float = 0.0
    h: float = 0.0                            # relative std of heterogeneous gamma_k (unused in v2 suite; kept for RNG replay)
    L: int = 1                                # repetition-code half-distance; N_code = 2L+1
    sigma: float = 0.0                        # shot-to-shot signal offset std (combined, sequential, hwindow)
    inference: Literal["fl", "ps"] = "fl"     # full-likelihood vs post-selection (rejection) for ed/combined
    marg: bool = False                        # combined marginalised-visibility inference
    marg_cap: bool = False                    # marg: cap the depth at the information-optimal M = 1/(2 sigma sqrt(N_total))
    q: float = 0.0                            # readout flip probability per measured bit
    k: float = 1.0                            # entangling-gate fidelity (prep visibility k^(n-1) F)
    F: float = 1.0                            # single-qubit init fidelity
    estimator: Literal["mismatched", "aware"] = "mismatched"   # how inference treats q, k, F
    model: Literal["exact", "theorem"] = "exact"               # ED/combined likelihood: exact circuit or thm:bitflip closed form
    extended: bool = False                    # extended eps range (legacy --extended)
    eps_tar: Optional[float] = None           # hwindow: single target eps
    N_fixed: Optional[int] = None             # hwindow: fixed probe size
    stop: Literal["posterior", "legacy"] = "posterior"   # stopping rule (see EngineConfig.stop)

    @property
    def N_code(self) -> int:
        return 2 * self.L + 1

    @property
    def N_total(self) -> int:
        return 3 * self.N_code if self.protocol == "combined" else self.N_code

    def mode_label(self) -> str:
        """Legacy 'mode' column labels (plan: CSV schema)."""
        if self.protocol == "bare_adaptive":
            return "bare_hamiltonian_adaptive"
        if self.protocol == "bare_fixed":
            return "bare_hamiltonian"
        if self.protocol == "ed":
            return "ed_full_likelihood" if self.inference == "fl" else "ed_postselect"
        if self.protocol == "combined":
            if self.marg:
                return "combined_full_likelihood_marg_capped" if self.marg_cap else "combined_full_likelihood_marg"
            return "combined_full_likelihood" if self.inference == "fl" else "combined_postselect"
        if self.protocol == "sequential":
            return "sequential"
        return "hwindow"


def epsilon_grid(extended: bool = False, protocol: str = "bare_adaptive") -> np.ndarray:
    """The exact legacy np.geomspace calls (eps feeds the per-eps RNG seed)."""
    if not extended:
        return np.geomspace(1e-4, 1e-1, 60)          # sim:216, bh:213, ed:329, cb:461, sq:59
    # legacy extended ranges differ between simulation.py:213 and bh:210 (plan item 16)
    if protocol == "bare_adaptive":
        return np.geomspace(10 ** (-4.5), 1e-1, 70)  # sim:213
    return np.geomspace(1e-6, 1e-4, 40)              # bh:210, ed:326, cb:458, sq:148


def grid_resolution(N: int, eps_min: float) -> int:
    """Grid exponent m = max(16, ceil(log2(4 N floor(1/eps_min)))) (bh:50-55)."""
    import math
    M_max_needed = max(1, int(1.0 / eps_min))
    return max(16, math.ceil(math.log2(4 * N * M_max_needed)))


def window_N_grid(sigma: float, n_points: int = 12) -> list:
    """Heisenberg-window sweep: 12 log-spaced integers from N*/16 to 4 N*, N* = 1/(2 sigma^2)."""
    N_star = 1.0 / (2.0 * float(sigma) ** 2)
    Ns = np.rint(np.geomspace(N_star / 16.0, 4.0 * N_star, n_points)).astype(int)
    return sorted(set(int(n) for n in np.maximum(Ns, 1)))


CSV_COLUMNS = [
    "protocol", "mode", "seed", "epsilon", "gamma", "h", "sigma_epsilon", "L", "N_code", "N_total", "N",
    "q", "k", "F", "estimator", "model",
    "M_max", "M_exec_max", "grid_m", "total_resources", "converged", "final_error",
    "n_rounds", "n_updates", "n_accepted", "n_rejected", "acceptance_rate",
    "n_syndrome_used", "mean_d", "sum_F_obs", "sum_M_sq", "visibility",
    "stop_rule", "posterior_std",
    "engine_version", "jax_version", "timestamp",
]
GROUP_KEYS = ["protocol", "mode", "gamma", "h", "sigma_epsilon", "L", "q", "k", "F", "estimator", "model", "stop_rule"]
