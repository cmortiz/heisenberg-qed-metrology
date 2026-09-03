"""Engine configuration and run specification (shared contract; no jax import)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

OMEGA_TRUE = 0.3
MAX_N_ADAPTIVE = 16384          # Nyquist limit of the 2^16 grid
MIN_N_ADAPTIVE = 10


@dataclass(frozen=True)
class EngineConfig:
    """Per-protocol loop constants."""
    max_experiments: int                      # 10000 bare_adaptive; 50000 bare_fixed/ed/combined; 2000000 hwindow
    tolerance: float = 1.2                    # converged when |omega_hat - omega|_circ < tolerance * eps
    cache_mb: int = 2048


@dataclass(frozen=True)
class RunSpec:
    """Everything that identifies one simulated cell (the CSV grouping keys)."""
    protocol: Literal["bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow"]
    gamma: float = 0.0
    L: int = 1                                # repetition-code half-distance; N_code = 2L+1
    sigma: float = 0.0                        # shot-to-shot signal offset std (combined, sequential, hwindow)
    inference: Literal["fl", "ps"] = "fl"     # full-likelihood vs post-selection (rejection) for ed/combined
    marg: bool = False                        # combined marginalised-visibility inference
    marg_cap: bool = False                    # marg: cap the depth at the information-optimal M = 1/(2 sigma sqrt(N_total))
    q: float = 0.0                            # readout flip probability per measured bit
    k: float = 1.0                            # entangling-gate fidelity (prep visibility k^(n-1) F)
    F: float = 1.0                            # single-qubit init fidelity
    estimator: Literal["mismatched", "aware"] = "mismatched"   # how inference treats q, k, F
    extended: bool = False                    # extended eps range
    eps_tar: Optional[float] = None           # hwindow: single target eps
    N_fixed: Optional[int] = None             # hwindow: fixed probe size

    @property
    def N_code(self) -> int:
        return 2 * self.L + 1

    @property
    def N_total(self) -> int:
        return 3 * self.N_code if self.protocol == "combined" else self.N_code

    def mode_label(self) -> str:
        """The CSV 'mode' column."""
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
    """Target grid; eps feeds the per-eps RNG seed, so the exact float64 values matter."""
    if not extended:
        return np.geomspace(1e-4, 1e-1, 60)          # the 60 targets of every suite cell
    if protocol == "bare_adaptive":
        return np.geomspace(10 ** (-4.5), 1e-1, 70)
    return np.geomspace(1e-6, 1e-4, 40)


def grid_resolution(N: int, eps_min: float) -> int:
    """Grid exponent m = max(16, ceil(log2(4 N floor(1/eps_min))))."""
    import math
    M_max_needed = max(1, int(1.0 / eps_min))
    return max(16, math.ceil(math.log2(4 * N * M_max_needed)))


def window_N_grid(sigma: float, n_points: int = 12) -> list:
    """Heisenberg-window sweep: 12 log-spaced integers from N*/16 to 4 N*, N* = 1/(2 sigma^2)."""
    N_star = 1.0 / (2.0 * float(sigma) ** 2)
    Ns = np.rint(np.geomspace(N_star / 16.0, 4.0 * N_star, n_points)).astype(int)
    return sorted(set(int(n) for n in np.maximum(Ns, 1)))


# The CSV column set of the v2 suite.  ``h``, ``model`` and ``stop_rule`` are constants
# (0, "exact", "posterior"): every paper row uses the exact likelihood model and the
# posterior stopping rule, and the columns are kept so the merged CSVs stay valid.
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
