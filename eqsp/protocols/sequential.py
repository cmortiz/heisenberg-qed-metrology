"""Sequential binary search with product-state probes, NumPy only.

K = 30 shots per round, C_M = pi/2, W0 = pi, at most 200 rounds; quenched offsets
eps_k from seed_rng(seed); per-target RNG eps_rng(seed, eps); truth wrapped
omega % W0 and pi-circular error.  The CSV records both M_exec_max (deepest
executed round depth) and the analytic bound M_max_used = ceil(C_M/(2 eps sqrt(N)))
(stored in M_max).  Realistic noise: prep flip with probability (1-F)/2 per qubit
per shot (XOR on the outcome), readout flip with probability q per Z outcome,
both drawn after the outcome uniforms of the round.
"""
from __future__ import annotations

import numpy as np

from ..config import OMEGA_TRUE, RunSpec
from ..engine import Counters, EpsResult
from ..io import eps_rng, make_row, seed_rng

W0 = np.pi                    # prior width
C_M = np.pi / 2               # depth constant M_r = ceil(C_M / W_r)
SHOTS_PER_ROUND = 30
MAX_ROUNDS = 200
CONVERGENCE_TOLERANCE = 1.2


class Sequential:
    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.N = spec.N_code
        self.omega_true = OMEGA_TRUE
        self.draws: dict = {}

    def run_seed(self, seed: int, epsilons, engine_version: str, jax_version: str = "none") -> list[dict]:
        spec = self.spec
        N = self.N
        rng_seed = seed_rng(seed)
        if spec.sigma > 0:
            epsilons_k = rng_seed.normal(0.0, spec.sigma, N)
        else:
            epsilons_k = np.zeros(N)
        f_prep = 0.5 * (1.0 - spec.F)
        rows = []
        for eps_target in np.asarray(epsilons, np.float64):
            eps_target = float(eps_target)
            target_rng = eps_rng(seed, eps_target)
            omega_wrapped = self.omega_true % W0
            lo, hi = 0.0, W0
            total_resources = 0
            n_rounds = 0
            M_exec_max = 0
            draws = []
            for r in range(MAX_ROUNDS):
                width = hi - lo
                if width <= 2.0 * eps_target:
                    break
                mid = 0.5 * (lo + hi)
                M_r = max(1, int(np.ceil(C_M / width)))
                arg = M_r * (omega_wrapped - mid + epsilons_k) + 0.25 * np.pi
                p_one = np.sin(arg) ** 2
                uniforms = target_rng.random((SHOTS_PER_ROUND, N))
                outcomes_one = (uniforms < p_one[None, :]).astype(np.int32)
                draws.append(uniforms)
                if spec.F < 1.0:
                    prep = target_rng.random((SHOTS_PER_ROUND, N)) < f_prep
                    outcomes_one = outcomes_one ^ prep.astype(np.int32)
                if spec.q > 0.0:
                    flips = target_rng.random((SHOTS_PER_ROUND, N)) < spec.q
                    outcomes_one = outcomes_one ^ flips.astype(np.int32)
                n_one = int(np.sum(outcomes_one))
                n_total = SHOTS_PER_ROUND * N
                if n_one > n_total // 2:                                 # majority vote
                    lo = mid
                else:
                    hi = mid
                total_resources += SHOTS_PER_ROUND * N * M_r
                n_rounds = r + 1
                M_exec_max = max(M_exec_max, M_r)
            omega_hat = 0.5 * (lo + hi)
            diff = abs(omega_hat - omega_wrapped)
            final_error = min(diff, W0 - diff)
            converged = bool(final_error < CONVERGENCE_TOLERANCE * eps_target)
            self.draws = {"uniforms": draws}
            ctr = Counters(resources=total_resources, n_rounds=n_rounds, M_exec_max=M_exec_max)
            M_max_used = max(1, int(np.ceil(C_M / (eps_target * 2.0 * np.sqrt(N)))))
            res = EpsResult(converged, final_error, ctr,
                            {"N": N, "N_code": N, "N_total": N, "M_max": M_max_used})
            rows.append(make_row(spec, seed, eps_target, res, engine_version, jax_version))
        return rows

    def run_seeds(self, seeds, epsilons, engine_version: str, jax_version: str = "none",
                  progress: bool = False, on_seed_done=None) -> list[dict]:
        rows_all = []
        for seed in seeds:
            rows = self.run_seed(int(seed), epsilons, engine_version, jax_version)
            if on_seed_done is not None:
                on_seed_done(int(seed), rows)
            rows_all.extend(rows)
        return rows_all


def make(spec: RunSpec) -> Sequential:
    return Sequential(spec)
