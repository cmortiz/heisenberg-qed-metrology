"""Bare GHZ, Hamiltonian noise, adaptive N = clip(floor(1/eps), 10, 16384), n ~ U{1..N}, M = 1.

Loop constants and grid from simulation.py (sim:32-37, :137-146, :161-162, :196-204):
10000 experiments, check every 100 rounds including round 0, best error on
non-convergence, [0, pi) float32 jnp.linspace grid with m = 16.  The likelihood
is the exact bare parity r^n cos(n theta) + [n even] C^n
(closed_forms.bare_parity) evaluated by the fixed-point kernel
hamiltonian.bare_adaptive_update; at gamma = 0 it reproduces the legacy
integer-mod angle bitwise.  Truth side in float64 NumPy (plan, Per-protocol mapping).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .. import closed_forms as cf
from .. import hamiltonian as H
from ..config import MAX_N_ADAPTIVE, MIN_N_ADAPTIVE, OMEGA_TRUE, EngineConfig, RunSpec
from ..engine import Shot, check_due, make_grid
from ..io import eps_rng


class BareAdaptive:
    period = float(np.pi)
    grid_builder = "jax32"
    family = "ham"

    def __init__(self, spec: RunSpec):
        if spec.h != 0.0:
            raise ValueError("bare_adaptive: heterogeneous gamma (h > 0) is not supported (dropped in v2)")
        self.spec = spec
        self.cfg = EngineConfig(max_experiments=10000, check_every=100, check_counter="round0",
                                track_best_error=True)
        self.omega_true = OMEGA_TRUE
        self.gamma = float(spec.gamma)
        self.aware = spec.estimator == "aware"
        self.draws: dict = {}

    # -- run level -----------------------------------------------------------
    def grid_m(self, eps_min: float) -> int:
        return 16                                                       # sim:32

    def rng_for(self, seed: int, eps: float):
        return eps_rng(seed, eps)

    def begin_run(self, seed: int, m: int) -> None:
        self.m = m
        self.xs = make_grid(m, np.pi, "jax32")                          # sim:143
        self.tables = H.bare_adaptive_tables(m, self.gamma)

    # -- per eps ---------------------------------------------------------------
    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N = int(1.0 / eps)
        N = max(MIN_N_ADAPTIVE, min(N, MAX_N_ADAPTIVE))                  # sim:137-138
        self.N = N
        n_exp = self.cfg.max_experiments
        ns = rng.integers(1, N + 1, n_exp)                              # sim:161
        uniforms = rng.random(n_exp)                                    # sim:162
        self.draws = {"ns": ns, "uniforms": uniforms}
        # truth: exact bare parity, prep visibility k^(n-1) F, float64
        coherent, population = H.bare_parity_batch(self.omega_true, self.gamma, 1, ns)
        V = cf.prep_visibility(spec.k, spec.F, ns)
        p_plus = 0.5 * (1.0 + V * coherent + population)      # V scales the coherent term only
        outcomes = np.where(uniforms < p_plus, 0, 1)
        if spec.q > 0:
            u_r = rng.random(n_exp)                                     # appended after legacy draws
            self.draws["uniforms_readout_parity"] = u_r
            outcomes = outcomes ^ (u_r < H.parity_flip_prob(spec.q, ns)).astype(int)
        self.ns = ns
        self.outcomes = outcomes
        if self.aware:
            self.log_ro = ns * math.log1p(-2.0 * spec.q)
            self.log_V = (ns - 1) * math.log(spec.k) + math.log(spec.F)
        else:
            self.log_ro = np.zeros(n_exp)
            self.log_V = np.zeros(n_exp)

    def draw(self, i: int) -> Shot:
        n = int(self.ns[i])
        return Shot(cost=n, rejected=False, outcome=int(self.outcomes[i]), key=n,
                    aux={"n": n, "log_ro": float(self.log_ro[i]), "log_V": float(self.log_V[i])})

    def apply(self, logpk, shot: Shot):
        q0, q1, q2, logr, logC = self.tables
        return H.bare_adaptive_update(logpk, q0, q1, q2, logr, logC, shot.key,
                                      shot.aux["log_ro"], shot.aux["log_V"], shot.outcome)

    def account(self, shot: Shot, ctr) -> None:
        ctr.sum_M_sq += 1.0

    def check_due(self, i: int, ctr) -> bool:
        return check_due(self.cfg, i, ctr)

    def row(self, ctr, converged: bool, error: float) -> dict:
        spec = self.spec
        return {"N": self.N, "M_max": 1, "grid_m": self.m,
                "visibility": cf.prep_visibility(spec.k, spec.F, self.N) * cf.readout_visibility(spec.q, self.N)}


def make(spec: RunSpec) -> BareAdaptive:
    return BareAdaptive(spec)
