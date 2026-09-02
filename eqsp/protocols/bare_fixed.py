"""Bare GHZ, Hamiltonian noise, fixed odd N = 2L+1, M ~ U{1..M_max} (simulation_bare_hamiltonian.py).

Loop constants bh:44-48 (50000, check (i+1) % 100, final error on non-convergence),
M_max = min(floor(1/eps), 2^(m-2)//N) (bh:145-147), [0, 2pi) float64-then-float32
grid (bh:156-158), float32 truth kernel ghz_x_parity_true (bh:177, legacy
convention 10), draw order gammas-if-h, M_evols, uniforms_outcome (bh:149-164).
For odd N the legacy kernel is the exact parity (the even-N C^N term vanishes),
so ``model`` has no effect here.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .. import closed_forms as cf
from .. import hamiltonian as H
from ..config import OMEGA_TRUE, EngineConfig, RunSpec, grid_resolution
from ..engine import GridCache, Shot, check_due, make_grid
from ..io import eps_rng


class BareFixed:
    period = 2.0 * float(np.pi)
    grid_builder = "np64"
    family = "ham"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.cfg = EngineConfig(stop=spec.stop, max_experiments=50000, check_every=100, check_counter="round1")
        self.omega_true = OMEGA_TRUE
        self.N = spec.N_code
        if self.N % 2 == 0:
            raise ValueError("bare_fixed assumes odd N = 2L+1 (bh:143)")
        self.theorem = spec.model == "theorem"
        if self.theorem:
            self.period, self.grid_offset = 2.0 * float(np.pi), 0.0        # legacy [0, 2pi)
        else:
            if spec.h > 0:
                raise ValueError("bare_fixed exact model: heterogeneous gamma (h > 0) is legacy-only")
            # prior: one period pi/N of the M = 1 fringe, centred on omega (the parity likelihood
            # is pi/N-periodic in the phase up to O(gamma^2/Omega^2) amplitude terms, so a wider
            # prior carries near-degenerate mirror modes; finding 2026-09-02)
            self.period = float(np.pi) / self.N
            self.grid_offset = OMEGA_TRUE - 0.5 * self.period
        self.V = cf.prep_visibility(spec.k, spec.F, self.N)
        self.log_vis = (self.N * math.log1p(-2.0 * spec.q) + math.log(self.V)) if spec.estimator == "aware" else 0.0
        self.draws: dict = {}

    def grid_m(self, eps_min: float) -> int:
        return grid_resolution(self.N, eps_min)

    def rng_for(self, seed: int, eps: float):
        return eps_rng(seed, eps)

    def begin_run(self, seed: int, m: int) -> None:
        self.m = m
        self.xs = make_grid(m, self.period, "np64", offset=self.grid_offset)
        self.omega_grid = self.xs                                        # bh:158 (same values)
        self.omega_jax = jnp.float32(self.omega_true)
        self.cache = GridCache(2**m, self.cfg.cache_mb)                  # bh:166-167 (persisted when h = 0)

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N = self.N
        M_max = max(1, int(1.0 / eps))                                   # bh:145
        nyquist_limit = H.nyquist_cap(self.m, N, self.period)            # bh:146-147 on a 2 pi grid
        M_max = min(M_max, max(1, nyquist_limit))
        self.M_max = M_max
        n_exp = self.cfg.max_experiments
        self.draws = {}
        if spec.h > 0:
            gammas_np = np.clip(rng.normal(spec.gamma, spec.gamma * spec.h, N), 0.0, 10.0)   # bh:150-151
            self.draws["gammas"] = gammas_np
            self.cache = GridCache(2**self.m, self.cfg.cache_mb)         # gammas change per eps
        else:
            gammas_np = np.full(N, spec.gamma)
        self.gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)
        M_evols = rng.integers(1, M_max + 1, n_exp)                      # bh:163
        uniforms_outcome = rng.random(n_exp)                             # bh:164
        self.draws.update({"M_evols": M_evols, "uniforms_outcome": uniforms_outcome})
        self.M_evols = M_evols
        self.uniforms_outcome = uniforms_outcome
        self.u_readout = None
        if spec.q > 0:
            self.u_readout = rng.random(n_exp)
            self.draws["uniforms_readout_parity"] = self.u_readout
            self.p_flip = H.parity_flip_prob(spec.q, N)
        # exact model: per-shot preparation phase (drawn last) and float64 truth
        self.thetas = np.zeros(n_exp)
        if not self.theorem:
            self.thetas = rng.uniform(0.0, 2.0 * np.pi, n_exp)
            self.draws["thetas"] = self.thetas
            self.coh_true = cf.bare_parity_coherent(self.omega_true, spec.gamma, M_evols, N, self.thetas)

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        if self.theorem:
            x_true = float(H.ghz_x_parity_true(self.omega_jax, self.gammas_jax, jnp.float32(M_evol)))   # bh:177
        else:
            x_true = float(self.coh_true[i])                             # odd N: no population term
        p_plus = (1.0 + self.V * x_true) / 2.0                          # bh:178 (V = 1 legacy)
        outcome = 0 if self.uniforms_outcome[i] < p_plus else 1         # bh:179
        if self.u_readout is not None and self.u_readout[i] < self.p_flip:
            outcome ^= 1
        return Shot(cost=self.N * M_evol, rejected=False, outcome=outcome, key=M_evol,
                    aux={"M": M_evol, "theta": float(self.thetas[i])})

    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        if not self.theorem:
            coh = self.cache.get(M_evol, lambda: H.ghz_x_parity_grid_complex(self.omega_grid, self.gammas_jax, jnp.float32(M_evol)))
            return H.bare_parity_update_theta(logpk, coh, jnp.float32(shot.aux["theta"]), jnp.float32(self.log_vis), shot.outcome)
        grid = self.cache.get(M_evol, lambda: H.ghz_x_parity_grid(self.omega_grid, self.gammas_jax, jnp.float32(M_evol)))
        if self.log_vis == 0.0:
            return H.bare_parity_update(logpk, grid, shot.outcome)       # bh:190
        return H.bare_parity_update_vis(logpk, grid, shot.outcome, jnp.float32(self.log_vis))

    def account(self, shot: Shot, ctr) -> None:
        ctr.sum_M_sq += shot.key ** 2

    def check_due(self, i: int, ctr) -> bool:
        return check_due(self.cfg, i, ctr)

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "N_code": self.N, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> BareFixed:
    return BareFixed(spec)
