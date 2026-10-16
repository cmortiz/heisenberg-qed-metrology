"""Bare GHZ, Hamiltonian noise, fixed odd N = 2L+1, M ~ U{1..M_max}.

50000 experiments per target, M_max = min(floor(1/eps), Nyquist cap), float64
grid cast to float32 on one M = 1 fringe period pi/N centred on omega.  Truth
side in float64 NumPy (odd N: the even-N population term vanishes, so the parity
is the coherent term alone).  Draw order: M_evols, uniforms_outcome, readout
uniforms (q > 0 only), thetas.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .. import closed_forms as cf
from .. import hamiltonian as H
from ..config import OMEGA_TRUE, EngineConfig, RunSpec, grid_resolution
from ..engine import GridCache, Shot, make_grid
from ..io import eps_rng


class BareFixed:
    period = 2.0 * float(np.pi)
    grid_builder = "np64"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.cfg = EngineConfig(max_experiments=50000)
        self.omega_true = OMEGA_TRUE
        self.N = spec.N_code
        if self.N % 2 == 0:
            raise ValueError("bare_fixed assumes odd N = 2L+1")
        # prior: one period pi/N of the M = 1 fringe, centred on omega (the parity likelihood
        # is pi/N-periodic in the phase up to O(gamma^2/Omega^2) amplitude terms, so a wider
        # prior carries near-degenerate mirror modes)
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
        self.omega_grid = self.xs
        self.cache = GridCache(2**m, self.cfg.cache_mb)

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N = self.N
        # matched_depth: the ED/combined schedule floor(1/(eps N)).  Both likelihoods carry the
        # same fringe frequency 2(N-d)M omega, so resolving eps needs N*M_max ~ 1/eps; the
        # legacy floor(1/eps) runs N times deeper and inflates the per-shot cost N*M by N.
        M_max = max(1, int(1.0 / (eps * N))) if spec.matched_depth else max(1, int(1.0 / eps))
        nyquist_limit = H.nyquist_cap(self.m, N, self.period)
        M_max = min(M_max, max(1, nyquist_limit))
        self.M_max = M_max
        n_exp = self.cfg.max_experiments
        self.draws = {}
        gammas_np = np.full(N, spec.gamma)
        self.gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)
        M_evols = rng.integers(1, M_max + 1, n_exp)
        uniforms_outcome = rng.random(n_exp)
        self.draws.update({"M_evols": M_evols, "uniforms_outcome": uniforms_outcome})
        self.M_evols = M_evols
        self.uniforms_outcome = uniforms_outcome
        self.u_readout = None
        if spec.q > 0:
            self.u_readout = rng.random(n_exp)
            self.draws["uniforms_readout_parity"] = self.u_readout
            self.p_flip = H.parity_flip_prob(spec.q, N)
        # per-shot preparation phase (drawn last) and float64 truth
        self.thetas = rng.uniform(0.0, 2.0 * np.pi, n_exp)
        self.draws["thetas"] = self.thetas
        self.coh_true = cf.bare_parity_coherent(self.omega_true, spec.gamma, M_evols, N, self.thetas)

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        x_true = float(self.coh_true[i])                                 # odd N: no population term
        p_plus = (1.0 + self.V * x_true) / 2.0
        outcome = 0 if self.uniforms_outcome[i] < p_plus else 1
        if self.u_readout is not None and self.u_readout[i] < self.p_flip:
            outcome ^= 1
        return Shot(cost=self.N * M_evol, rejected=False, outcome=outcome, key=M_evol,
                    aux={"M": M_evol, "theta": float(self.thetas[i])})

    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        coh = self.cache.get(M_evol, lambda: H.ghz_x_parity_grid_complex(self.omega_grid, self.gammas_jax, jnp.float32(M_evol)))
        return H.bare_parity_update_theta(logpk, coh, jnp.float32(shot.aux["theta"]), jnp.float32(self.log_vis), shot.outcome)

    def account(self, shot: Shot, ctr) -> None:
        ctr.sum_M_sq += shot.key ** 2

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "N_code": self.N, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> BareFixed:
    return BareFixed(spec)
