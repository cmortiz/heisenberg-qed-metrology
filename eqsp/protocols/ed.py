"""Error-detected GHZ, one [[N=2L+1, 1]] repetition-code block.

Inference ``fl`` (full likelihood: every shot updates, the syndrome weight enters
the likelihood) or ``ps`` (rejection: a shot with a non-trivial observed syndrome
is charged and discarded).  50000 experiments per target,
M_max = min(floor(1/(eps N)), Nyquist cap), grid on one M = 1 fringe period pi/N
centred on omega.  Truth samples the parity from the exact coset mixture
(hamiltonian.exact_p_plus); inference is hamiltonian.exact_ed_update on the
observed syndrome.  Draw order: M_evols, uniforms_outcome,
uniforms_syndrome(50000, N), readout uniforms (q > 0 only), thetas.
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


class ErrorDetected:
    grid_builder = "np64"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.fl = spec.inference == "fl"
        self.N = spec.N_code
        self.L = spec.L
        self.cfg = EngineConfig(max_experiments=50000)
        self.omega_true = OMEGA_TRUE
        self.period = float(np.pi) / self.N                               # one M = 1 fringe period, centred on omega
        self.grid_offset = OMEGA_TRUE - 0.5 * self.period
        self.V = cf.prep_visibility(spec.k, spec.F, self.N)
        self.aware = spec.estimator == "aware"
        self.qaware = self.aware and spec.q > 0
        self.log_vis = (self.N * math.log1p(-2.0 * spec.q) + math.log(self.V)) if self.aware else 0.0
        self.draws: dict = {}

    def grid_m(self, eps_min: float) -> int:
        return grid_resolution(self.N, eps_min)

    def rng_for(self, seed: int, eps: float):
        return eps_rng(seed, eps)

    def begin_run(self, seed: int, m: int) -> None:
        self.m = m
        self.xs = make_grid(m, self.period, "np64", offset=self.grid_offset)
        self.omega_grid = self.xs
        self._new_caches()

    def _new_caches(self):
        G = 2**self.m
        self.cache_phi = GridCache(G, self.cfg.cache_mb)      # phi_k grid per depth
        self.cache_p = GridCache(G, self.cfg.cache_mb)        # p_k grid per depth

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N = self.N
        M_max = max(1, int(1.0 / (eps * N)))
        nyquist_limit = H.nyquist_cap(self.m, N, self.period)
        M_max = min(M_max, max(1, nyquist_limit))
        self.M_max = M_max
        n_exp = self.cfg.max_experiments
        self.draws = {}
        gammas_np = np.full(N, spec.gamma)
        self.gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)
        M_evols = rng.integers(1, M_max + 1, n_exp)
        uniforms_outcome = rng.random(n_exp)
        uniforms_syndrome = rng.random((n_exp, N))
        self.draws.update({"M_evols": M_evols, "uniforms_outcome": uniforms_outcome,
                           "uniforms_syndrome": uniforms_syndrome})
        self.M_evols = M_evols

        # truth side (float64, batched over shots)
        p_all, phi_all = H.exact_params(np.full(N, self.omega_true), gammas_np, M_evols)
        errors_all = uniforms_syndrome[:, :N] < p_all
        d_all = errors_all.sum(axis=1)
        syn_true = H.block_syndromes(errors_all, 1)                       # (S, 1, N-1)
        if spec.q > 0:
            u_syn = rng.random((n_exp, N - 1))
            u_par = rng.random(n_exp)
            self.draws["uniforms_readout_syndrome"] = u_syn
            self.draws["uniforms_readout_parity"] = u_par
            syn_obs = syn_true ^ (u_syn < spec.q)[:, None, :]
        else:
            syn_obs = syn_true
        pattern_obs = H.decode_min_weight(syn_obs)                        # (S, N), weight <= L
        d_obs = pattern_obs.sum(axis=1)
        # per-shot preparation phase, drawn after every outcome and readout draw
        self.thetas = rng.uniform(0.0, 2.0 * np.pi, n_exp)
        self.draws["thetas"] = self.thetas

        p_plus = H.exact_p_plus(p_all, phi_all, errors_all, 1, V=self.V, theta=self.thetas)
        outcomes = np.where(uniforms_outcome < p_plus, 0, 1)
        if spec.q > 0:
            outcomes = outcomes ^ (u_par < H.parity_flip_prob(spec.q, N)).astype(int)
        self.outcomes = outcomes
        self.d_all = d_all
        self.d_inf = d_obs            # inference sees only the decoded observed syndrome
        self.reject = d_obs > 0       # rejection (ps): discard any non-trivial observed syndrome
        if self.qaware:
            self.logc_all = H.weight_log_coeffs(syn_obs[:, 0, :], spec.q, N)

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        d = int(self.d_all[i])
        rejected = bool((not self.fl) and self.reject[i])
        aux = {"M": M_evol, "d": d, "N_eff": self.N - d, "d_inf": int(self.d_inf[i]), "theta": float(self.thetas[i])}
        if self.qaware:
            aux["logc"] = self.logc_all[i]
        return Shot(cost=self.N * M_evol, rejected=rejected, outcome=int(self.outcomes[i]), key=M_evol, aux=aux)

    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        Mf = jnp.float32(M_evol)
        N = self.N
        p_grid = self.cache_p.get(M_evol, lambda: H.p_grid_homogeneous_exact(self.omega_grid, self.gammas_jax[0], Mf))
        phi_grid = self.cache_phi.get(M_evol, lambda: H.phi_grid_homogeneous(self.omega_grid, self.gammas_jax[0], Mf))
        if self.qaware:
            logc = shot.aux["logc"]
        else:
            d = shot.aux["d_inf"]
            logc = np.full(N + 1, -np.inf)                               # the two coset members of the observed syndrome
            logc[d] = 0.0
            logc[N - d] = 0.0
        return H.exact_ed_update(logpk, p_grid, phi_grid, jnp.asarray(logc, dtype=jnp.float32),
                                 jnp.float32(self.log_vis), shot.outcome, jnp.float32(shot.aux["theta"]))

    def account(self, shot: Shot, ctr) -> None:
        M = shot.key
        d = shot.aux["d"]
        if self.fl:
            if d > 0:
                ctr.n_syndrome_used += 1
            ctr.sum_d += d
            ctr.sum_F_obs += 4.0 * (M ** 2) * ((self.N - d) ** 2)
        else:
            ctr.sum_F_obs += 4.0 * (M ** 2) * (self.N ** 2)
        ctr.sum_M_sq += M ** 2

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "N_code": self.N, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> ErrorDetected:
    return ErrorDetected(spec)
