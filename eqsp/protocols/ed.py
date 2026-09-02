"""Error-detected GHZ, one [[N=2L+1, 1]] repetition-code block (simulation_error_detected.py).

Inference ``fl`` (full likelihood, every round, check every 10 updates, ed:267)
or ``ps`` (rejection of d > 0, check every 100 accepted, ed:276-278, :298).
M_max = min(floor(1/(eps N)), 2^(m-2)//N) (ed:161-164).  Draw order gammas-if-h,
M_evols, uniforms_outcome, uniforms_syndrome(50000, N) (ed:166-182), then the
readout flips (syndrome bits, parity) appended when q > 0.

model="theorem": legacy kernels and legacy truth (realized error pattern used
on both sides, ed:221-265).  model="exact": truth samples the parity from the
exact coset mixture (hamiltonian.exact_p_plus) and inference uses
hamiltonian.exact_ed_update on the observed syndrome.
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


class ErrorDetected:
    period = 2.0 * float(np.pi)
    grid_builder = "np64"
    family = "ham"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.fl = spec.inference == "fl"
        self.theorem = spec.model == "theorem"
        self.N = spec.N_code
        self.L = spec.L
        if spec.h > 0 and (self.fl or not self.theorem):
            raise ValueError("ed: h > 0 is supported only for inference=ps with model=theorem (ed:157)")
        self.cfg = EngineConfig(max_experiments=50000, check_every=10 if self.fl else 100, check_counter="update")
        self.omega_true = OMEGA_TRUE
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
        self.xs = make_grid(m, 2.0 * np.pi, "np64")
        self.omega_grid = self.xs
        self._new_caches()

    def _new_caches(self):
        G = 2**self.m
        self.cache_phi = GridCache(G, self.cfg.cache_mb)      # phi_single_cache (ed:203)
        self.cache_p = GridCache(G, self.cfg.cache_mb)        # p_k_cache (ed:204)
        self.cache_phieff = GridCache(G, self.cfg.cache_mb)   # phi_eff_cache (ed:202)

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N = self.N
        M_max = max(1, int(1.0 / (eps * N)))                             # ed:161
        nyquist_limit = (2 ** (self.m - 2)) // N
        M_max = min(M_max, max(1, nyquist_limit))                        # ed:163-164
        self.M_max = M_max
        n_exp = self.cfg.max_experiments
        self.draws = {}
        if spec.h > 0:
            gammas_np = np.clip(rng.normal(spec.gamma, spec.gamma * spec.h, N), 0.0, 10.0)   # ed:167-168
            self.draws["gammas"] = gammas_np
            self._new_caches()
        else:
            gammas_np = np.full(N, spec.gamma)
        self.gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)
        M_evols = rng.integers(1, M_max + 1, n_exp)                      # ed:180
        uniforms_outcome = rng.random(n_exp)                             # ed:181
        uniforms_syndrome = rng.random((n_exp, N))                       # ed:182
        self.draws.update({"M_evols": M_evols, "uniforms_outcome": uniforms_outcome,
                           "uniforms_syndrome": uniforms_syndrome})
        self.M_evols = M_evols

        # truth side (float64, batched over shots)
        if self.theorem:
            p_all, phi_all = H.legacy_params_ed(self.omega_true, gammas_np, M_evols)      # ed:184-200
        else:
            p_all, phi_all = H.exact_params(np.full(N, self.omega_true), gammas_np, M_evols)
        errors_all = uniforms_syndrome[:, :N] < p_all                     # ed:221
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

        if self.theorem:
            if self.fl:
                N_eff = N - d_all
                phi_eff_true = N_eff * phi_all[:, 0]                      # ed:236-237 (homogeneous)
            else:
                phi_eff_true = np.sum(phi_all, axis=1)                    # ed:200, :284
            p_plus = (1.0 + self.V * np.cos(2.0 * phi_eff_true)) / 2.0
        else:
            p_plus = H.exact_p_plus(p_all, phi_all, errors_all, 1, V=self.V)
        outcomes = np.where(uniforms_outcome < p_plus, 0, 1)             # ed:240, :286
        if spec.q > 0:
            outcomes = outcomes ^ (u_par < H.parity_flip_prob(spec.q, N)).astype(int)
        self.outcomes = outcomes
        self.d_all = d_all
        # inference-side pattern weight: legacy uses the realized pattern (theorem, q = 0)
        self.d_inf = d_all if (self.theorem and spec.q == 0) else d_obs
        # rejection (ps): legacy on realized d (ed:276); with observed syndromes on d_obs
        self.reject = (d_all > 0) if (self.theorem and spec.q == 0) else (d_obs > 0)
        if self.qaware:
            self.logc_all = H.weight_log_coeffs(syn_obs[:, 0, :], spec.q, N,
                                                theorem_L=(self.L if self.theorem else None))

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        d = int(self.d_all[i])
        rejected = bool((not self.fl) and self.reject[i])
        aux = {"M": M_evol, "d": d, "N_eff": self.N - d, "d_inf": int(self.d_inf[i])}
        if self.qaware:
            aux["logc"] = self.logc_all[i]
        return Shot(cost=self.N * M_evol, rejected=rejected, outcome=int(self.outcomes[i]), key=M_evol, aux=aux)

    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        Mf = jnp.float32(M_evol)
        N = self.N
        if self.theorem and not self.qaware:
            if self.fl:
                d = shot.aux["d_inf"]
                N_eff = N - d
                phi_single = self.cache_phi.get(M_evol, lambda: H.phi_grid_homogeneous(self.omega_grid, self.gammas_jax[0], Mf))
                phi_eff_grid = N_eff * phi_single                          # ed:253, outside jit
                p_k_grid = self.cache_p.get(M_evol, lambda: H.p_grid_homogeneous(self.omega_grid, self.gammas_jax[0], Mf))
                logpk = H.syndrome_update_homogeneous(logpk, p_k_grid, d, N)   # ed:264
            else:
                phi_eff_grid = self.cache_phieff.get(M_evol, lambda: H.phi_eff_grid_ed(self.omega_grid, self.gammas_jax, Mf))
            if self.log_vis == 0.0:
                return H.phase_update(logpk, phi_eff_grid, shot.outcome)   # ed:265, :296
            return H.phase_update_vis(logpk, phi_eff_grid, shot.outcome, jnp.float32(self.log_vis))
        # exact model, or q-aware sum (either model)
        p_fun = H.p_grid_homogeneous if self.theorem else H.p_grid_homogeneous_exact
        p_grid = self.cache_p.get(M_evol, lambda: p_fun(self.omega_grid, self.gammas_jax[0], Mf))
        phi_grid = self.cache_phi.get(M_evol, lambda: H.phi_grid_homogeneous(self.omega_grid, self.gammas_jax[0], Mf))
        if self.qaware:
            logc = shot.aux["logc"]
        else:
            d = shot.aux["d_inf"]
            logc = np.full(N + 1, -np.inf)
            logc[d] = 0.0
            logc[N - d] = 0.0
            if self.theorem:
                logc[N - d] = -np.inf                                     # theorem + q>0 mismatched: one coset member
        return H.exact_ed_update(logpk, p_grid, phi_grid, jnp.asarray(logc, dtype=jnp.float32),
                                 jnp.float32(self.log_vis), shot.outcome)

    def account(self, shot: Shot, ctr) -> None:
        M = shot.key
        d = shot.aux["d"]
        if self.fl:
            if d > 0:
                ctr.n_syndrome_used += 1                                  # ed:230-231
            ctr.sum_d += d
            ctr.sum_F_obs += 4.0 * (M ** 2) * ((self.N - d) ** 2)          # ed:233
        else:
            ctr.sum_F_obs += 4.0 * (M ** 2) * (self.N ** 2)                # ed:282
        ctr.sum_M_sq += M ** 2

    def check_due(self, i: int, ctr) -> bool:
        return check_due(self.cfg, i, ctr)

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "N_code": self.N, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> ErrorDetected:
    return ErrorDetected(spec)
