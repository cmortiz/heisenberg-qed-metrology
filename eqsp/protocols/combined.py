"""Combined protocol: logical GHZ over three [[N_code, 1]] repetition-code blocks
with per-qubit offsets omega_k = omega + eps_k (simulation_combined.py).

Quenched offsets eps_k ~ N(0, sigma^2) from seed_rng(seed) once per seed
(cb:465-469); marg mode redraws them per shot on the truth side
(cb:239-249) and infers with eps = 0 and visibility exp(-2 M^2 N_clean sigma^2)
(cb:234-238, :376).  M_max = min(floor(1/(eps N_total)), 2^(m-2)//N_total)
(cb:209-211).  Draw order gammas-if-h, M_evols, uniforms_outcome,
uniforms_syndrome(50000, N_total), epsilons_per_shot-if-marg (cb:213-242), then
readout flips when q > 0.  FL update order: per-qubit syndrome factor, phi_eff
(cached d = 0 grid or compute_phi_eff_with_errors), parity update (cb:342-381).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .. import closed_forms as cf
from .. import hamiltonian as H
from ..config import OMEGA_TRUE, EngineConfig, RunSpec, grid_resolution
from ..engine import GridCache, Shot, check_due, make_grid
from ..io import eps_rng, seed_rng

N_BLOCKS = 3


class Combined:
    period = 2.0 * float(np.pi)
    grid_builder = "np64"
    family = "ham"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.fl = spec.inference == "fl"
        self.theorem = spec.model == "theorem"
        self.marg = bool(spec.marg)
        if self.marg and not self.fl:
            raise ValueError("combined: marg requires inference=fl (mode combined_full_likelihood_marg)")
        if spec.h > 0 and not self.theorem:
            raise ValueError("combined: h > 0 is supported only with model=theorem")
        self.Nc = spec.N_code
        self.N = N_BLOCKS * self.Nc
        self.L = spec.L
        self.sigma = float(spec.sigma)
        self.cfg = EngineConfig(stop=spec.stop, max_experiments=50000, check_every=10 if self.fl else 100, check_counter="update")
        self.omega_true = OMEGA_TRUE
        self.V = cf.prep_visibility(spec.k, spec.F, self.Nc) ** N_BLOCKS      # product over blocks
        self.aware = spec.estimator == "aware"
        self.qaware = self.aware and spec.q > 0
        self.log_vis = (self.N * math.log1p(-2.0 * spec.q) + math.log(self.V)) if self.aware else 0.0
        self.X_all = None
        if self.qaware:
            self.X_all = jnp.asarray(np.broadcast_to(H.all_patterns(self.Nc)[None], (N_BLOCKS, 2**self.Nc, self.Nc)).astype(np.float32))
        self.draws: dict = {}

    def grid_m(self, eps_min: float) -> int:
        return grid_resolution(self.N, eps_min)

    def rng_for(self, seed: int, eps: float):
        return eps_rng(seed, eps)

    def begin_run(self, seed: int, m: int) -> None:
        self.m = m
        self.xs = make_grid(m, 2.0 * np.pi, "np64")
        self.omega_grid = self.xs
        rng_seed = seed_rng(seed)                                        # cb:465
        if self.sigma > 0:
            self.epsilons_np = rng_seed.normal(0.0, self.sigma, self.N)  # cb:467
        else:
            self.epsilons_np = np.zeros(self.N)
        self.omegas_np = self.omega_true + self.epsilons_np
        # inference offsets: zero in marg mode (cb:234-238)
        eps_inf = np.zeros(self.N) if self.marg else self.epsilons_np
        self.epsilons_jax = jnp.array(eps_inf, dtype=jnp.float32)
        self._new_caches()

    def _new_caches(self):
        G = 2**self.m
        self.cache_phieff = GridCache(G, self.cfg.cache_mb)                     # cb:266
        self.cache_phi_pq = GridCache(G, self.cfg.cache_mb, entry_width=self.N) # cb:267, :270
        self.cache_p_pq = GridCache(G, self.cfg.cache_mb, entry_width=self.N)   # cb:268, :270

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N, Nc = self.N, self.Nc
        M_max = max(1, int(1.0 / (eps * N)))                             # cb:209
        nyquist_limit = (2 ** (self.m - 2)) // N
        M_max = min(M_max, max(1, nyquist_limit))                        # cb:210-211
        self.M_max = M_max
        n_exp = self.cfg.max_experiments
        self.draws = {}
        if spec.h > 0:
            gammas_np = np.clip(rng.normal(spec.gamma, spec.gamma * spec.h, N), 0.0, 10.0)   # cb:214-215
            self.draws["gammas"] = gammas_np
            self._new_caches()
        else:
            gammas_np = np.full(N, spec.gamma)
        self.gammas_jax = jnp.array(gammas_np, dtype=jnp.float32)
        M_evols = rng.integers(1, M_max + 1, n_exp)                      # cb:230
        uniforms_outcome = rng.random(n_exp)                             # cb:231
        uniforms_syndrome = rng.random((n_exp, N))                       # cb:232
        self.draws.update({"M_evols": M_evols, "uniforms_outcome": uniforms_outcome,
                           "uniforms_syndrome": uniforms_syndrome})
        self.M_evols = M_evols
        if self.marg:
            epsilons_per_shot = rng.normal(0.0, self.sigma, (n_exp, N))  # cb:242
            self.draws["epsilons_per_shot"] = epsilons_per_shot
            omegas = self.omega_true + epsilons_per_shot                 # (S, N)
        else:
            omegas = self.omegas_np                                      # (N,)
        if self.theorem:
            p_all, phi_all = H.legacy_params_cb(omegas, gammas_np, M_evols)    # cb:249-264 / :287-298
        else:
            p_all, phi_all = H.exact_params(omegas, gammas_np, M_evols)
        errors_all = uniforms_syndrome[:, :N] < p_all                     # cb:302
        d_blocks = errors_all.reshape(n_exp, N_BLOCKS, Nc).sum(axis=2)   # cb:305-307
        d_total = d_blocks.sum(axis=1)
        syn_true = H.block_syndromes(errors_all, N_BLOCKS)               # (S, B, Nc-1)
        if spec.q > 0:
            u_syn = rng.random((n_exp, N_BLOCKS, Nc - 1))
            u_par = rng.random(n_exp)
            self.draws["uniforms_readout_syndrome"] = u_syn
            self.draws["uniforms_readout_parity"] = u_par
            syn_obs = syn_true ^ (u_syn < spec.q)
        else:
            syn_obs = syn_true
        pattern_obs = H.decode_min_weight(syn_obs)                        # (S, N)
        d_obs_blocks = pattern_obs.reshape(n_exp, N_BLOCKS, Nc).sum(axis=2)

        if self.theorem:
            if self.fl:
                mask_clean = np.where(errors_all, 0.0, 1.0)              # cb:323
                phi_eff_true = np.sum(mask_clean * phi_all, axis=1)       # cb:325-327
            else:
                phi_eff_true = np.sum(phi_all, axis=1)                    # cb:264, :402-404
            p_plus = (1.0 + self.V * np.cos(2.0 * phi_eff_true)) / 2.0
        else:
            p_plus = H.exact_p_plus(p_all, phi_all, errors_all, N_BLOCKS, V=self.V)
        outcomes = np.where(uniforms_outcome < p_plus, 0, 1)             # cb:330, :407
        if spec.q > 0:
            outcomes = outcomes ^ (u_par < H.parity_flip_prob(spec.q, N)).astype(int)
        self.outcomes = outcomes
        self.errors_all = errors_all
        self.d_total = d_total
        legacy_side = self.theorem and spec.q == 0
        self.pattern_inf = errors_all if legacy_side else pattern_obs
        self.reject = (d_total > 0) if legacy_side else (d_obs_blocks.sum(axis=1) > 0)   # cb:392
        if self.qaware:
            tL = self.L if self.theorem else None
            self.logw_all = np.stack([H.pattern_log_weights(syn_obs[:, b, :], spec.q, Nc, theorem_L=tL)
                                      for b in range(N_BLOCKS)], axis=1)   # (S, B, P)

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        d = int(self.d_total[i])
        rejected = bool((not self.fl) and self.reject[i])
        aux = {"M": M_evol, "d": d, "N_eff": self.N - d, "pattern_inf": self.pattern_inf[i],
               "true_errors": self.errors_all[i]}
        if self.qaware:
            aux["logw"] = self.logw_all[i]
        return Shot(cost=self.N * M_evol, rejected=rejected, outcome=int(self.outcomes[i]), key=M_evol, aux=aux)

    # -- inference -------------------------------------------------------------
    def _phi_eff_theorem(self, M_evol, Mf, pattern):
        if not pattern.any():
            return self.cache_phieff.get(M_evol, lambda: H.phi_eff_grid_cb(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        phi_pq = self.cache_phi_pq.get(M_evol, lambda: H.phi_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        return H.phi_eff_with_errors(phi_pq, jnp.array(pattern.astype(np.float32), dtype=jnp.float32))

    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        Mf = jnp.float32(M_evol)
        pattern = shot.aux["pattern_inf"]
        N_clean = self.N - int(pattern.sum())
        if self.theorem and not self.qaware:
            if self.fl:
                error_mask_jax = jnp.array(pattern.astype(np.float32), dtype=jnp.float32)   # cb:338-340
                phi_eff_grid = self._phi_eff_theorem(M_evol, Mf, pattern)
                p_pq = self.cache_p_pq.get(M_evol, lambda: H.p_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
                logpk = H.syndrome_update_per_qubit(logpk, p_pq, error_mask_jax)             # cb:373
            else:
                phi_eff_grid = self.cache_phieff.get(M_evol, lambda: H.phi_eff_grid_cb(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
            log_visibility = self.log_vis
            if self.marg:
                log_visibility = log_visibility - 2.0 * (M_evol ** 2) * N_clean * (self.sigma ** 2)   # cb:376, :420
            if log_visibility == 0.0:
                return H.phase_update(logpk, phi_eff_grid, shot.outcome)                     # cb:381, :425
            return H.phase_update_vis(logpk, phi_eff_grid, shot.outcome, jnp.float32(log_visibility))
        # exact model (or q-aware sum for either model)
        p_pq = self.cache_p_pq.get(M_evol, lambda: H.p_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        phi_pq = self.cache_phi_pq.get(M_evol, lambda: H.phi_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        if self.qaware:
            X = self.X_all
            logw = jnp.asarray(shot.aux["logw"].astype(np.float32))
        else:
            pb = pattern.reshape(N_BLOCKS, self.Nc)
            Xn = np.stack([pb, ~pb], axis=1).astype(np.float32)          # (B, 2, Nc): E_b and complement
            logw_np = np.zeros((N_BLOCKS, 2), np.float32)
            if self.theorem:
                logw_np[:, 1] = -np.inf                                  # theorem + q>0 mismatched: one coset member
            X = jnp.asarray(Xn)
            logw = jnp.asarray(logw_np)
        marg_coef = 2.0 * (M_evol ** 2) * (self.sigma ** 2) if self.marg else 0.0
        return H.exact_blocks_update(logpk, p_pq, phi_pq, X, logw, jnp.float32(self.log_vis),
                                     jnp.float32(marg_coef), shot.outcome)

    def account(self, shot: Shot, ctr) -> None:
        M = shot.key
        d = shot.aux["d"]
        if self.fl:
            if d > 0:
                ctr.n_syndrome_used += 1                                  # cb:316-317
            ctr.sum_d += d
            ctr.sum_F_obs += 4.0 * (M ** 2) * ((self.N - d) ** 2)          # cb:319
        else:
            ctr.sum_F_obs += 4.0 * (M ** 2) * (self.N ** 2)                # cb:398
        ctr.sum_M_sq += M ** 2

    def check_due(self, i: int, ctr) -> bool:
        return check_due(self.cfg, i, ctr)

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "N_code": self.Nc, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> Combined:
    return Combined(spec)
