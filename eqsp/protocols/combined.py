"""Combined protocol: logical GHZ over three [[N_code, 1]] repetition-code blocks
with per-qubit offsets omega_k = omega + eps_k.

Quenched offsets eps_k ~ N(0, sigma^2) from seed_rng(seed) once per seed; marg
mode redraws them per shot on the truth side and infers with eps = 0 and a
per-pattern visibility damping exp(-2 M^2 N_clean sigma^2).
M_max = min(floor(1/(eps N_total)), Nyquist cap), grid on one M = 1 fringe period
pi/N_total centred on omega.  Draw order: M_evols, uniforms_outcome,
uniforms_syndrome(50000, N_total), epsilons_per_shot (marg only), readout
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
from ..io import eps_rng, seed_rng

N_BLOCKS = 3


class Combined:
    grid_builder = "np64"

    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.fl = spec.inference == "fl"
        self.marg = bool(spec.marg)
        self.marg_cap = bool(spec.marg_cap)
        if self.marg and not self.fl:
            raise ValueError("combined: marg requires inference=fl (mode combined_full_likelihood_marg)")
        if self.marg_cap and not (self.marg and spec.sigma > 0):
            raise ValueError("combined: marg_cap requires marg with sigma > 0")
        self.Nc = spec.N_code
        self.N = N_BLOCKS * self.Nc
        self.M_cap = max(1, int(math.ceil(1.0 / (2.0 * spec.sigma * math.sqrt(self.N))))) if self.marg_cap else 0
        self.period = float(np.pi) / self.N                               # one M = 1 fringe period of the N_total parity
        self.grid_offset = OMEGA_TRUE - 0.5 * self.period
        self.L = spec.L
        self.sigma = float(spec.sigma)
        self.cfg = EngineConfig(max_experiments=50000)
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
        self.xs = make_grid(m, self.period, "np64", offset=self.grid_offset)
        self.omega_grid = self.xs
        rng_seed = seed_rng(seed)
        if self.sigma > 0:
            self.epsilons_np = rng_seed.normal(0.0, self.sigma, self.N)
        else:
            self.epsilons_np = np.zeros(self.N)
        self.omegas_np = self.omega_true + self.epsilons_np
        # inference offsets: zero in marg mode
        eps_inf = np.zeros(self.N) if self.marg else self.epsilons_np
        self.epsilons_jax = jnp.array(eps_inf, dtype=jnp.float32)
        self._new_caches()

    def _new_caches(self):
        G = 2**self.m
        self.cache_phi_pq = GridCache(G, self.cfg.cache_mb, entry_width=self.N)  # phi_k grids per depth
        self.cache_p_pq = GridCache(G, self.cfg.cache_mb, entry_width=self.N)    # p_k grids per depth

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        N, Nc = self.N, self.Nc
        M_max = max(1, int(1.0 / (eps * N)))
        nyquist_limit = H.nyquist_cap(self.m, N, self.period)
        M_max = min(M_max, max(1, nyquist_limit))
        if self.marg_cap:
            # Information-optimal depth under marginalised offsets: 4 M^2 N^2 exp(-4 M^2 N sigma^2)
            # peaks at M = 1/(2 sigma sqrt(N)) (Corollary cor:heisenberg-threshold read at depth M).
            # Without this cap the uniform draw on {1..M_max} spends an eps-shrinking fraction of
            # its shots at zero visibility.
            M_max = min(M_max, self.M_cap)
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
        if self.marg:
            epsilons_per_shot = rng.normal(0.0, self.sigma, (n_exp, N))
            self.draws["epsilons_per_shot"] = epsilons_per_shot
            omegas = self.omega_true + epsilons_per_shot                 # (S, N)
        else:
            omegas = self.omegas_np                                      # (N,)
        p_all, phi_all = H.exact_params(omegas, gammas_np, M_evols)
        errors_all = uniforms_syndrome[:, :N] < p_all
        d_blocks = errors_all.reshape(n_exp, N_BLOCKS, Nc).sum(axis=2)
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
        # per-shot preparation phase, drawn after every outcome and readout draw
        self.thetas = rng.uniform(0.0, 2.0 * np.pi, n_exp)
        self.draws["thetas"] = self.thetas

        p_plus = H.exact_p_plus(p_all, phi_all, errors_all, N_BLOCKS, V=self.V, theta=self.thetas)
        outcomes = np.where(uniforms_outcome < p_plus, 0, 1)
        if spec.q > 0:
            outcomes = outcomes ^ (u_par < H.parity_flip_prob(spec.q, N)).astype(int)
        self.outcomes = outcomes
        self.errors_all = errors_all
        self.d_total = d_total
        self.pattern_inf = pattern_obs                        # inference sees only the decoded observed syndrome
        self.reject = d_obs_blocks.sum(axis=1) > 0            # rejection (ps): any non-trivial observed syndrome
        if self.qaware:
            self.logw_all = np.stack([H.pattern_log_weights(syn_obs[:, b, :], spec.q, Nc)
                                      for b in range(N_BLOCKS)], axis=1)   # (S, B, P)

    def draw(self, i: int) -> Shot:
        M_evol = int(self.M_evols[i])
        d = int(self.d_total[i])
        rejected = bool((not self.fl) and self.reject[i])
        aux = {"M": M_evol, "d": d, "N_eff": self.N - d, "pattern_inf": self.pattern_inf[i],
               "true_errors": self.errors_all[i], "theta": float(self.thetas[i])}
        if self.qaware:
            aux["logw"] = self.logw_all[i]
        return Shot(cost=self.N * M_evol, rejected=rejected, outcome=int(self.outcomes[i]), key=M_evol, aux=aux)

    # -- inference -------------------------------------------------------------
    def apply(self, logpk, shot: Shot):
        M_evol = shot.key
        Mf = jnp.float32(M_evol)
        pattern = shot.aux["pattern_inf"]
        p_pq = self.cache_p_pq.get(M_evol, lambda: H.p_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        phi_pq = self.cache_phi_pq.get(M_evol, lambda: H.phi_grid_per_qubit(self.omega_grid, self.gammas_jax, self.epsilons_jax, Mf))
        if self.qaware:
            X = self.X_all
            logw = jnp.asarray(shot.aux["logw"].astype(np.float32))
        else:
            pb = pattern.reshape(N_BLOCKS, self.Nc)
            Xn = np.stack([pb, ~pb], axis=1).astype(np.float32)          # (B, 2, Nc): E_b and complement
            logw_np = np.zeros((N_BLOCKS, 2), np.float32)                # both coset members, equal weight
            X = jnp.asarray(Xn)
            logw = jnp.asarray(logw_np)
        marg_coef = 2.0 * (M_evol ** 2) * (self.sigma ** 2) if self.marg else 0.0
        return H.exact_blocks_update(logpk, p_pq, phi_pq, X, logw, jnp.float32(self.log_vis),
                                     jnp.float32(marg_coef), shot.outcome, jnp.float32(shot.aux["theta"]))

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
        return {"N": self.N, "N_code": self.Nc, "N_total": self.N, "M_max": self.M_max, "grid_m": self.m,
                "visibility": self.V * cf.readout_visibility(self.spec.q, self.N)}


def make(spec: RunSpec) -> Combined:
    return Combined(spec)
