"""Heisenberg-window sweep: bare GHZ at fixed N, M = 1, gamma = 0, shot-to-shot
offsets redrawn per shot, single target eps_tar (plan, Per-protocol mapping).

Truth: P(+) = (1 + cos(2 (N omega + S_i) - theta_i))/2 with S_i ~ N(0, sigma^2 N)
the realized offset sum and theta_i ~ U[0, 2pi) a per-shot readout phase.
Inference: (1 + lambda_N cos(2 N omega - theta_i))/2, lambda_N = e^{-2 N sigma^2}
(sensing.tex, Heisenberg-window section).  Grid on one period [0, pi/N) with the
truth wrapped and period-pi/N circular error; RNG stream_rng(seed, eps_tar, N)
with draw order S, thetas, uniforms.  The device-side loop checks the stopping
rule after every update (T(N) resolution; a coarser cadence would quantise T
near the resource minimum at N*/2).
"""
from __future__ import annotations

import math

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .. import closed_forms as cf
from .. import hamiltonian as H
from ..config import OMEGA_TRUE, EngineConfig, RunSpec
from ..engine import Shot, make_grid, posterior_stats
from ..io import stream_rng

GRID_M = 16
CHUNK = 2048      # updates per device-side while_loop call (one host sync per chunk)


@partial(jax.jit, static_argnames=("N", "period", "chunk"))
def run_chunk_kernel(logpk, xs, N, log_lambda, thetas, outcomes, i0, omega_true, period, stop_std, chunk):
    """Up to ``chunk`` hwindow updates on device starting at shot i0, stopping at the first
    update whose posterior circular std <= stop_std.  Same hwindow_update and
    posterior_stats as the host loop; returns (logpk, n_done, err, std)."""
    def cond(state):
        _, n, _, std = state
        return (n < chunk) & (std > stop_std)

    def body(state):
        logpk, n, _, _ = state
        logpk = H.hwindow_update(logpk, xs, N, log_lambda, thetas[i0 + n], outcomes[i0 + n])
        err, std = posterior_stats(logpk, xs, omega_true, period=period)
        return logpk, n + 1, err, std

    init = (logpk, jnp.int32(0), jnp.float32(jnp.inf), jnp.float32(jnp.inf))
    return lax.while_loop(cond, body, init)


class HWindow:
    grid_builder = "np64"

    def __init__(self, spec: RunSpec):
        if spec.N_fixed is None or spec.eps_tar is None:
            raise ValueError("hwindow requires N_fixed and eps_tar")
        if spec.gamma != 0.0:
            raise ValueError("hwindow is defined at gamma = 0")
        self.spec = spec
        self.N = int(spec.N_fixed)
        self.sigma = float(spec.sigma)
        self.period = float(np.pi / self.N)
        self.cfg = EngineConfig(max_experiments=2_000_000)
        # The prior is one period pi/N wide and the error is period-pi/N circular, so a
        # target with 1.2 eps_tar comparable to pi/(2N) converges trivially.  Require a
        # factor 20 between period and tolerance: the suite's eps_tar = 2e-2/N_* (suite.py WINDOW_EPS) gives
        # pi/N / (1.2 eps_tar) = 33 at the largest N = 4 N_*, so every suite point passes; the chupacabras quick
        # run of 2026-09-02 at eps_tar = 1e-3, N = 800 (ratio 3.3) showed 100% "convergence" after ~1 shot.
        if math.pi / spec.N_fixed < 20.0 * self.cfg.tolerance * spec.eps_tar:
            raise ValueError(f"hwindow: period pi/N = {math.pi/spec.N_fixed:.3g} is not >> tolerance "
                             f"{self.cfg.tolerance*spec.eps_tar:.3g} at N = {spec.N_fixed}; lower eps_tar")
        self.omega_true = OMEGA_TRUE % self.period
        self.V = cf.prep_visibility(spec.k, spec.F, self.N)
        self.log_lambda = -2.0 * self.N * self.sigma**2
        if spec.estimator == "aware":
            self.log_lambda += self.N * math.log1p(-2.0 * spec.q) + math.log(self.V)
        self.draws: dict = {}

    def grid_m(self, eps_min: float) -> int:
        return GRID_M

    def rng_for(self, seed: int, eps: float):
        return stream_rng(seed, eps, self.N)

    def begin_run(self, seed: int, m: int) -> None:
        self.m = m
        self.xs = make_grid(m, self.period, "np64")

    def begin_epsilon(self, eps: float, rng) -> None:
        spec = self.spec
        n_exp = self.cfg.max_experiments
        N = self.N
        S = rng.normal(0.0, self.sigma * math.sqrt(N), n_exp)
        thetas = rng.uniform(0.0, 2.0 * np.pi, n_exp)
        uniforms = rng.random(n_exp)
        self.draws = {"S": S, "thetas": thetas, "uniforms": uniforms}
        p_plus = 0.5 * (1.0 + self.V * np.cos(2.0 * (N * OMEGA_TRUE + S) - thetas))
        outcomes = np.where(uniforms < p_plus, 0, 1)
        if spec.q > 0:
            u_r = rng.random(n_exp)
            self.draws["uniforms_readout_parity"] = u_r
            outcomes = outcomes ^ (u_r < H.parity_flip_prob(spec.q, N)).astype(int)
        self.thetas = thetas
        self.outcomes = outcomes
        self.thetas_jax = jnp.asarray(thetas, dtype=jnp.float32)
        self.outcomes_jax = jnp.asarray(outcomes, dtype=jnp.int32)

    def draw(self, i: int) -> Shot:
        return Shot(cost=self.N, rejected=False, outcome=int(self.outcomes[i]), key=1,
                    aux={"theta": float(self.thetas[i])})

    def apply(self, logpk, shot: Shot):
        return H.hwindow_update(logpk, self.xs, self.N, jnp.float32(self.log_lambda),
                                jnp.float32(shot.aux["theta"]), shot.outcome)

    def account(self, shot: Shot, ctr) -> None:
        ctr.sum_M_sq += 1.0

    # --- device-side chunked loop (the engine uses it whenever an adapter defines it) ---
    chunk = CHUNK

    def run_chunk(self, logpk, i0: int, B: int, omega_true, stop_std: float):
        return run_chunk_kernel(logpk, self.xs, self.N, jnp.float32(self.log_lambda), self.thetas_jax,
                                self.outcomes_jax, jnp.int32(i0), omega_true, self.period, jnp.float32(stop_std), B)

    def account_chunk(self, ctr, n_done: int) -> None:
        ctr.resources += self.N * n_done
        ctr.n_rounds += n_done
        ctr.n_updates += n_done
        ctr.n_accepted += n_done
        ctr.sum_M_sq += float(n_done)
        ctr.M_exec_max = max(ctr.M_exec_max, 1)

    def row(self, ctr, converged: bool, error: float) -> dict:
        return {"N": self.N, "M_max": 1, "grid_m": self.m, "visibility": math.exp(-2.0 * self.N * self.sigma**2)}


def make(spec: RunSpec) -> HWindow:
    return HWindow(spec)
