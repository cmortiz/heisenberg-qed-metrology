"""Shared grid-Bayesian engine: grid, posterior, MAP error, caches, the per-eps loop.

One loop for every Bayesian protocol.  Adapters supply the truth side (``draw``)
and the jitted inference update (``apply``); the engine owns the posterior, the
resource accounting and the stopping rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.special import logsumexp

from .config import EngineConfig, RunSpec
from .io import make_row

jax.config.update("jax_enable_x64", False)   # every kernel is float32


# --------------------------------------------------------------------------
# Grid and posterior
# --------------------------------------------------------------------------
def make_grid(m: int, period: float, builder: Literal["jax32", "np64"], offset: float = 0.0) -> jnp.ndarray:
    """2^m float32 points on [offset, offset + period).

    jax32: jnp.linspace(..., dtype=float32)          (bare_adaptive, period pi)
    np64:  float64 np.linspace then cast to float32  (bare_fixed, ed, combined, hwindow)
    The two round differently in the last bit.
    """
    grid_size = 2**m
    if builder == "jax32":
        return jnp.linspace(offset, offset + period, grid_size, endpoint=False, dtype=jnp.float32)
    if builder == "np64":
        xs_np = np.linspace(offset, offset + period, grid_size, endpoint=False, dtype=np.float64)
        return jnp.array(xs_np, dtype=jnp.float32)
    raise ValueError(f"unknown grid builder {builder!r}")


def init_log_posterior(m: int) -> jnp.ndarray:
    """Uniform log prior on 2^m points."""
    return jnp.full(2**m, -m * jnp.log(2.0), dtype=jnp.float32)


@jit
def log_update(logpk, log_lik):
    """Add a log-likelihood and renormalise (the physics kernels fuse these two lines)."""
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


@partial(jit, static_argnames=("period",))
def posterior_stats(logpk, xs, omega_true, period: float):
    """(MAP circular error, posterior circular std) for the truth-free stopping rule.

    The MAP estimate is xs[argmax] and the error its circular distance to
    omega_true on [0, period); ``period`` is static so the constant is baked in.
    The std is the root of the posterior central second moment (variance) of the
    wrapped displacement from the MAP, d_j = wrap(x_j - x_MAP) in (-period/2, period/2];
    computing it from displacements (not from the resultant length 1 - R, which
    float32 cannot resolve below std ~ 1e-3 period) keeps float32 accurate down
    to std ~ 1e-6 period."""
    idx = jnp.argmax(logpk)
    estimate = xs[idx]
    diff = jnp.abs(omega_true - estimate)
    err = jnp.minimum(diff, period - diff)
    w = jnp.exp(logpk - logsumexp(logpk))
    d = xs - estimate
    d = d - period * jnp.round(d / period)
    m1 = jnp.sum(w * d)
    m2 = jnp.sum(w * d * d)
    return err, jnp.sqrt(jnp.maximum(m2 - m1 * m1, 0.0))


class GridCache:
    """Per-key grid arrays with clear-when-full eviction.  max entries =
    cache_mb / (grid_size * 4 bytes), divided by ``entry_width`` grid-sized
    arrays per entry."""

    def __init__(self, grid_size: int, cache_mb: int = 2048, entry_width: int = 1):
        base = (cache_mb * 1024 * 1024) // (grid_size * 4)
        self.max_entries = base if entry_width == 1 else max(1, base // entry_width)
        self._d: dict = {}

    def get(self, key, build):
        if key not in self._d:
            if len(self._d) >= self.max_entries:
                self._d.clear()
            self._d[key] = build()
        return self._d[key]


# --------------------------------------------------------------------------
# Loop state and adapter contract
# --------------------------------------------------------------------------
@dataclass
class Counters:
    resources: int = 0
    n_rounds: int = 0
    n_updates: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_syndrome_used: int = 0
    sum_d: int = 0
    sum_F_obs: float = 0.0
    sum_M_sq: float = 0.0
    M_exec_max: int = 0
    posterior_std: float = math.nan


@dataclass
class EpsResult:
    converged: bool
    error: float
    ctr: Counters
    extra: dict = field(default_factory=dict)


class Shot(NamedTuple):
    cost: int            # physical resources charged (also when the shot is rejected: it was still run)
    rejected: bool
    outcome: int         # parity outcome, 0 = +1
    key: int             # n (adaptive) or M (fixed-N); cache key
    aux: dict            # d, N_eff, error_mask, true_errors, syndrome_obs, ...


class BayesianProtocol(Protocol):
    """Adapter contract.  Attributes ``xs`` (grid), ``m`` (grid exponent) and
    ``omega_true`` (truth on the grid's period) are set by ``begin_run``;
    ``draws`` (dict of every per-eps RNG array, in draw order) by
    ``begin_epsilon``."""
    cfg: EngineConfig
    period: float
    grid_builder: Literal["jax32", "np64"]
    spec: RunSpec
    xs: jnp.ndarray
    m: int
    omega_true: float
    draws: dict

    def grid_m(self, eps_min: float) -> int: ...
    def rng_for(self, seed: int, eps: float) -> np.random.Generator: ...
    def begin_run(self, seed: int, m: int) -> None: ...
    def begin_epsilon(self, eps: float, rng: np.random.Generator) -> None: ...
    def draw(self, i: int) -> Shot: ...
    def apply(self, logpk: jnp.ndarray, shot: Shot) -> jnp.ndarray: ...
    def account(self, shot: Shot, ctr: Counters) -> None: ...
    def row(self, ctr: Counters, converged: bool, error: float) -> dict: ...


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def _run_epsilon_device(p, logpk, ctr: Counters, omega, threshold: float, stop_std: float) -> EpsResult:
    """Loop for protocols that expose ``run_chunk``: the update + stopping test runs
    in a device-side while_loop over chunks of ``p.chunk`` pre-drawn shots (one host sync per chunk
    instead of per update).  Same semantics as the host loop: stop at the first update with
    2 std <= eps, converged iff the MAP error there is < threshold; exhausting the budget is never
    converged."""
    cfg = p.cfg
    i = 0
    err = std = math.nan
    while i < cfg.max_experiments:
        B = min(p.chunk, cfg.max_experiments - i)
        logpk, n_done, e, s = p.run_chunk(logpk, i, B, omega, stop_std)
        n_done, err, std = int(n_done), float(e), float(s)
        i += n_done
        p.account_chunk(ctr, n_done)
        ctr.posterior_std = std
        if std <= stop_std:
            conv = err < threshold
            return EpsResult(conv, err, ctr, p.row(ctr, conv, err))
    return EpsResult(False, err, ctr, p.row(ctr, False, err))


def run_epsilon(p: BayesianProtocol, eps: float, rng: np.random.Generator) -> EpsResult:
    cfg = p.cfg
    p.begin_epsilon(eps, rng)
    logpk = init_log_posterior(p.m)
    ctr = Counters()
    omega = jnp.float32(p.omega_true)
    threshold = cfg.tolerance * eps
    stop_std = 0.5 * eps                          # 95% credible half-width (2 std) <= eps
    if hasattr(p, "run_chunk"):   # hwindow implements the device-side loop; the host loop below is the contract every other adapter fulfils
        return _run_epsilon_device(p, logpk, ctr, omega, threshold, stop_std)
    for i in range(cfg.max_experiments):
        shot = p.draw(i)
        ctr.resources += shot.cost
        ctr.n_rounds = i + 1
        if shot.key > ctr.M_exec_max:                 # executed depth, counted before the rejection test: a rejected shot was still run (cost charged at ed:224)
            ctr.M_exec_max = shot.key
        if shot.rejected:
            ctr.n_rejected += 1
            continue                                  # rejected: charged, but no update and no stopping test
        logpk = p.apply(logpk, shot)
        ctr.n_updates += 1
        ctr.n_accepted += 1
        p.account(shot, ctr)
        e, s = posterior_stats(logpk, p.xs, omega, period=p.period)
        err, std = float(e), float(s)
        ctr.posterior_std = std
        if std <= stop_std:
            conv = err < threshold
            return EpsResult(conv, err, ctr, p.row(ctr, conv, err))
    # Exhausted max_experiments: never converged.
    e, s = posterior_stats(logpk, p.xs, omega, period=p.period)
    err, ctr.posterior_std = float(e), float(s)
    return EpsResult(False, err, ctr, p.row(ctr, False, err))


def run_seed(p: BayesianProtocol, spec: RunSpec, seed: int, epsilons, engine_version: str,
             jax_version: str, progress: bool = False) -> list[dict]:
    epsilons = np.asarray(epsilons, dtype=np.float64)
    m = p.grid_m(float(np.min(epsilons)))
    p.begin_run(seed, m)
    rows = []
    it = epsilons
    if progress:
        from tqdm import tqdm
        it = tqdm(epsilons, desc=f"  {spec.protocol} seed={seed}", leave=False)
    for eps in it:
        eps = float(eps)
        res = run_epsilon(p, eps, p.rng_for(seed, eps))
        rows.append(make_row(spec, seed, eps, res, engine_version, jax_version))
    return rows


def run_seeds(p: BayesianProtocol, spec: RunSpec, seeds, epsilons, engine_version: str,
              jax_version: str, progress: bool = False, on_seed_done=None) -> list[dict]:
    """Loop seeds in one process (amortises JIT); ``begin_run`` re-derives
    seed_rng(seed) per seed.  ``on_seed_done(seed, rows)`` lets the caller write
    one shard per seed."""
    rows_all = []
    for seed in seeds:
        rows = run_seed(p, spec, int(seed), epsilons, engine_version, jax_version, progress)
        if on_seed_done is not None:
            on_seed_done(int(seed), rows)
        rows_all.extend(rows)
    return rows_all
