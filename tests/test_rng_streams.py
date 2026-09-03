"""RNG streams: the per-target draw order every adapter uses, and its determinism.

The data are reproducible only if a (seed, epsilon) pair always consumes the same
stream in the same order, so each test rebuilds the arrays directly from
``default_rng(seed + int(eps * 1e8) % 2**31)`` and compares them with the arrays
the adapter recorded in ``draws``.

Skips until ``eqsp.protocols`` and ``eqsp.io`` exist.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

protocols = pytest.importorskip("eqsp.protocols")
eio = pytest.importorskip("eqsp.io")

from eqsp.config import RunSpec, epsilon_grid, grid_resolution  # noqa: E402

SEED = 1
EPS = float(epsilon_grid()[2])          # any grid target works; index 2 avoids the two endpoints
EPS_OTHER = float(epsilon_grid()[7])
EPS_MIN = float(epsilon_grid().min())

MAX_N_ADAPTIVE, MIN_N_ADAPTIVE = 16384, 10
N_EXP_ADAPTIVE = 10000                  # bare_adaptive max_experiments
N_EXP = 50000                           # bare_fixed, ed, combined max_experiments


def reference_rng(seed, eps):
    """The seed expression the adapters use, written out."""
    return np.random.default_rng(seed + int(eps * 1e8) % (2**31))


def build(spec, seed=SEED, eps=EPS, max_experiments=None):
    p = protocols.get_protocol(spec, "cpu")
    if max_experiments is not None:
        p.cfg = replace(p.cfg, max_experiments=max_experiments)
    m = p.grid_m(EPS_MIN)
    p.begin_run(seed, m)
    p.begin_epsilon(eps, p.rng_for(seed, eps))
    return p, m


def same(draws, key, expected):
    assert key in draws, f"adapter draws lack {key!r}; keys = {list(draws)}"
    got = np.asarray(draws[key])
    exp = np.asarray(expected)
    assert np.array_equal(got, exp.astype(got.dtype) if got.dtype != exp.dtype else exp), key


def M_max(eps, N_unit, N_code_for_nyquist, m):
    """min(floor(1/(eps N_unit)), Nyquist cap) on the prior window pi/N_code."""
    return min(max(1, int(1.0 / (eps * N_unit))), max(1, (2 ** (m - 2)) // N_code_for_nyquist))


# --------------------------------------------------------------------------
# The seed expressions
# --------------------------------------------------------------------------
def test_eps_rng_seed_expression():
    a = eio.eps_rng(SEED, EPS).random(16)   # 16 draws: two PCG64 output blocks, enough to see a wrong seed
    b = reference_rng(SEED, EPS).random(16)
    assert np.array_equal(a, b)
    # the float64 eps objects of the grid feed the seed, so every one of the 60 targets must match
    for eps in epsilon_grid():
        assert eio.eps_rng(SEED, eps).integers(0, 2**31) == reference_rng(SEED, eps).integers(0, 2**31)


def test_eps_key_separates_the_targets():
    keys = [eio.eps_key(float(e)) for e in epsilon_grid()]
    assert len(set(keys)) == len(keys)
    assert max(keys) < 2**31                 # int(eps*1e8) <= 1e7 on this grid: the modulo never wraps


def test_seed_rng_is_default_rng_seed():
    assert np.array_equal(eio.seed_rng(SEED).normal(0.0, 0.01, 9), np.random.default_rng(SEED).normal(0.0, 0.01, 9))


def test_stream_rng_depends_on_every_argument():
    base = eio.stream_rng(SEED, EPS, 7).random(8)
    assert not np.array_equal(base, eio.stream_rng(SEED + 1, EPS, 7).random(8))
    assert not np.array_equal(base, eio.stream_rng(SEED, EPS_OTHER, 7).random(8))
    assert not np.array_equal(base, eio.stream_rng(SEED, EPS, 8).random(8))


# --------------------------------------------------------------------------
# Per-protocol draw order
# --------------------------------------------------------------------------
def test_bare_adaptive_draw_order():
    spec = RunSpec(protocol="bare_adaptive", gamma=0.0)
    p, _ = build(spec)
    N = max(MIN_N_ADAPTIVE, min(int(1.0 / EPS), MAX_N_ADAPTIVE))
    rng = reference_rng(SEED, EPS)
    same(p.draws, "ns", rng.integers(1, N + 1, N_EXP_ADAPTIVE))
    same(p.draws, "uniforms", rng.random(N_EXP_ADAPTIVE))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP_ADAPTIVE))
    assert list(p.draws) == ["ns", "uniforms", "thetas"]


def test_bare_adaptive_readout_uniforms_precede_the_phases():
    spec = RunSpec(protocol="bare_adaptive", gamma=0.0, q=1e-2, estimator="aware")
    p, _ = build(spec)
    N = max(MIN_N_ADAPTIVE, min(int(1.0 / EPS), MAX_N_ADAPTIVE))
    rng = reference_rng(SEED, EPS)
    same(p.draws, "ns", rng.integers(1, N + 1, N_EXP_ADAPTIVE))
    same(p.draws, "uniforms", rng.random(N_EXP_ADAPTIVE))
    same(p.draws, "uniforms_readout_parity", rng.random(N_EXP_ADAPTIVE))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP_ADAPTIVE))
    assert list(p.draws) == ["ns", "uniforms", "uniforms_readout_parity", "thetas"]


def test_bare_fixed_draw_order():
    L, gamma = 1, 0.05
    N_code = 2 * L + 1
    p, m = build(RunSpec(protocol="bare_fixed", gamma=gamma, L=L))
    assert m == grid_resolution(N_code, EPS_MIN)
    rng = reference_rng(SEED, EPS)
    same(p.draws, "M_evols", rng.integers(1, M_max(EPS, 1, N_code, m) + 1, N_EXP))
    same(p.draws, "uniforms_outcome", rng.random(N_EXP))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP))
    assert list(p.draws) == ["M_evols", "uniforms_outcome", "thetas"]


@pytest.mark.parametrize("inference", ["fl", "ps"])
def test_ed_draw_order(inference):
    L, gamma = 1, 0.01
    N_code = 2 * L + 1
    p, m = build(RunSpec(protocol="ed", gamma=gamma, L=L, inference=inference))
    assert m == grid_resolution(N_code, EPS_MIN)
    rng = reference_rng(SEED, EPS)
    same(p.draws, "M_evols", rng.integers(1, M_max(EPS, N_code, N_code, m) + 1, N_EXP))
    same(p.draws, "uniforms_outcome", rng.random(N_EXP))
    same(p.draws, "uniforms_syndrome", rng.random((N_EXP, N_code)))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP))
    assert list(p.draws) == ["M_evols", "uniforms_outcome", "uniforms_syndrome", "thetas"]


def test_ed_readout_uniforms_precede_the_phases():
    L, gamma, q = 1, 0.01, 1e-2
    N_code = 2 * L + 1
    p, m = build(RunSpec(protocol="ed", gamma=gamma, L=L, inference="fl", q=q, estimator="aware"))
    rng = reference_rng(SEED, EPS)
    rng.integers(1, M_max(EPS, N_code, N_code, m) + 1, N_EXP)
    rng.random(N_EXP)
    rng.random((N_EXP, N_code))
    same(p.draws, "uniforms_readout_syndrome", rng.random((N_EXP, N_code - 1)))
    same(p.draws, "uniforms_readout_parity", rng.random(N_EXP))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP))
    assert list(p.draws) == ["M_evols", "uniforms_outcome", "uniforms_syndrome",
                             "uniforms_readout_syndrome", "uniforms_readout_parity", "thetas"]


@pytest.mark.parametrize("inference,marg", [("fl", False), ("ps", False), ("fl", True)])
@pytest.mark.parametrize("sigma", [0.01, 0.0])
def test_combined_draw_order_and_quenched_offsets(inference, marg, sigma):
    if marg and sigma == 0.0:
        pytest.skip("marg with sigma = 0 is the quenched run")
    L, gamma = 1, 0.01
    N_total = 3 * (2 * L + 1)
    p, m = build(RunSpec(protocol="combined", gamma=gamma, L=L, sigma=sigma, inference=inference, marg=marg))
    assert m == grid_resolution(N_total, EPS_MIN)
    # quenched offsets: default_rng(seed).normal(0, sigma, N_total), once per seed
    expected_offsets = np.random.default_rng(SEED).normal(0.0, sigma, N_total) if sigma > 0 else np.zeros(N_total)
    assert np.array_equal(p.epsilons_np, expected_offsets)
    rng = reference_rng(SEED, EPS)
    same(p.draws, "M_evols", rng.integers(1, M_max(EPS, N_total, N_total, m) + 1, N_EXP))
    same(p.draws, "uniforms_outcome", rng.random(N_EXP))
    same(p.draws, "uniforms_syndrome", rng.random((N_EXP, N_total)))
    order = ["M_evols", "uniforms_outcome", "uniforms_syndrome"]
    if marg:
        same(p.draws, "epsilons_per_shot", rng.normal(0.0, sigma, (N_EXP, N_total)))
        order.append("epsilons_per_shot")
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, N_EXP))
    assert list(p.draws) == order + ["thetas"]


def test_hwindow_stream_is_keyed_on_the_probe_size():
    sigma, eps_tar, n_exp = 0.05, 1e-4, 512
    spec = RunSpec(protocol="hwindow", sigma=sigma, eps_tar=eps_tar, N_fixed=100)
    p, _ = build(spec, eps=eps_tar, max_experiments=n_exp)
    rng = eio.stream_rng(SEED, eps_tar, 100)
    same(p.draws, "S", rng.normal(0.0, sigma * np.sqrt(100), n_exp))
    same(p.draws, "thetas", rng.uniform(0.0, 2.0 * np.pi, n_exp))
    same(p.draws, "uniforms", rng.random(n_exp))
    assert list(p.draws) == ["S", "thetas", "uniforms"]
    # a different probe size is a different stream at the same (seed, eps_tar)
    other, _ = build(replace(spec, N_fixed=200), eps=eps_tar, max_experiments=n_exp)
    assert not np.array_equal(np.asarray(other.draws["uniforms"]), np.asarray(p.draws["uniforms"]))


def test_sequential_round_stream():
    """The first round draws uniforms of shape (K, N) with K = 30 from the per-target stream."""
    L, sigma = 1, 0.01
    N = 2 * L + 1
    p = protocols.get_protocol(RunSpec(protocol="sequential", L=L, sigma=sigma), "cpu")
    p.run_seed(SEED, np.array([EPS]), "test", "none")
    first = np.asarray(p.draws["uniforms"][0])
    assert first.shape == (30, N)
    assert np.array_equal(first, reference_rng(SEED, EPS).random((30, N)))


# --------------------------------------------------------------------------
# Determinism across fresh runs, and separation across targets
# --------------------------------------------------------------------------
SPECS = {
    "bare_adaptive": RunSpec(protocol="bare_adaptive", gamma=0.05),
    "bare_fixed": RunSpec(protocol="bare_fixed", gamma=0.05, L=1),
    "ed_fl": RunSpec(protocol="ed", gamma=0.01, L=1, inference="fl"),
    "ed_ps": RunSpec(protocol="ed", gamma=0.01, L=1, inference="ps"),
    "combined_fl": RunSpec(protocol="combined", gamma=0.01, L=1, sigma=0.01, inference="fl"),
    "combined_marg": RunSpec(protocol="combined", gamma=0.01, L=1, sigma=0.01, inference="fl", marg=True),
}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_same_seed_and_eps_replay_the_same_draws(name):
    a, _ = build(SPECS[name])
    b, _ = build(SPECS[name])
    assert list(a.draws) == list(b.draws)
    for key in a.draws:
        assert np.array_equal(np.asarray(a.draws[key]), np.asarray(b.draws[key])), key


@pytest.mark.parametrize("name", sorted(SPECS))
def test_different_eps_is_a_different_stream(name):
    a, _ = build(SPECS[name], eps=EPS)
    b, _ = build(SPECS[name], eps=EPS_OTHER)
    key = "uniforms" if "uniforms" in a.draws else "uniforms_outcome"
    assert not np.array_equal(np.asarray(a.draws[key]), np.asarray(b.draws[key]))


@pytest.mark.parametrize("name", sorted(SPECS))
def test_different_seed_is_a_different_stream(name):
    a, _ = build(SPECS[name], seed=SEED)
    b, _ = build(SPECS[name], seed=SEED + 1)
    key = "uniforms" if "uniforms" in a.draws else "uniforms_outcome"
    assert not np.array_equal(np.asarray(a.draws[key]), np.asarray(b.draws[key]))
