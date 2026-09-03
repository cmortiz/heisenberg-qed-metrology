"""RNG replay: each adapter's ``begin_epsilon`` must draw exactly the arrays the
legacy script drew, in the legacy order, from the per-epsilon stream
``default_rng(seed + int(eps * 1e8) % 2**31)``.

Legacy sequences replayed here (file:line as of tag v1-jax-suite):
  simulation.py:161-162, :239               bare_adaptive (mode ghz, h = 0)
  simulation_bare_hamiltonian.py:149-177,:235   bare_fixed
  simulation_error_detected.py:166-182, :363    ed
  simulation_combined.py:213-243, :465-469, :507   combined (+ quenched offsets, + marg per-shot offsets)
  simulation_sequential.py:59-70, :141-145      sequential

Draw-order facts recorded while reading the scripts (reported in the final
summary as well):
  * every per-eps stream draws the heterogeneous gammas first when h > 0
    (normal, then clip: (0, 0.99) in sim, (0, 10) in bh/ed/cb);
  * bare_adaptive then draws ns = integers(1, N+1, 10000), uniforms = random(10000);
  * bh/ed/cb draw M_evols = integers(1, M_max+1, 50000), uniforms_outcome = random(50000),
    then (ed/cb) uniforms_syndrome = random((50000, N)); cb in marg mode
    (legacy --marginalize-noise, which produced combined_full_likelihood_marg.csv)
    additionally draws epsilons_per_shot = normal(0, sigma, (50000, N_total));
  * cb and sq draw the quenched offsets from default_rng(seed).normal(0, sigma, N)
    once per seed (zeros when sigma == 0, no draw);
  * the seed expression int(eps*1e8) % 2**31 never wraps on the standard grid
    (int(eps*1e8) <= 1e7), so operator precedence cannot change any seed there.

Skips until ``eqsp.protocols`` and ``eqsp.io`` exist.
"""
from __future__ import annotations

import numpy as np
import pytest

protocols = pytest.importorskip("eqsp.protocols")
eio = pytest.importorskip("eqsp.io")

from eqsp.config import RunSpec, epsilon_grid, grid_resolution  # noqa: E402

SEED = 1
EPS = float(epsilon_grid()[2])          # any grid target works; index 2 avoids the two endpoints, which both extended grids share
EPS_MIN = float(epsilon_grid().min())

# legacy constants
SIM_MAX_EXPERIMENTS = 10000       # simulation.py:35
SIM_MAX_N, SIM_MIN_N = 16384, 10  # simulation.py:33-34
MAX_EXPERIMENTS = 50000           # bh:45, ed:37, cb:39
MIN_M_EVOL = 1                    # bh:44, ed:36, cb:38


def legacy_rng(seed, eps):
    return np.random.default_rng(seed + int(eps * 1e8) % (2**31))


def build(spec, seed=SEED, eps=EPS):
    p = protocols.get_protocol(spec, "cpu")
    m = p.grid_m(EPS_MIN)
    p.begin_run(seed, m)
    p.begin_epsilon(eps, eio.eps_rng(seed, eps))
    return p, m


def same(draws, key, expected):
    assert key in draws, f"adapter draws lack legacy key {key!r}; keys = {list(draws)}"
    got = np.asarray(draws[key])
    exp = np.asarray(expected)
    if got.dtype != exp.dtype:
        # sim casts uniforms to float32 after drawing (sim:162); accept either
        assert np.array_equal(got, exp.astype(got.dtype)), key
    else:
        assert np.array_equal(got, exp), key


def legacy_M_max(eps, N_unit, N_code_for_nyquist, m):
    M_max = max(1, int(1.0 / (eps * N_unit)))
    nyquist_limit = (2 ** (m - 2)) // N_code_for_nyquist
    return min(M_max, max(1, nyquist_limit))


# --------------------------------------------------------------------------
def test_eps_rng_is_legacy_seed_expression():
    a = eio.eps_rng(SEED, EPS).random(16)   # 16 draws: two PCG64 output blocks, enough to see a wrong seed
    b = legacy_rng(SEED, EPS).random(16)
    assert np.array_equal(a, b)
    # the float64 eps objects of the grid are the legacy ones, so seeds match for all 60
    for eps in epsilon_grid():
        assert eio.eps_rng(SEED, eps).integers(0, 2**31) == legacy_rng(SEED, eps).integers(0, 2**31)


def test_seed_rng_is_default_rng_seed():
    assert np.array_equal(eio.seed_rng(SEED).normal(0.0, 0.01, 9), np.random.default_rng(SEED).normal(0.0, 0.01, 9))


# --------------------------------------------------------------------------
def test_bare_adaptive_draw_order():
    spec = RunSpec(protocol="bare_adaptive", gamma=0.0, model="theorem")
    p, m = build(spec)
    N = max(SIM_MIN_N, min(int(1.0 / EPS), SIM_MAX_N))
    rng = legacy_rng(SEED, EPS)
    ns = rng.integers(1, N + 1, SIM_MAX_EXPERIMENTS)              # sim:161
    uniforms = rng.random(SIM_MAX_EXPERIMENTS)                    # sim:162
    same(p.draws, "ns", ns)
    same(p.draws, "uniforms", uniforms)
    assert list(p.draws)[:2] == ["ns", "uniforms"]


@pytest.mark.parametrize("h", [0.0, 0.1])
def test_bare_fixed_draw_order(h):
    L, gamma = 1, 0.05
    N_code = 2 * L + 1
    spec = RunSpec(protocol="bare_fixed", gamma=gamma, h=h, L=L, model="theorem")
    p, m = build(spec)
    assert m == grid_resolution(N_code, EPS_MIN)
    M_max = legacy_M_max(EPS, 1, N_code, m)                       # bh:145-147
    rng = legacy_rng(SEED, EPS)
    keys = []
    if h > 0:
        gammas_np = np.clip(rng.normal(gamma, gamma * h, N_code), 0.0, 10.0)   # bh:150-151
        same(p.draws, "gammas", gammas_np)
        keys.append("gammas")
    M_evols = rng.integers(MIN_M_EVOL, M_max + 1, MAX_EXPERIMENTS)            # bh:163
    uniforms_outcome = rng.random(MAX_EXPERIMENTS)                             # bh:164
    same(p.draws, "M_evols", M_evols)
    same(p.draws, "uniforms_outcome", uniforms_outcome)
    assert list(p.draws)[: len(keys) + 2] == keys + ["M_evols", "uniforms_outcome"]


@pytest.mark.parametrize("inference", ["fl", "ps"])
@pytest.mark.parametrize("h", [0.0, 0.1])
def test_ed_draw_order(inference, h):
    if inference == "fl" and h > 0:
        pytest.skip("legacy FL mode asserts h == 0 (ed:156)")
    L, gamma = 1, 0.01
    N_code = 2 * L + 1
    spec = RunSpec(protocol="ed", gamma=gamma, h=h, L=L, inference=inference, model="theorem")
    p, m = build(spec)
    assert m == grid_resolution(N_code, EPS_MIN)
    M_max = legacy_M_max(EPS, N_code, N_code, m)                  # ed:161-164
    rng = legacy_rng(SEED, EPS)
    keys = []
    if h > 0:
        gammas_np = np.clip(rng.normal(gamma, gamma * h, N_code), 0.0, 10.0)   # ed:167-168
        same(p.draws, "gammas", gammas_np)
        keys.append("gammas")
    M_evols = rng.integers(MIN_M_EVOL, M_max + 1, MAX_EXPERIMENTS)            # ed:180
    uniforms_outcome = rng.random(MAX_EXPERIMENTS)                             # ed:181
    uniforms_syndrome = rng.random((MAX_EXPERIMENTS, N_code))                  # ed:182
    same(p.draws, "M_evols", M_evols)
    same(p.draws, "uniforms_outcome", uniforms_outcome)
    same(p.draws, "uniforms_syndrome", uniforms_syndrome)
    assert list(p.draws)[: len(keys) + 3] == keys + ["M_evols", "uniforms_outcome", "uniforms_syndrome"]


def _find_offsets(p, expected):
    """Locate the quenched offsets on the adapter (any attribute or draws entry
    equal to the expected array)."""
    cands = dict(getattr(p, "draws", {}) or {})
    for name in dir(p):
        if name.startswith("__"):
            continue
        try:
            v = getattr(p, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            cands[name] = v
    hits = [k for k, v in cands.items() if np.shape(v) == np.shape(expected) and np.array_equal(np.asarray(v, float), expected)]
    return hits, [k for k, v in cands.items() if np.shape(v) == np.shape(expected)]


@pytest.mark.parametrize("inference,marg", [("fl", False), ("ps", False), ("fl", True)])
@pytest.mark.parametrize("sigma", [0.01, 0.0])
def test_combined_draw_order_and_quenched_offsets(inference, marg, sigma):
    L, gamma = 1, 0.01
    N_code = 2 * L + 1
    N_total = 3 * N_code
    spec = RunSpec(protocol="combined", gamma=gamma, L=L, sigma=sigma, inference=inference, marg=marg, model="theorem")
    p, m = build(spec)
    assert m == grid_resolution(N_total, EPS_MIN)
    # quenched offsets: default_rng(seed).normal(0, sigma, N_total) once per seed (cb:465-469)
    seed_rng = np.random.default_rng(SEED)
    expected_offsets = seed_rng.normal(0.0, sigma, N_total) if sigma > 0 else np.zeros(N_total)
    hits, shaped = _find_offsets(p, expected_offsets)
    assert hits, f"no adapter attribute equals the legacy quenched offsets; candidates with the right shape: {shaped}"
    # per-eps draws (cb:230-232, then :243 in marg mode)
    M_max = legacy_M_max(EPS, N_total, N_total, m)                # cb:209-211
    rng = legacy_rng(SEED, EPS)
    M_evols = rng.integers(MIN_M_EVOL, M_max + 1, MAX_EXPERIMENTS)
    uniforms_outcome = rng.random(MAX_EXPERIMENTS)
    uniforms_syndrome = rng.random((MAX_EXPERIMENTS, N_total))
    same(p.draws, "M_evols", M_evols)
    same(p.draws, "uniforms_outcome", uniforms_outcome)
    same(p.draws, "uniforms_syndrome", uniforms_syndrome)
    order = ["M_evols", "uniforms_outcome", "uniforms_syndrome"]
    if marg:
        # legacy --marginalize-noise (the mode that produced combined_full_likelihood_marg.csv)
        # draws truth-side per-shot offsets after the syndrome uniforms (cb:242)
        epsilons_per_shot = rng.normal(0.0, sigma, (MAX_EXPERIMENTS, N_total))
        same(p.draws, "epsilons_per_shot", epsilons_per_shot)
        order.append("epsilons_per_shot")
    assert list(p.draws)[: len(order)] == order


@pytest.mark.parametrize("sigma", [0.0, 0.01])
def test_sequential_streams(sigma):
    """Per-target stream (sq:66): the first round draws uniforms of shape (K, N)
    with K = 30 (sq:94) from the legacy per-target generator.  The quenched
    offsets (sq:141-145, default_rng(seed)) are not exposed by the NumPy-only
    adapter; they are covered bitwise by the committed-CSV replay in
    test_regression.test_sequential_reproduces_committed_rows_exactly."""
    L = 1
    N = 2 * L + 1
    spec = RunSpec(protocol="sequential", L=L, sigma=sigma, model="theorem")
    p = protocols.get_protocol(spec, "cpu")
    p.run_seed(SEED, np.array([EPS]), "test", "none")
    first = np.asarray(p.draws["uniforms"][0])
    assert first.shape == (30, N)
    assert np.array_equal(first, legacy_rng(SEED, EPS).random((30, N)))
