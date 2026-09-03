"""Engine loop tests against the interface fixed in the plan ('Engine contract',
'Protocol adapter').  A toy protocol with a fully controlled posterior drives
``run_epsilon``; nothing here depends on a physics kernel.

Skips until ``eqsp.engine`` exists.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

engine = pytest.importorskip("eqsp.engine")
jnp = pytest.importorskip("jax.numpy")

from eqsp.config import EngineConfig  # noqa: E402

OMEGA = 0.3


def _logsumexp64(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return m + math.log(np.exp(x - m).sum())


def _ctr(res):
    """Counters object from an EpsResult, whichever attribute carries it."""
    for name in ("ctr", "counters", "counter"):
        if hasattr(res, name):
            return getattr(res, name)
    return res


# --------------------------------------------------------------------------
# grid, posterior primitives
# --------------------------------------------------------------------------
@pytest.mark.parametrize("m", [4, 10])
def test_init_log_posterior_is_float32_uniform_normalised(m):
    logpk = engine.init_log_posterior(m)
    assert logpk.dtype == jnp.float32
    assert logpk.shape == (2**m,)
    assert abs(_logsumexp64(np.asarray(logpk))) < 1e-5
    assert np.allclose(np.asarray(logpk), -m * math.log(2.0), atol=1e-6)


def test_log_update_normalises_and_keeps_float32():
    m = 8
    logpk = engine.init_log_posterior(m)
    rng = np.random.default_rng(0)
    log_lik = jnp.asarray(rng.normal(size=2**m).astype(np.float32))
    out = engine.log_update(logpk, log_lik)
    assert out.dtype == jnp.float32
    assert abs(_logsumexp64(np.asarray(out))) < 1e-5
    ref = np.asarray(logpk, np.float64) + np.asarray(log_lik, np.float64)
    ref = ref - _logsumexp64(ref)
    assert np.max(np.abs(np.asarray(out, np.float64) - ref)) < 1e-4
    # a second update composes
    out2 = engine.log_update(out, log_lik)
    ref2 = ref + np.asarray(log_lik, np.float64)
    ref2 = ref2 - _logsumexp64(ref2)
    assert np.max(np.abs(np.asarray(out2, np.float64) - ref2)) < 1e-4


@pytest.mark.parametrize("builder", ["jax32", "np64"])
@pytest.mark.parametrize("period", [math.pi, 2 * math.pi])
def test_make_grid_is_float32_half_open_period(builder, period):
    m = 12
    xs = engine.make_grid(m, period, builder)
    xs_np = np.asarray(xs)
    assert xs_np.dtype == np.float32
    assert xs_np.shape == (2**m,)
    assert xs_np[0] == 0.0
    assert xs_np[-1] < period
    # spacing period / 2^m to float32 rounding
    assert abs(float(xs_np[1]) - period / 2**m) < 1e-6 * period
    # builders: jax32 = jnp.linspace(float32) (legacy sim:143); np64 = float64 linspace then cast (bh:156-158)
    if builder == "np64":
        ref = np.linspace(0, period, 2**m, endpoint=False, dtype=np.float64).astype(np.float32)
        assert np.array_equal(xs_np, ref)
    else:
        ref = np.asarray(jnp.linspace(0, period, 2**m, endpoint=False, dtype=jnp.float32))
        assert np.array_equal(xs_np, ref)


# --------------------------------------------------------------------------
# map_error: period handling
# --------------------------------------------------------------------------
def _delta_logpk(n, idx, weight=None):
    """Log-posterior concentrated on grid index/indices ``idx``."""
    p = np.full(n, 1e-30, dtype=np.float64)
    idx = np.atleast_1d(idx)
    w = np.ones(len(idx)) if weight is None else np.asarray(weight, float)
    p[idx] = w / w.sum()
    p /= p.sum()
    return jnp.asarray(np.log(p).astype(np.float32))


def test_map_error_bimodal_pi_periodic_posterior():
    """At gamma = 0 the bare likelihood is pi-periodic: on a [0, 2pi) grid the
    posterior is exactly bimodal (omega and omega + pi) and the MAP error is pi;
    on the [0, pi) grid it is unimodal and the error is at the grid resolution."""
    m = 12
    n = 2**m
    xs2 = engine.make_grid(m, 2 * math.pi, "np64")
    i1 = int(round(OMEGA / (2 * math.pi) * n))
    i2 = int(round((OMEGA + math.pi) / (2 * math.pi) * n))
    logpk = _delta_logpk(n, [i1, i2])
    err = float(engine.map_error(logpk, xs2, OMEGA, 2 * math.pi))
    # argmax picks one of the two equal peaks; circular distance is either ~0 or ~pi.
    # Break the tie towards the wrong peak to show the failure mode is real:
    logpk_wrong = _delta_logpk(n, [i1, i2], weight=[1.0, 1.0 + 1e-3])
    err_wrong = float(engine.map_error(logpk_wrong, xs2, OMEGA, 2 * math.pi))
    assert abs(err_wrong - math.pi) < 2 * math.pi / n + 1e-6
    assert min(abs(err), abs(err - math.pi)) < 2 * math.pi / n + 1e-6
    xs1 = engine.make_grid(m, math.pi, "jax32")
    j1 = int(round(OMEGA / math.pi * n))
    err1 = float(engine.map_error(_delta_logpk(n, j1), xs1, OMEGA, math.pi))
    assert err1 < math.pi / n + 1e-6


@pytest.mark.parametrize("period", [math.pi, 2 * math.pi])
def test_map_error_is_circular_distance(period):
    m = 10
    n = 2**m
    xs = engine.make_grid(m, period, "np64")
    # peak just below the wrap, truth just above zero: distance wraps through the period
    k = n - 2
    truth = 3 * period / n
    err = float(engine.map_error(_delta_logpk(n, k), xs, truth, period))
    assert abs(err - 5 * period / n) < 1e-5
    # peak at the truth's grid point
    k2 = 17
    err2 = float(engine.map_error(_delta_logpk(n, k2), xs, float(np.asarray(xs)[k2]), period))
    assert err2 < 1e-6
    # error never exceeds period/2
    err3 = float(engine.map_error(_delta_logpk(n, 0), xs, period / 2, period))
    assert abs(err3 - period / 2) < 1e-5


# --------------------------------------------------------------------------
# Toy protocol for run_epsilon
# --------------------------------------------------------------------------
class ToyProtocol:
    """Adapter with a scripted posterior.  ``script[i]`` is either None (reject)
    or a grid index at which the posterior is concentrated after round i; the
    default is the truth index (converged at the first check)."""

    period = 2 * math.pi
    grid_builder = "np64"
    family = "ham"
    omega_true = OMEGA

    def __init__(self, cfg, m, script=None, flat=False):
        self.cfg = cfg
        self.m = m
        self.xs = engine.make_grid(m, self.period, self.grid_builder)
        self.n = 2**m
        self.i_true = int(round(OMEGA / self.period * self.n))
        self.script = script or {}
        self.flat = flat
        self.check_calls = []
        self.apply_calls = []
        self.account_calls = []
        self.dtypes_seen = set()

    def grid_m(self, eps_min):
        return self.m

    def begin_run(self, seed, m):
        self.seed = seed

    def begin_epsilon(self, eps, rng):
        self.eps = eps
        self.rng = rng
        self.draws = {"u": rng.random(4)}

    def draw(self, i):
        rejected = (i in self.script) and self.script[i] is None
        return engine.Shot(cost=7, rejected=rejected, outcome=0, key=1, aux={"i": i})

    def apply(self, logpk, shot):
        self.dtypes_seen.add(str(np.asarray(logpk).dtype))
        self.apply_calls.append(shot.aux["i"])
        if self.flat:
            return engine.log_update(logpk, jnp.zeros_like(logpk))
        idx = self.script.get(shot.aux["i"], self.i_true)
        return _delta_logpk(self.n, idx)

    def account(self, shot, ctr):
        self.account_calls.append(shot.aux["i"])

    def check_due(self, i, ctr):
        self.check_calls.append(i)
        f = self.cfg.check_every
        if self.cfg.check_counter == "round0":
            return i % f == 0
        if self.cfg.check_counter == "round1":
            return (i + 1) % f == 0
        return ctr.n_updates % f == 0

    def row(self, ctr, converged, error):
        return {"converged": converged, "final_error": error}


def _cfg(**kw):
    base = dict(max_experiments=20, check_every=1, check_counter="round0", tolerance=1.2, track_best_error=False,
                stop="legacy")
    base.update(kw)
    return EngineConfig(**base)


def _run(p, eps=1e-2, seed=0):
    p.begin_run(seed, p.m)
    return engine.run_epsilon(p, eps, np.random.default_rng(seed))


def test_run_epsilon_converges_at_first_check_and_posterior_is_float32():
    p = ToyProtocol(_cfg(), m=12)
    res = _run(p)
    assert res.converged is True or res.converged == 1
    assert float(res.error) < 1.2 * 1e-2
    ctr = _ctr(res)
    assert ctr.n_rounds == 1 and ctr.n_updates == 1 and ctr.resources == 7
    assert p.dtypes_seen == {"float32"}


def test_rejected_shots_charge_cost_but_neither_update_nor_check():
    script = {0: None, 1: None, 2: None}          # three rejected rounds first
    p = ToyProtocol(_cfg(check_counter="update"), m=12, script=script)
    res = _run(p)
    ctr = _ctr(res)
    assert ctr.n_rejected == 3
    assert ctr.resources == 7 * 4                 # cost charged on rejected rounds (ed:224)
    assert ctr.n_updates == 1 and ctr.n_accepted == 1
    assert p.apply_calls == [3] and p.account_calls == [3]
    assert all(i >= 3 for i in p.check_calls)     # no check during rejected rounds
    assert res.converged


def test_exhausting_max_experiments_is_never_converged():
    p = ToyProtocol(_cfg(max_experiments=6), m=12, flat=True)
    res = _run(p, eps=1e-9)                       # tolerance 1.2e-9 rad: no check can pass
    ctr = _ctr(res)
    # legacy checks on the last round too (sim:196, bh:192); falling through is never converged
    assert not res.converged
    assert ctr.n_rounds == 6 and ctr.n_updates == 6
    # with a flat posterior the MAP sits at grid index 0 -> error = omega_true
    assert abs(float(res.error) - OMEGA) < 1e-5


def test_all_rejected_never_converged_and_no_update():
    p = ToyProtocol(_cfg(max_experiments=5), m=10, script={i: None for i in range(5)})
    res = _run(p)
    ctr = _ctr(res)
    assert not res.converged
    assert ctr.n_rejected == 5 and ctr.n_updates == 0 and ctr.resources == 35


@pytest.mark.parametrize("track_best", [True, False])
def test_track_best_error_semantics(track_best):
    """Posterior peaks at the truth in round 0 (error at grid resolution, but the
    tolerance is tiny so it does not converge), then jumps far away and stays.
    best-error mode reports the round-0 error; final mode reports the last."""
    m = 12
    n = 2**m
    i_true = int(round(OMEGA / (2 * math.pi) * n))
    far = (i_true + n // 4) % n
    script = {i: far for i in range(1, 6)}
    p = ToyProtocol(_cfg(max_experiments=6, track_best_error=track_best), m=m, script=script)
    res = _run(p, eps=1e-9)
    assert not res.converged
    err = float(res.error)
    if track_best:
        assert err < 2 * math.pi / n + 1e-6
    else:
        assert abs(err - math.pi / 2) < 2 * math.pi / n + 1e-5


@pytest.mark.parametrize("counter,expected_rounds", [("round0", 1), ("round1", 3), ("update", 4)])
def test_check_cadence_counters(counter, expected_rounds):
    """check_every = 3; one rejected round at i = 0.
    round0: i % 3 == 0 -> checks at i = 0, but that round is rejected, so the
            first check is i = 3 ... unless a converged posterior exists earlier;
            with the reject at i = 0 the first eligible check is i = 3 (4 rounds).
    Use no reject for round0 to keep the plan's semantics visible."""
    if counter == "round0":
        p = ToyProtocol(_cfg(check_every=3, check_counter="round0"), m=10)
    elif counter == "round1":
        p = ToyProtocol(_cfg(check_every=3, check_counter="round1"), m=10)
    else:
        # update counter: reject i = 0, updates at i = 1, 2, 3 -> third update at i = 3
        p = ToyProtocol(_cfg(check_every=3, check_counter="update"), m=10, script={0: None})
    res = _run(p)
    assert res.converged
    assert _ctr(res).n_rounds == expected_rounds


def test_last_round_is_always_checked():
    """i == max_experiments - 1 forces a check even off-cadence; the posterior
    is converged from round 0, so the result converges only at that last round."""
    p = ToyProtocol(_cfg(max_experiments=5, check_every=100, check_counter="round1"), m=10)
    res = _run(p)
    assert res.converged
    assert _ctr(res).n_rounds == 5


def test_tolerance_is_strict_multiple_of_eps():
    """Converged iff error < tolerance * eps.  Put the peak two grid points off
    the truth and pick eps so the error straddles the threshold."""
    m = 12
    n = 2**m
    xs = np.asarray(engine.make_grid(m, 2 * math.pi, "np64"))
    i_true = int(round(OMEGA / (2 * math.pi) * n))
    idx = i_true + 2
    err = abs(float(xs[idx]) - OMEGA)
    for eps, expect in [(err / 1.2 * 1.01, True), (err / 1.2 * 0.99, False)]:
        p = ToyProtocol(_cfg(max_experiments=3), m=m, script={i: idx for i in range(3)})
        res = _run(p, eps=eps)
        assert bool(res.converged) is expect, (eps, err, float(res.error))


# ---------------------------------------------------------------------------
# posterior (truth-free) stopping rule
# ---------------------------------------------------------------------------

def _gauss_logpk(xs, center, std, period):
    x = np.asarray(xs, np.float64)
    d = x - center
    d = d - period * np.round(d / period)
    return jnp.asarray(-0.5 * (d / std) ** 2, dtype=jnp.float32)


@pytest.mark.parametrize("std", [3e-2, 1e-3, 5e-5])
def test_posterior_stats_recovers_gaussian_std_and_map_error(std):
    """Circular std from wrapped displacements stays accurate in float32 down to
    std ~ 1e-5 rad (the resultant-length route, 1 - R, cannot resolve these)."""
    m = 20
    period = 2 * math.pi
    xs = engine.make_grid(m, period, "np64")
    center = OMEGA + 4 * std
    err, s = engine.posterior_stats(_gauss_logpk(xs, center, std, period), xs, OMEGA, period=period)
    assert abs(float(err) - 4 * std) < 2 * period / 2**m + 1e-6
    assert abs(float(s) / std - 1.0) < 2e-2


class GaussToy(ToyProtocol):
    """Posterior after update i is a wrapped Gaussian centred at ``center`` with
    std ``widths[i]`` (last width repeats)."""

    def __init__(self, cfg, m, widths, center=OMEGA):
        super().__init__(cfg, m)
        self.widths = widths
        self.center = center

    def apply(self, logpk, shot):
        i = min(shot.aux["i"], len(self.widths) - 1)
        self.apply_calls.append(shot.aux["i"])
        return _gauss_logpk(self.xs, self.center, self.widths[i], self.period)


def test_posterior_rule_ignores_lucky_map_hits_while_posterior_is_wide():
    """MAP sits exactly on the truth from the first update, but the posterior is
    10x wider than the target: the legacy rule stops at once, the posterior rule
    waits until 2 std <= eps."""
    eps = 1e-3
    widths = [10 * eps, 5 * eps, 2 * eps, eps, 0.5 * eps, 0.1 * eps]
    legacy = GaussToy(_cfg(max_experiments=10), m=18, widths=widths)
    res_l = _run(legacy, eps=eps)
    assert res_l.converged and _ctr(res_l).n_updates == 1
    post = GaussToy(_cfg(max_experiments=10, stop="posterior"), m=18, widths=widths)
    res_p = _run(post, eps=eps)
    ctr = _ctr(res_p)
    assert res_p.converged
    assert ctr.n_updates == 5                      # first update with 2 * width <= eps
    assert ctr.posterior_std <= 0.5 * eps
    assert ctr.resources == 7 * 5


def test_posterior_rule_stop_with_wrong_map_is_not_converged():
    eps = 1e-3
    post = GaussToy(_cfg(max_experiments=10, stop="posterior"), m=18, widths=[0.1 * eps], center=OMEGA + 5 * eps)
    res = _run(post, eps=eps)
    ctr = _ctr(res)
    assert not res.converged and ctr.n_updates == 1
    assert abs(float(res.error) - 5 * eps) < 1e-4


def test_posterior_rule_exhausting_budget_is_not_converged_and_records_std():
    p = ToyProtocol(_cfg(max_experiments=6, stop="posterior"), m=12, flat=True)
    res = _run(p, eps=1e-2)
    ctr = _ctr(res)
    assert not res.converged and ctr.n_rounds == 6 and ctr.n_updates == 6
    assert abs(ctr.posterior_std - 2 * math.pi / math.sqrt(12)) < 1e-2   # uniform on the circle
    assert p.check_calls == []                                            # cadence never consulted


def test_make_grid_offset_window():
    xs = np.asarray(engine.make_grid(8, 0.5, "np64", offset=0.3 - 0.25))
    assert abs(xs[0] - 0.05) < 1e-6 and xs[-1] < 0.55 and abs((xs[1] - xs[0]) - 0.5 / 256) < 1e-6
    xs2 = np.asarray(engine.make_grid(8, 0.5, "jax32", offset=0.05))
    assert np.allclose(xs, xs2, atol=1e-6)


def test_fixed_N_complex_kernel_matches_closed_form_sign():
    """ghz_x_parity_grid_complex returns prod_k (A_k - i B_k) of closed_forms.bare_bloch, including
    the sign of the imaginary part (the legacy real-part-only kernel hid a conjugation)."""
    from eqsp import closed_forms as cf, hamiltonian as H
    om = np.array([0.1, 0.3, 0.7], np.float32)
    for gamma, M, N in [(0.05, 7, 3), (0.1, 123, 5)]:
        Z = np.asarray(H.ghz_x_parity_grid_complex(jnp.asarray(om), jnp.full(N, gamma, np.float32), jnp.float32(M)))
        A, B, _ = cf.bare_bloch(om.astype(float), gamma, M)
        ref = (A - 1j * B) ** N
        assert np.allclose(Z[0], ref.real, atol=1e-4) and np.allclose(Z[1], ref.imag, atol=1e-4), (gamma, M, N)   # float32 M Omega
        for theta in (0.7, 2.9):
            assert np.allclose(Z[0] * np.cos(theta) + Z[1] * np.sin(theta),
                               cf.bare_parity_coherent(om.astype(float), gamma, M, N, theta), atol=1e-4)


def test_exact_ed_update_finite_when_syndrome_probability_saturates():
    """p = 1 exactly (grid crossing omega = 0 at sin^2(M Omega) = 1) must not produce a NaN posterior
    (v2_full2 finding: ED gamma = 0.10, L <= 2, 36 exhausted rows with error = half window)."""
    import jax.numpy as jnp
    from eqsp import hamiltonian as H
    G, N = 64, 3
    p_grid = jnp.linspace(0.0, 1.0, G, dtype=jnp.float32)          # includes exact 0 and exact 1
    phi_grid = jnp.linspace(0.0, 1.0, G, dtype=jnp.float32)
    logpk = jnp.full((G,), -jnp.log(G), dtype=jnp.float32)
    logc = jnp.array([-jnp.inf, 0.0, 0.0, -jnp.inf], dtype=jnp.float32)   # observed weight 1 (coset {1, 2})
    for outcome in (0, 1):
        out = H.exact_ed_update(logpk, p_grid, phi_grid, logc, jnp.float32(0.0), outcome, jnp.float32(0.3))
        assert bool(jnp.isfinite(out).all()), "NaN/inf in the exact ED update at p in {0, 1}"
        assert abs(float(jnp.exp(out).sum()) - 1.0) < 1e-4
    out = H.syndrome_update_homogeneous(logpk, p_grid, 0, N)
    assert bool(jnp.isfinite(out).all())


def test_marg_cap_limits_depth_to_the_information_optimal_M():
    """marg_cap caps the uniform depth draw at M = 1/(2 sigma sqrt(N_total)), the peak of
    4 M^2 N^2 exp(-4 M^2 N sigma^2); without it the draw reaches 1/(eps N_total)."""
    import numpy as np
    from eqsp.config import RunSpec
    import eqsp.protocols.combined as C

    sigma, N_total = 0.01, 9
    M_opt = 1.0 / (2.0 * sigma * math.sqrt(N_total))
    capped = C.make(RunSpec("combined", gamma=0.01, L=1, sigma=sigma, inference="fl", marg=True, marg_cap=True))
    plain = C.make(RunSpec("combined", gamma=0.01, L=1, sigma=sigma, inference="fl", marg=True))
    assert capped.M_cap == math.ceil(M_opt)
    for p in (capped, plain):
        p.begin_run(1, 20)
    for eps in (1e-2, 1e-3, 1e-4):
        capped.begin_epsilon(eps, np.random.default_rng(0))
        plain.begin_epsilon(eps, np.random.default_rng(0))
        assert capped.M_max == min(plain.M_max, capped.M_cap)
        assert capped.M_max <= math.ceil(M_opt)
    assert capped.spec.mode_label() == "combined_full_likelihood_marg_capped"


def test_marg_cap_requires_marg_with_offsets():
    from eqsp.config import RunSpec
    import eqsp.protocols.combined as C

    with pytest.raises(ValueError):
        C.make(RunSpec("combined", gamma=0.01, L=1, sigma=0.01, inference="fl", marg_cap=True))
    with pytest.raises(ValueError):
        C.make(RunSpec("combined", gamma=0.01, L=1, sigma=0.0, inference="fl", marg=True, marg_cap=True))
