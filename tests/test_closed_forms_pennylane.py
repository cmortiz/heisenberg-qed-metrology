"""Closed-form likelihoods versus an independent PennyLane state-vector circuit.

Written from the theorem statements in ``sensing.tex`` (Eq. ``eq:Ukdef``, the
decomposition ``eq:bitflip-decomp``, Theorem ``thm:bitflip``, Lemma
``lem:mapping``, Lemma ``lem:ghz-noise-marginalization``) and the public
docstrings of ``eqsp/closed_forms.py``.  No legacy simulator and no engine
kernel was consulted when this file was written.

Circuit side (built here, never through ``closed_forms`` internals):
    GHZ by Hadamard + CNOT chain over every data wire (for the combined layout
    the GHZ spans all 3 blocks, i.e. |0_L 0_L 0_L> + |1_L 1_L 1_L>);
    per-qubit ``qml.QubitUnitary(expm(-i M (omega_k Z + gamma_k X)))``;
    Z_i Z_{i+1} syndromes on consecutive block members extracted by CNOT pairs
    onto one ancilla wire each;  Hadamard on the data wires;  ``qml.probs`` on
    every wire;  classical post-processing to P(syndrome, parity) where parity
    is the XOR of the data bits (0 = +1 eigenvalue of X^{(x)N}).

Closed-form side: only the public functions of ``eqsp.closed_forms``.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from scipy.linalg import expm  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.stats import binom  # noqa: E402

from eqsp import closed_forms as cf  # noqa: E402

# Tolerance rule.  TOL = 1e-6 is the closed-form-vs-circuit bar of the plan (Tests subsection):
# it is loose enough for the float64 state-vector round-off accumulated over M-fold phase
# products at M = 123, and tight enough to catch a wrong O(gamma^2/Omega^2) term (>= 1e-3 at
# the fixtures).  Identities between two closed forms evaluated in float64 use 1e-12 (a few
# hundred ulp); sums that must be exactly 1 use 1e-10 to 1e-15 according to the number of
# terms; the theorem-form gap table uses 1e-8 because it compares O(1e-2) quantities.
TOL = 1e-6
OMEGA = 0.3
# (gamma, M) fixtures verified in the planning session against exp(-i M (omega Z + gamma X))
# (plan, "Verified this session"): weak noise / small M, weak noise / large M, strong noise.
GM = [(0.1, 7), (0.05, 123), (0.3, 5)]
LAYOUTS = {
    "N3": [[0, 1, 2]],
    "N5": [[0, 1, 2, 3, 4]],
    "N7": [list(range(7))],
    "3x3": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
}
# heterogeneous fixtures: omega_k = OMEGA + offset_k, per-qubit gamma_k.  Arbitrary
# O(sigma_eps = 0.01 to 0.03) offsets and gamma_k spread +-40% about 0.1 so that no two
# qubits share (omega_k, gamma_k) and a per-qubit bug cannot cancel in the product.
HET = {
    "N3": (np.array([0.02, -0.01, 0.005]), np.array([0.10, 0.07, 0.12])),
    "3x3": (
        np.array([0.02, -0.01, 0.005, 0.0, 0.015, -0.02, 0.01, -0.005, 0.03]),
        np.array([0.10, 0.07, 0.12, 0.09, 0.11, 0.08, 0.10, 0.13, 0.06]),
    ),
}

_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


# --------------------------------------------------------------------------
# Circuit helpers (independent implementation)
# --------------------------------------------------------------------------
def unitary(omega_k, gamma_k, M):
    """exp(-i M (omega Z + gamma X)) as an explicit 2x2 matrix (eq:Ukdef)."""
    return expm(-1j * M * (omega_k * _Z + gamma_k * _X))


def syndrome_pairs(blocks):
    """(i, i+1) wire pairs whose Z_i Z_{i+1} product is a stabilizer bit."""
    return [(b[j], b[j + 1]) for b in blocks for j in range(len(b) - 1)]


def joint_probs(omegas, gammas, M, blocks, ghz_sign=+1, bitflip_q=None, theta=0.0):
    """P over (data wires, ancilla wires) after GHZ -> U^{(x)N} -> syndromes ->
    Hadamards.  ``bitflip_q`` applies qml.BitFlip to every wire (default.mixed)."""
    N = len(omegas)
    pairs = syndrome_pairs(blocks)
    n_wires = N + len(pairs)
    Us = [unitary(o, g, M) for o, g in zip(omegas, gammas)]
    dev = qml.device("default.mixed" if bitflip_q is not None else "default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(wires=0)
        if ghz_sign < 0:
            qml.PauliZ(wires=0)
        if theta:
            qml.PhaseShift(-theta, wires=0)       # (|0..0> + e^{-i theta} |1..1>)/sqrt2 after the CNOTs
        for k in range(N - 1):
            qml.CNOT(wires=[k, k + 1])
        for k in range(N):
            qml.QubitUnitary(Us[k], wires=k)
        for idx, (i, j) in enumerate(pairs):
            qml.CNOT(wires=[i, N + idx])
            qml.CNOT(wires=[j, N + idx])
        for k in range(N):
            qml.Hadamard(wires=k)
        if bitflip_q is not None:
            for w in range(n_wires):
                qml.BitFlip(bitflip_q, wires=w)
        return qml.probs(wires=list(range(n_wires)))

    return np.asarray(circ(), dtype=float)


def state_after_evolution(omegas, gammas, M, ghz_sign=+1):
    """State vector of U^{(x)N} |GHZ_sign> on the data wires only."""
    N = len(omegas)
    Us = [unitary(o, g, M) for o, g in zip(omegas, gammas)]
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(wires=0)
        if ghz_sign < 0:
            qml.PauliZ(wires=0)
        for k in range(N - 1):
            qml.CNOT(wires=[k, k + 1])
        for k in range(N):
            qml.QubitUnitary(Us[k], wires=k)
        return qml.state()

    return np.asarray(circ())


def bits_of(n_states, n_wires):
    """(n_states, n_wires) bit table, wire 0 = most significant (PennyLane order)."""
    idx = np.arange(n_states)
    return (idx[:, None] >> (n_wires - 1 - np.arange(n_wires))[None, :]) & 1


def to_syndrome_parity(probs, N, n_syn):
    """P(syndrome tuple, parity) from the joint distribution over all wires."""
    n_wires = N + n_syn
    bits = bits_of(len(probs), n_wires)
    parity = bits[:, :N].sum(axis=1) % 2
    syn_code = bits[:, N:] @ (1 << np.arange(n_syn)[::-1]) if n_syn else np.zeros(len(probs), int)
    out = {}
    for s_code in range(1 << n_syn):
        s = tuple(int(b) for b in np.binary_repr(s_code, width=n_syn)) if n_syn else ()
        for v in (0, 1):
            out[(s, v)] = float(probs[(syn_code == s_code) & (parity == v)].sum())
    return out


def convolve_bitflips(probs, n_wires, q):
    """Classical readout-flip channel: every wire's bit flips independently with
    probability q.  Applied along each axis of the (2,)*n_wires tensor."""
    K = np.array([[1.0 - q, q], [q, 1.0 - q]])
    T = probs.reshape((2,) * n_wires)
    for ax in range(n_wires):
        T = np.moveaxis(np.tensordot(K, T, axes=([1], [ax])), 0, ax)
    return T.reshape(-1)


def representative_pattern(syn, blocks, N):
    """A pattern with the given Z_i Z_{i+1} syndrome: first member of every
    block clean, then x_{j+1} = x_j XOR s_j."""
    x = np.zeros(N, dtype=bool)
    pos = 0
    for b in blocks:
        x[b[0]] = False
        for j in range(len(b) - 1):
            x[b[j + 1]] = x[b[j]] ^ bool(syn[pos])
            pos += 1
    return x


def min_weight_pattern(syn, blocks, N):
    """Theorem thm:bitflip decoder convention: minimum-weight representative
    per block (complement a block whose representative has weight > |b|/2)."""
    x = representative_pattern(syn, blocks, N)
    for b in blocks:
        if x[b].sum() * 2 > len(b):
            x[b] ^= True
    return x


def my_syndrome(pattern, blocks):
    return tuple(int(pattern[i] ^ pattern[j]) for i, j in syndrome_pairs(blocks))


def homogeneous(blocks, gamma):
    N = sum(len(b) for b in blocks)
    return np.full(N, OMEGA), np.full(N, gamma)


def all_syndromes(n_syn):
    return list(itertools.product((0, 1), repeat=n_syn))


# --------------------------------------------------------------------------
# (i) exact code probabilities, homogeneous and heterogeneous
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("gamma,M", GM)
def test_ghz_code_probs_exact_matches_circuit(layout, gamma, M):
    blocks = LAYOUTS[layout]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks), N, n_syn)
    assert abs(sum(P.values()) - 1.0) < 1e-12
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    worst = 0.0
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        assert my_syndrome(pat, blocks) == s
        assert tuple(cf.syndrome_of(pat, blocks).tolist()) == s, "closed_forms.syndrome_of convention differs"
        pp, pm = cf.ghz_code_probs(a, c, blocks, pat)
        worst = max(worst, abs(pp - P[(s, 0)]), abs(pm - P[(s, 1)]))
        assert abs(pp - P[(s, 0)]) < TOL, (layout, gamma, M, s, pp, P[(s, 0)])
        assert abs(pm - P[(s, 1)]) < TOL, (layout, gamma, M, s, pm, P[(s, 1)])
    print(f"\n[exact] {layout} gamma={gamma} M={M}: max |circuit - ghz_code_probs| = {worst:.2e}")


@pytest.mark.parametrize("layout", list(HET))
@pytest.mark.parametrize("gamma,M", GM)
def test_ghz_code_probs_exact_heterogeneous(layout, gamma, M):
    blocks = LAYOUTS[layout]
    offsets, gam_scale = HET[layout]
    omegas = OMEGA + offsets
    gammas = gam_scale * (gamma / 0.1)       # keep the fixture's gamma scale
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks), N, n_syn)
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    worst = 0.0
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs(a, c, blocks, pat)
        worst = max(worst, abs(pp - P[(s, 0)]), abs(pm - P[(s, 1)]))
        assert abs(pp - P[(s, 0)]) < TOL, (layout, gamma, M, s)
        assert abs(pm - P[(s, 1)]) < TOL, (layout, gamma, M, s)
    print(f"\n[exact het] {layout} gamma-scale={gamma} M={M}: max diff = {worst:.2e}")


def test_ghz_code_probs_accepts_grid_batch():
    """(G, N) broadcasting over an omega grid must reproduce the per-omega values."""
    blocks = LAYOUTS["N3"]
    grid = np.linspace(0.0, 2 * np.pi, 7)
    gammas = np.full(3, 0.1)
    a, c = cf.single_qubit_amplitudes(grid[:, None] * np.ones(3)[None, :], gammas[None, :], 7)
    pat = np.array([False, True, False])
    pp, pm = cf.ghz_code_probs(a, c, blocks, pat)
    assert pp.shape == grid.shape
    for g, w in enumerate(grid):
        a1, c1 = cf.single_qubit_amplitudes(np.full(3, w), gammas, 7)
        pp1, pm1 = cf.ghz_code_probs(a1, c1, blocks, pat)
        assert abs(pp1 - pp[g]) < 1e-12 and abs(pm1 - pm[g]) < 1e-12


# --------------------------------------------------------------------------
# (ii) single-qubit p_k, phi_k and the theorem's decomposition
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gamma,M", GM)
@pytest.mark.parametrize("omega", [OMEGA, 0.32, 0.29])
def test_bitflip_params_single_qubit(gamma, M, omega):
    U = unitary(omega, gamma, M)
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def evolve(start):
        if start:
            qml.PauliX(wires=0)
        qml.QubitUnitary(U, wires=0)
        return qml.state()

    psi0 = np.asarray(evolve(0))          # U|0> = a|0> + c|1>
    psi1 = np.asarray(evolve(1))          # U|1> = c|0> + conj(a)|1>
    p, phi = cf.bitflip_params(omega, gamma, M)
    # probability of a flip
    assert abs(abs(psi0[1]) ** 2 - p) < 1e-12
    assert abs(abs(psi1[0]) ** 2 - p) < 1e-12
    # phase of the no-flip amplitude: <0|U|0> = beta e^{-i phi}, <1|U|1> = beta e^{+i phi}
    assert abs(np.exp(-1j * phi) * math.sqrt(1.0 - p) - psi0[0]) < 1e-12
    assert abs(np.exp(+1j * phi) * math.sqrt(1.0 - p) - psi1[1]) < 1e-12
    # relative phase between the |0> and |1> branches is 2 phi
    rel = np.angle(psi1[1] / psi0[0])
    assert abs(np.exp(1j * rel) - np.exp(2j * phi)) < 1e-12
    # a flipped qubit carries no signal phase: the flip amplitude is purely
    # imaginary and identical for |0>->|1> and |1>->|0> (eq:single-qubit-0/1)
    assert abs(psi0[1].real) < 1e-12 and abs(psi0[1] - psi1[0]) < 1e-12
    # theorem formulas: p = sin^2(M Omega) gamma^2/Omega^2, tan phi = omega tan(M Omega)/Omega,
    # beta^2 = cos^2(M Omega) + sin^2(M Omega) omega^2/Omega^2
    Om = math.sqrt(omega**2 + gamma**2)
    assert abs(p - math.sin(M * Om) ** 2 * gamma**2 / Om**2) < 1e-12
    beta = math.sqrt(math.cos(M * Om) ** 2 + math.sin(M * Om) ** 2 * omega**2 / Om**2)
    assert abs(beta - math.sqrt(1.0 - p)) < 1e-12
    # tan phi = omega tan(M Omega)/Omega, written pole-free
    assert abs(math.sin(phi) * math.cos(M * Om) * Om - omega * math.sin(M * Om) * math.cos(phi)) < 1e-12
    # closed_forms.single_qubit_amplitudes is the same object
    a, c = cf.single_qubit_amplitudes(omega, gamma, M)
    assert abs(a - psi0[0]) < 1e-12 and abs(c - psi0[1]) < 1e-12


@pytest.mark.parametrize("layout", ["N3", "N5", "N7"])
@pytest.mark.parametrize("gamma,M", GM)
def test_conditional_phase_is_two_Neff_phi_to_leading_order(layout, gamma, M):
    """Theorem thm:bitflip item 1 at amplitude level: the post-syndrome state
    carries relative phase 2 (N - d) phi between its |x> and |x-bar>
    components.  The circuit amplitude of |x> also receives the |1...1>-branch
    contribution, of relative size (p/(1-p))^{(N-2d)/2} (the square root of
    the coset partner's probability weight); for odd N these amplitude
    corrections cancel in the parity statistics (see the mixture tests), so
    this is a statement about amplitudes only.  Assert the leading term and
    bound the rest."""
    blocks = LAYOUTS[layout]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    psi = state_after_evolution(omegas, gammas, M)
    p, phi = cf.bitflip_params(OMEGA, gamma, M)
    ratio = p / (1.0 - p)
    n_syn = N - 1
    for s in all_syndromes(n_syn):
        x = min_weight_pattern(s, blocks, N)
        d = int(x.sum())
        ix = int("".join(str(int(b)) for b in x), 2)
        ixbar = int("".join(str(int(b)) for b in ~x), 2)
        rel = np.angle(psi[ixbar] / psi[ix])
        gap = abs(np.exp(1j * rel) - np.exp(2j * (N - d) * phi))
        # the |1...1>-branch amplitude enters psi[ix] and psi[ixbar] with relative size
        # r = ratio^((N-2d)/2) < 1; each factor (1 + r e^{ia}) has phase at most arcsin(r),
        # so the ratio's phase is off by at most 2 arcsin(r) <= pi r, and |e^{i t} - 1| <= |t|
        # gives gap <= pi r < 6 r (largest fixture r = 0.76 at gamma = 0.3, M = 5, d = (N-1)/2)
        r = ratio ** ((N - 2 * d) / 2)
        assert r < 1.0, (layout, gamma, M, s, r)
        bound = 6.0 * r
        assert gap <= bound, (layout, gamma, M, s, gap, bound)


# --------------------------------------------------------------------------
# (iii) bare GHZ parity including the even-n term
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("gamma,M", [(0.1, 1), (0.05, 1), (0.3, 1)] + GM)
def test_bare_parity_matches_circuit(n, gamma, M):
    omegas, gammas = np.full(n, OMEGA), np.full(n, gamma)
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, [[k] for k in range(n)]), n, 0)
    # blocks of size 1 produce no syndrome wires; parity = XOR of data bits
    x_exp = P[((), 0)] - P[((), 1)]
    val = cf.bare_parity(OMEGA, gamma, M, n)
    assert abs(val - x_exp) < TOL, (n, gamma, M, val, x_exp)
    assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, 0) - P[((), 0)]) < TOL
    assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, 1) - P[((), 1)]) < TOL
    r, theta, C = cf.bare_polar(OMEGA, gamma, M)
    assert abs(r**2 + C**2 - 1.0) < 1e-12
    if n % 2 == 0 and abs(C**n) > 10 * TOL:
        # wherever the even-n term C^n is resolvable at TOL, the odd-N-only
        # form r^n cos(n theta) fails this test
        assert abs(r**n * math.cos(n * theta) - x_exp) > TOL, (n, gamma, M, C)
    if n == 2:
        assert abs(C**n) > 10 * TOL, (gamma, M, C)


# --------------------------------------------------------------------------
# (iv) readout flips: (1-2q)^n visibility
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0.01, 0.1])
@pytest.mark.parametrize("gamma,M", [(0.1, 1), (0.1, 7)])
def test_readout_flip_bitflip_channel_N3(q, gamma, M):
    n = 3
    omegas, gammas = np.full(n, OMEGA), np.full(n, gamma)
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, [[k] for k in range(n)], bitflip_q=q), n, 0)
    for v in (0, 1):
        assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, v, q=q) - P[((), v)]) < TOL, (q, v)
    assert abs(cf.readout_visibility(q, n) - (1 - 2 * q) ** n) < 1e-15


@pytest.mark.parametrize("q", [0.001, 0.03])
@pytest.mark.parametrize("gamma,M", [(0.1, 1), (0.3, 5)])
@pytest.mark.parametrize("n", [5, 4])
def test_readout_flip_classical_convolution(q, gamma, M, n):
    omegas, gammas = np.full(n, OMEGA), np.full(n, gamma)
    probs = joint_probs(omegas, gammas, M, [[k] for k in range(n)])
    P = to_syndrome_parity(convolve_bitflips(probs, n, q), n, 0)
    for v in (0, 1):
        assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, v, q=q) - P[((), v)]) < TOL, (q, v)


# --------------------------------------------------------------------------
# (v) preparation visibility: mixture of GHZ+ and GHZ-
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["N3", "3x3"])
@pytest.mark.parametrize("k,F", [(0.99, 0.99), (0.995, 1.0)])
def test_prep_visibility_code_probs(layout, k, F):
    blocks = LAYOUTS[layout]
    gamma, M = GM[0]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    V = cf.prep_visibility(k, F, N)
    assert abs(V - k ** (N - 1) * F) < 1e-15
    Pp = joint_probs(omegas, gammas, M, blocks, ghz_sign=+1)
    Pm = joint_probs(omegas, gammas, M, blocks, ghz_sign=-1)
    P = to_syndrome_parity(0.5 * (1 + V) * Pp + 0.5 * (1 - V) * Pm, N, n_syn)
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs(a, c, blocks, pat, V=V)
        assert abs(pp - P[(s, 0)]) < TOL, (layout, s)
        assert abs(pm - P[(s, 1)]) < TOL, (layout, s)
        # syndrome statistics are V-independent
        pp1, pm1 = cf.ghz_code_probs(a, c, blocks, pat)
        assert abs((pp + pm) - (pp1 + pm1)) < 1e-12


EVEN_N_PREP_FINDING = (
    "FINDING: closed_forms.bare_parity_prob multiplies the whole <X^n> by V, "
    "including the even-n population term C^n; under the dephased-GHZ model of the "
    "ghz_code_probs docstring the circuit gives V Re[(A-iB)^n] + [n even] C^n "
    "(C^n unscaled). Odd n unaffected. Affects bare_adaptive at even N with k, F < 1."
)


@pytest.mark.parametrize("n", [3, 4])  # n = 4 failed before the 2026-09-02 fix; the assertion message is EVEN_N_PREP_FINDING
def test_prep_visibility_bare_parity(n):
    gamma, M = 0.1, 7
    V = cf.prep_visibility(0.99, 0.99, n)
    omegas, gammas = np.full(n, OMEGA), np.full(n, gamma)
    blocks = [[k] for k in range(n)]
    Pp = joint_probs(omegas, gammas, M, blocks, ghz_sign=+1)
    Pm = joint_probs(omegas, gammas, M, blocks, ghz_sign=-1)
    P = to_syndrome_parity(0.5 * (1 + V) * Pp + 0.5 * (1 - V) * Pm, n, 0)
    for v in (0, 1):
        assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, v, V=V) - P[((), v)]) < TOL, EVEN_N_PREP_FINDING


# --------------------------------------------------------------------------
# (vi) q-aware estimator likelihood
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["N3", "N5", "3x3"])
@pytest.mark.parametrize("q", [0.01, 0.03])
def test_qaware_probs_is_flip_convolution_and_normalised(layout, q):
    blocks = LAYOUTS[layout]
    gamma, M = GM[0]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    probs = joint_probs(omegas, gammas, M, blocks)
    P_obs = to_syndrome_parity(convolve_bitflips(probs, N + n_syn, q), N, n_syn)
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    total = 0.0
    for s in all_syndromes(n_syn):
        pp, pm = cf.qaware_probs(a, c, blocks, np.array(s), q)
        total += pp + pm
        assert abs(pp - P_obs[(s, 0)]) < TOL, (layout, q, s, pp, P_obs[(s, 0)])
        assert abs(pm - P_obs[(s, 1)]) < TOL, (layout, q, s, pm, P_obs[(s, 1)])
    assert abs(total - 1.0) < 1e-10
    # q = 0 reduces to the exact form (every syndrome)
    for s in all_syndromes(n_syn):
        pp0, pm0 = cf.qaware_probs(a, c, blocks, np.array(s), 0.0)
        pp, pm = cf.ghz_code_probs(a, c, blocks, representative_pattern(s, blocks, N))
        assert abs(pp0 - pp) < 1e-12 and abs(pm0 - pm) < 1e-12


def test_flip_weights_normalise():
    """For a fixed true pattern the weights q^dH (1-q)^(n_syn-dH) over all observed
    syndromes sum to 1 (binomial identity; the model the function implements)."""
    blocks = LAYOUTS["N5"]
    q, n_syn = 0.02, 4
    s_true = cf.syndrome_of(np.array([False, True, False, False, False]), blocks)
    total = sum(q ** int(np.sum(np.array(s) != s_true)) * (1 - q) ** (n_syn - int(np.sum(np.array(s) != s_true)))
                for s in all_syndromes(n_syn))
    assert abs(total - 1.0) < 1e-12


def test_flipped_syndrome_weights_sum_over_observed():
    """The function's weight for (E, s_obs) is q^dH (1-q)^(n_syn-dH) with dH the Hamming
    distance between the observed syndrome and the syndrome of E, for every pattern E it
    returns; the weights over all s_obs for a fixed E sum to 1 (checked by the binomial
    identity in test_flip_weights_normalise below)."""
    blocks = LAYOUTS["N5"]
    q = 0.02
    n_syn = 4
    for E2, w in cf.flipped_syndrome_weights(np.array([0, 1, 0, 0]), blocks, q):
        dH = int(np.sum(cf.syndrome_of(E2, blocks) != np.array([0, 1, 0, 0])))
        assert abs(w - q**dH * (1 - q) ** (n_syn - dH)) < 1e-15


# --------------------------------------------------------------------------
# (vii) theorem-form gap table
# --------------------------------------------------------------------------
def _gap_rows():
    rows = []
    for layout in ["N3", "N5", "N7"]:
        blocks = LAYOUTS[layout]
        for gamma, M in GM:
            omegas, gammas = homogeneous(blocks, gamma)
            N = len(omegas)
            n_syn = N - 1
            P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks), N, n_syn)
            a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
            p, phi = cf.bitflip_params(omegas, gammas, M)
            for s in all_syndromes(n_syn):
                x = min_weight_pattern(s, blocks, N)
                d = int(x.sum())
                ex_p, ex_m = cf.ghz_code_probs(a, c, blocks, x)
                th_p, th_m = cf.ghz_code_probs_theorem(p, phi, x)
                rows.append(dict(
                    layout=layout, gamma=gamma, M=M, s="".join(map(str, s)), d=d,
                    circ_Ps=P[(s, 0)] + P[(s, 1)], exact_Ps=float(ex_p + ex_m), thm_Ps=float(th_p + th_m),
                    exact_cond=float(ex_p / (ex_p + ex_m)), thm_cond=float(th_p / (th_p + th_m)),
                    p=float(p[0]), order=float((p[0] / (1 - p[0])) ** (N - 2 * d)),
                    thm_p=float(th_p), thm_m=float(th_m), ex_p=float(ex_p), ex_m=float(ex_m),
                ))
    return rows


def test_theorem_form_gap_table(capsys):
    rows = _gap_rows()
    hdr = f"{'layout':>6} {'gamma':>5} {'M':>3} {'syndrome':>8} {'d':>1} {'P(s) exact':>11} {'P(s) thm':>10} {'P(+|s) exact':>13} {'P(+|s) thm':>11} {'rel gap':>8} {'(p/(1-p))^(N-2d)':>17}"
    lines = [hdr]
    for r in rows:
        rel = abs(r["exact_cond"] - r["thm_cond"]) / max(r["exact_cond"], 1e-300)
        lines.append(f"{r['layout']:>6} {r['gamma']:>5} {r['M']:>3} {r['s']:>8} {r['d']:>1} "
                     f"{r['exact_Ps']:>11.6f} {r['thm_Ps']:>10.6f} {r['exact_cond']:>13.6f} {r['thm_cond']:>11.6f} "
                     f"{rel:>8.3f} {r['order']:>17.4f}")
    with capsys.disabled():
        print("\n[theorem-form gap table] exact = circuit-verified ghz_code_probs; thm = ghz_code_probs_theorem "
              "(minimum-weight representative only); rel gap = |P(+|s) exact - thm| / P(+|s) exact; "
              "the omitted coset partner has relative probability weight (p/(1-p))^(N-2d)")
        print("\n".join(lines))
    # the exact form is the circuit
    for r in rows:
        assert abs(r["circ_Ps"] - r["exact_Ps"]) < TOL
    # theorem P(s) does not sum to one (coset partners omitted); exact does
    for layout in ["N3", "N5", "N7"]:
        for gamma, M in GM:
            sub = [r for r in rows if r["layout"] == layout and r["gamma"] == gamma and r["M"] == M]
            assert abs(sum(r["exact_Ps"] for r in sub) - 1.0) < 1e-10
            thm_total = sum(r["thm_Ps"] for r in sub)
            # theorem mass = P(weight <= L) of independent Bernoulli(p) errors,
            # so the omitted coset partners carry exactly the binomial tail
            N = {"N3": 3, "N5": 5, "N7": 7}[layout]
            tail = float(binom.sf(N // 2, N, sub[0]["p"]))
            assert tail > 0 and abs((1.0 - thm_total) - tail) < 1e-12, (layout, gamma, M, 1 - thm_total, tail)
    # single block, odd N: the exact likelihood is the classical mixture of the
    # theorem form over {x, x-bar}, so the P(s) gap is exactly the coset
    # partner's weight (p/(1-p))^{N-2d} P_thm(s), and every joint probability
    # can only grow: P_exact(s, v) >= P_thm(s, v).  The conditional gap is then
    # bounded by (p/(1-p))^{N-2d}.
    for r in rows:
        assert abs((r["exact_Ps"] - r["thm_Ps"]) / r["thm_Ps"] - r["order"]) < 1e-10, r
        assert r["ex_p"] >= r["thm_p"] - 1e-15 and r["ex_m"] >= r["thm_m"] - 1e-15, r
        assert abs(r["exact_cond"] - r["thm_cond"]) <= r["order"] + 1e-12, r
    # orchestrator's reference point: gamma=0.1, M=7, N=3, d=1
    ref = [r for r in rows if r["layout"] == "N3" and r["gamma"] == 0.1 and r["M"] == 7 and r["d"] == 1]
    assert len(ref) == 3  # N = 3: syndromes 01, 10, 11 all have minimum weight 1
    # 0.0750 / 0.0539: the conditional flip probabilities computed in the planning session
    # (2026-09-02) from the float64 matrix exponential at this fixture, pinned to 3 decimals so
    # the gap table cannot drift silently; 1e-3 is the rounding of the pinned values
    for r in ref:
        assert abs(r["exact_cond"] - 0.0750) < 1e-3, r
        assert abs(r["thm_cond"] - 0.0539) < 1e-3, r
        assert r["exact_cond"] > r["thm_cond"]


# --------------------------------------------------------------------------
# (vii b) exact likelihood as a classical mixture over the coset
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["N3", "N5", "N7", "3x3"])
@pytest.mark.parametrize("gamma,M", GM)
def test_mixture_equals_circuit_at_odd_fixtures(layout, gamma, M):
    """ghz_code_probs_mixture: P(s, v) = sum_{x in S(s)} P_cl(x) [1 + v V cos(2 sum_{k not in x} phi_k)]/2
    equals the circuit for odd N_code and an odd number of blocks."""
    blocks = LAYOUTS[layout]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks), N, n_syn)
    p, phi = cf.bitflip_params(omegas, gammas, M)
    worst = 0.0
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs_mixture(p, phi, blocks, pat)
        worst = max(worst, abs(pp - P[(s, 0)]), abs(pm - P[(s, 1)]))
        assert abs(pp - P[(s, 0)]) < TOL, (layout, gamma, M, s)
        assert abs(pm - P[(s, 1)]) < TOL, (layout, gamma, M, s)
    print(f"\n[mixture] {layout} gamma={gamma} M={M}: max |circuit - mixture| = {worst:.2e}")


@pytest.mark.parametrize("layout", ["N3", "3x3"])
def test_mixture_equals_circuit_heterogeneous_and_with_V(layout):
    blocks = LAYOUTS[layout]
    offsets, gam = HET[layout]
    omegas, gammas = OMEGA + offsets, gam
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    M = 7
    V = cf.prep_visibility(0.99, 0.99, N)
    Pp = joint_probs(omegas, gammas, M, blocks, ghz_sign=+1)
    Pm = joint_probs(omegas, gammas, M, blocks, ghz_sign=-1)
    P = to_syndrome_parity(0.5 * (1 + V) * Pp + 0.5 * (1 - V) * Pm, N, n_syn)
    p, phi = cf.bitflip_params(omegas, gammas, M)
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs_mixture(p, phi, blocks, pat, V=V)
        assert abs(pp - P[(s, 0)]) < TOL, (layout, s)
        assert abs(pm - P[(s, 1)]) < TOL, (layout, s)


@pytest.mark.parametrize("layout,blocks", [("1x4", [[0, 1, 2, 3]]), ("2x3", [[0, 1, 2], [3, 4, 5]])])
@pytest.mark.parametrize("gamma,M", [(0.1, 7), (0.3, 5)])
def test_mixture_fails_for_even_layouts_but_exact_form_holds(layout, blocks, gamma, M, capsys):
    """Even N or an even block count: an interference term survives, so the
    classical mixture is wrong while ghz_code_probs still matches the circuit."""
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks), N, n_syn)
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    p, phi = cf.bitflip_params(omegas, gammas, M)
    worst_exact = 0.0
    worst_mix = 0.0
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs(a, c, blocks, pat)
        worst_exact = max(worst_exact, abs(pp - P[(s, 0)]), abs(pm - P[(s, 1)]))
        mp, mm = cf.ghz_code_probs_mixture(p, phi, blocks, pat)
        worst_mix = max(worst_mix, abs(mp - P[(s, 0)]), abs(mm - P[(s, 1)]))
    with capsys.disabled():
        print(f"\n[even layout {layout} gamma={gamma} M={M}] max|circuit - exact| = {worst_exact:.2e}, "
              f"max|circuit - mixture| = {worst_mix:.2e}")
    assert worst_exact < TOL
    # the even-N interference term is O(p^{N/2}) = O(1e-3) or larger at these fixtures; 1e-4 sits
    # two decades above TOL so a mixture that "accidentally" matched would be caught
    assert worst_mix > 1e-4, (layout, gamma, M, worst_mix)


# --------------------------------------------------------------------------
# (viii) Heisenberg-window likelihood: Gaussian average of the truth model
# --------------------------------------------------------------------------
# fixtures: N* = 1/(2 sigma^2) for both suite sigmas, N*/16 (grid start), and 4 N* at sigma = 0.1
# (grid end, visibility e^{-16}: the quadrature must still reproduce the near-flat likelihood)
@pytest.mark.parametrize("N,sigma", [(200, 0.05), (50, 0.1), (13, 0.05), (800, 0.1)])
@pytest.mark.parametrize("theta", [0.0, 1.3])
def test_hwindow_prob_is_gaussian_average(N, sigma, theta):
    omega = OMEGA % (math.pi / N)
    sd = sigma * math.sqrt(N)
    for outcome in (0, 1):
        def integrand(S):
            return cf.hwindow_true_prob(omega, N, S, theta, outcome) * math.exp(-0.5 * (S / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        val, err = quad(integrand, -12 * sd, 12 * sd, limit=400, epsabs=1e-13, epsrel=1e-13)
        assert err < 1e-10
        expect = cf.hwindow_prob(omega, N, sigma, theta, outcome)
        assert abs(val - expect) < 1e-8, (N, sigma, theta, outcome, val, expect)
        # visibility is e^{-2 N sigma^2}
        assert abs((2 * expect - 1) - (1 if outcome == 0 else -1) * math.exp(-2 * N * sigma**2) * math.cos(2 * N * omega - theta)) < 1e-14


def test_marg_log_visibility_matches_gaussian_average():
    """cb marg mode: E[cos(2 M sum_k (omega + eps_k))] over N_clean i.i.d. offsets.
    Fixture: M sigma sqrt(N_clean) = 0.87, where the visibility e^{-2 M^2 N sigma^2} = 0.22 is
    neither ~1 nor ~0; 1e-8 is the quadrature's epsabs times a safety factor."""
    M, N_clean, sigma = 13, 5, 0.03
    sd = sigma * math.sqrt(N_clean)
    omega = OMEGA

    def integrand(S):
        return math.cos(2 * M * (N_clean * omega + S)) * math.exp(-0.5 * (S / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    val, _ = quad(integrand, -12 * sd, 12 * sd, limit=400, epsabs=1e-13)
    expect = math.exp(cf.marg_log_visibility(M, N_clean, sigma)) * math.cos(2 * M * N_clean * omega)
    assert abs(val - expect) < 1e-8


# --------------------------------------------------------------------------
# (ix) preparation phase theta: coherent term rotates, population term does not
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.7, 2.9])
@pytest.mark.parametrize("n,gamma,M", [(3, 0.1, 7), (4, 0.3, 5), (5, 0.05, 123)])
def test_bare_parity_with_preparation_phase(n, gamma, M, theta):
    omegas, gammas = np.full(n, OMEGA), np.full(n, gamma)
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, [[k] for k in range(n)], theta=theta), n, 0)
    assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, 0, theta=theta) - P[((), 0)]) < TOL
    assert abs(cf.bare_parity_prob(OMEGA, gamma, M, n, 1, theta=theta) - P[((), 1)]) < TOL
    r, psi, C = cf.bare_polar(OMEGA, gamma, M)
    expect = r**n * math.cos(n * psi - theta) + (C**n if n % 2 == 0 else 0.0)
    assert abs(expect - (P[((), 0)] - P[((), 1)])) < TOL


@pytest.mark.parametrize("theta", [0.7, 2.9])
@pytest.mark.parametrize("layout,gamma,M", [("N3", 0.1, 7), ("N5", 0.05, 123), ("3x3", 0.3, 5)])
def test_code_probs_with_preparation_phase(layout, gamma, M, theta):
    """Dense form and coset mixture with the branch phase e^{-i theta}: P(s, v) =
    sum_x P_cl(x) [1 + v cos(2 Phi_x - theta)]/2."""
    blocks = LAYOUTS[layout]
    omegas, gammas = homogeneous(blocks, gamma)
    N = len(omegas)
    n_syn = len(syndrome_pairs(blocks))
    P = to_syndrome_parity(joint_probs(omegas, gammas, M, blocks, theta=theta), N, n_syn)
    p, phi = cf.bitflip_params(omegas, gammas, M)
    a, c = cf.single_qubit_amplitudes(omegas, gammas, M)
    for s in all_syndromes(n_syn):
        pat = representative_pattern(s, blocks, N)
        pp, pm = cf.ghz_code_probs_mixture(p, phi, blocks, pat, theta=theta)
        assert abs(pp - P[(s, 0)]) < TOL and abs(pm - P[(s, 1)]) < TOL, (layout, s, theta)
        pp2, pm2 = cf.ghz_code_probs(a, c, blocks, pat, theta=theta)
        assert abs(pp2 - P[(s, 0)]) < TOL and abs(pm2 - P[(s, 1)]) < TOL, (layout, s, theta)
