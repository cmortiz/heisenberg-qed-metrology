"""Float64 NumPy reference closed forms: the physics contract for the engine.

Every likelihood the engine uses is derived here from the circuit
U_k = exp(-i M (omega Z_k + gamma_k X_k)) acting on a GHZ state
(|0...0> + |1...1>)/sqrt(2), followed by Z_i Z_{i+1} syndrome measurements
within repetition-code blocks and an X^{(x)N} parity readout.  Functions
suffixed ``_theorem`` reproduce the closed forms stated in Theorem
``thm:bitflip`` of sensing.tex (and the legacy simulators); the unsuffixed
``ghz_code_probs`` is the exact circuit likelihood, which additionally keeps
the logical-X coset partner of each syndrome (relative weight
(p/(1-p))^{(N-2d)/2}).  ``tests/test_closed_forms_pennylane.py`` checks these
against a PennyLane state-vector simulation; JAX kernels in ``hamiltonian.py``
must agree with these functions to float32 precision.

Conventions
-----------
* Truth omega is ``OMEGA_TRUE = 0.3``; per-qubit omega_k = omega + eps_k.
* A "pattern" is a bool array (N,) of realized X-errors (True = flipped).
* ``blocks`` is a list of index arrays, one per repetition-code block; the
  syndrome of a block is Z_i Z_{i+1} on consecutive block members, so it
  fixes the block's pattern up to complementation.
* Parity outcome 0 = +1 eigenvalue of X^{(x)N}, 1 = -1 (legacy convention).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

OMEGA_TRUE = 0.3


# --------------------------------------------------------------------------
# Single-qubit unitary
# --------------------------------------------------------------------------
def omega_eff(omega, gamma):
    """Omega_k = sqrt(omega^2 + gamma^2)."""
    return np.sqrt(np.asarray(omega, float) ** 2 + np.asarray(gamma, float) ** 2)


def single_qubit_amplitudes(omega, gamma, M):
    """Amplitudes of U = exp(-i M (omega Z + gamma X)).

    Returns (a, c) with U|0> = a|0> + c|1> and U|1> = c|0> + conj(a)|1>:
        a = cos(M Omega) - i sin(M Omega) omega/Omega  ( = beta e^{-i phi} )
        c = -i sin(M Omega) gamma/Omega               ( |c|^2 = p )
    Broadcasts over omega and gamma.
    """
    omega = np.asarray(omega, float)
    gamma = np.asarray(gamma, float)
    Om = omega_eff(omega, gamma)
    Om_safe = np.where(Om == 0, 1.0, Om)
    s, co = np.sin(M * Om), np.cos(M * Om)
    a = co - 1j * s * omega / Om_safe
    c = -1j * s * gamma / Om_safe
    return a, c


def bitflip_params(omega, gamma, M):
    """Theorem thm:bitflip per-qubit (p_k, phi_k):
    p_k = sin^2(M Omega) gamma^2/Omega^2, tan(phi_k) = omega tan(M Omega)/Omega,
    with phi_k = -arg(a) on the full circle (atan2), which is the branch that
    makes a = beta e^{-i phi} exact."""
    a, c = single_qubit_amplitudes(omega, gamma, M)
    return np.abs(c) ** 2, -np.angle(a)


# --------------------------------------------------------------------------
# Bare GHZ parity (no syndrome)
# --------------------------------------------------------------------------
def bare_bloch(omega, gamma, M):
    """Coefficients of U^dag X U = A X + B Y + C Z (homogeneous single qubit)."""
    omega = np.asarray(omega, float)
    gamma = np.asarray(gamma, float)
    Om = omega_eff(omega, gamma)
    Om_safe = np.where(Om == 0, 1.0, Om)
    s, co = np.sin(M * Om), np.cos(M * Om)
    rho = omega / Om_safe
    A = 1.0 - 2.0 * rho**2 * s**2
    B = -2.0 * co * s * rho
    C = 2.0 * s**2 * omega * gamma / Om_safe**2
    return A, B, C


def bare_parity(omega, gamma, M, n):
    """Exact <X^{(x)n}> on an n-qubit GHZ state after U^{(x)n}:
    Re[(A - iB)^n] + [n even] C^n.  The legacy kernel (odd N only) omits the
    even-n term."""
    A, B, C = bare_bloch(omega, gamma, M)
    val = ((A - 1j * B) ** n).real
    if n % 2 == 0:
        val = val + C**n
    return val


def bare_polar(omega, gamma, M):
    """(r, theta, C) with <X^n> = r^n cos(n theta) + [n even] C^n; r^2 + C^2 = 1."""
    A, B, C = bare_bloch(omega, gamma, M)
    return np.hypot(A, B), np.arctan2(-B, A), C


def bare_parity_coherent(omega, gamma, M, n):
    """Coherence part Re[(A - iB)^n] of <X^{(x)n}>: the GHZ off-diagonal term,
    which flips sign between GHZ+ and GHZ- and is scaled by the preparation
    visibility V."""
    A, B, _ = bare_bloch(omega, gamma, M)
    return ((A - 1j * B) ** n).real


def bare_parity_population(omega, gamma, M, n):
    """Population part [n even] C^n of <X^{(x)n}>: from the Z^{(x)n} component
    of U^dag X U, identical for GHZ+ and GHZ-, so independent of V."""
    _, _, C = bare_bloch(omega, gamma, M)
    return C**n if n % 2 == 0 else 0.0 * np.asarray(C)


def bare_parity_prob(omega, gamma, M, n, outcome, q=0.0, V=1.0):
    """P(outcome | omega) for bare GHZ parity.  Readout flips scale the whole
    parity expectation by (1-2q)^n; the preparation visibility V scales only
    the coherent term (the even-n population term C^n survives dephasing;
    finding from tests/test_closed_forms_pennylane.py, 2026-09-02).
    outcome 0 -> +."""
    sign = 1.0 if outcome == 0 else -1.0
    x_n = V * bare_parity_coherent(omega, gamma, M, n) + bare_parity_population(omega, gamma, M, n)
    return 0.5 * (1.0 + sign * readout_visibility(q, n) * x_n)


# --------------------------------------------------------------------------
# Repetition-code GHZ with syndromes (ED, combined)
# --------------------------------------------------------------------------
def syndrome_of(pattern, blocks):
    """Z_i Z_{i+1} syndrome bits within each block (1 = -1 eigenvalue),
    concatenated over blocks; length sum_b (|b| - 1)."""
    pattern = np.asarray(pattern, bool)
    bits = []
    for b in blocks:
        pb = pattern[np.asarray(b)]
        bits.extend((pb[:-1] ^ pb[1:]).astype(int).tolist())
    return np.array(bits, int)


def _consistent_patterns(pattern, blocks):
    """All 2^{n_blocks} patterns with the same syndrome as ``pattern``
    (complement any subset of blocks)."""
    pattern = np.asarray(pattern, bool)
    out = []
    for flips in itertools.product([False, True], repeat=len(blocks)):
        x = pattern.copy()
        for f, b in zip(flips, blocks):
            if f:
                x[np.asarray(b)] ^= True
        out.append(x)
    return out


def ghz_code_probs(a, c, blocks, pattern, V=1.0, q_parity_n=None, q=0.0):
    """Exact joint probabilities (P(s, +), P(s, -)) of observing the syndrome
    of ``pattern`` and each parity outcome, for a GHZ state over all qubits in
    ``blocks`` evolved by the per-qubit unitaries with amplitudes (a_k, c_k)
    (arrays broadcastable to shape (..., N), e.g. (G, N) over an omega grid).

    V is the preparation visibility (dephased GHZ: rho = (1+V)/2 |GHZ+><GHZ+|
    + (1-V)/2 |GHZ-><GHZ-|); the syndrome statistics are V-independent and the
    coherent parity term scales by V.  q is a per-bit readout flip probability
    applied to the parity readout as (1-2q)^{q_parity_n} (default: all N
    qubits); syndrome-bit flips are handled by ``flipped_syndrome_weights``.
    """
    a = np.asarray(a)
    c = np.asarray(c)
    N = a.shape[-1]
    pattern = np.asarray(pattern, bool)
    xs = _consistent_patterns(pattern, blocks)
    # Pair each x with its global complement; visit each pair once.
    seen = set()
    p_plus = 0.0
    p_minus = 0.0
    for x in xs:
        key = tuple(x.tolist())
        if key in seen:
            continue
        xbar = ~x
        seen.add(key)
        seen.add(tuple(xbar.tolist()))
        T0x, T1x = _branch_amplitudes(a, c, x)
        T0b, T1b = _branch_amplitudes(a, c, xbar)
        # diagonal (V-independent) and coherent (x V) pieces, see module doc
        diag = 0.5 * (np.abs(T0x) ** 2 + np.abs(T1x) ** 2 + np.abs(T0b) ** 2 + np.abs(T1b) ** 2)
        diag = diag + V * (np.conj(T0x) * T1x + np.conj(T0b) * T1b).real
        cross = (np.conj(T0x) * T0b + np.conj(T1x) * T1b).real
        cross = cross + V * (np.conj(T0x) * T1b + np.conj(T1x) * T0b).real
        # P(x-pair, +/-) = (rho_xx + rho_xbar,xbar)/2 +/- Re rho_{x,xbar}
        p_plus = p_plus + diag + cross
        p_minus = p_minus + diag - cross
    if q:
        n_par = N if q_parity_n is None else q_parity_n
        vis = readout_visibility(q, n_par)
        tot = p_plus + p_minus
        coh = p_plus - p_minus
        p_plus = 0.5 * (tot + vis * coh)
        p_minus = 0.5 * (tot - vis * coh)
    return p_plus, p_minus


def _branch_amplitudes(a, c, x):
    """(T0, T1): amplitude of basis state |x> from the |0...0> and |1...1>
    branches (each carries the 1/sqrt(2) of the GHZ state)."""
    x = np.asarray(x, bool)
    T0 = np.prod(np.where(x, c, a), axis=-1) / math.sqrt(2.0)
    T1 = np.prod(np.where(x, np.conj(a), c), axis=-1) / math.sqrt(2.0)
    return T0, T1


def ghz_code_probs_theorem(p, phi, pattern, V=1.0, q_parity_n=None, q=0.0):
    """Theorem thm:bitflip / legacy model: P(s) = prod_k p_k^{e_k}(1-p_k)^{1-e_k}
    (one representative pattern), P(+/-|s) = (1 +/- V cos(2 sum_{k clean} phi_k))/2."""
    p = np.asarray(p)
    phi = np.asarray(phi)
    pattern = np.asarray(pattern, bool)
    Ps = np.prod(np.where(pattern, p, 1.0 - p), axis=-1)
    phi_eff = np.sum(np.where(pattern, 0.0, phi), axis=-1)
    coh = V * np.cos(2.0 * phi_eff)
    if q:
        n_par = p.shape[-1] if q_parity_n is None else q_parity_n
        coh = coh * readout_visibility(q, n_par)
    return 0.5 * Ps * (1.0 + coh), 0.5 * Ps * (1.0 - coh)


# --------------------------------------------------------------------------
# Realistic-noise factors
# --------------------------------------------------------------------------
def readout_visibility(q, n):
    """Parity of n bits each flipped independently with probability q keeps a
    coherent-term visibility (1-2q)^n."""
    return (1.0 - 2.0 * q) ** n


def prep_visibility(k, F, n):
    """GHZ preparation visibility k^(n-1) F (k: entangling-gate fidelity,
    F: single-qubit initialisation fidelity)."""
    return k ** (n - 1) * F


def flipped_syndrome_weights(s_obs, blocks, q):
    """q-aware estimator: for every true error pattern E (2^N, one per
    representative modulo global complement is NOT taken here; callers use
    ``ghz_code_probs`` which already sums the coset) return the weight
    q^{d_H(s_obs, syndrome(E))} (1-q)^{n_syn - d_H}.  Returns a list of
    (pattern, weight) over all 2^N patterns; patterns sharing a syndrome
    share a weight."""
    N = sum(len(b) for b in blocks)
    n_syn = len(s_obs)
    s_obs = np.asarray(s_obs, int)
    out = []
    for bits in itertools.product([False, True], repeat=N):
        E = np.array(bits, bool)
        dH = int(np.sum(syndrome_of(E, blocks) != s_obs))
        out.append((E, q**dH * (1.0 - q) ** (n_syn - dH)))
    return out


def qaware_probs(a, c, blocks, s_obs, q, V=1.0, model="exact"):
    """P(s_obs, +/-) under readout flips on every syndrome bit and every
    parity bit: sum over true syndromes s of P(s, +/-)(omega) weighted by the
    flip probability of s -> s_obs, with the parity visibility (1-2q)^N.
    Each distinct true syndrome is visited once (its representative pattern
    of minimum weight)."""
    N = sum(len(b) for b in blocks)
    n_syn = len(s_obs)
    s_obs = np.asarray(s_obs, int)
    reps = {}
    for bits in itertools.product([False, True], repeat=N):
        E = np.array(bits, bool)
        key = tuple(syndrome_of(E, blocks).tolist())
        if key not in reps or E.sum() < reps[key].sum():
            reps[key] = E
    p_plus = 0.0
    p_minus = 0.0
    for key, E in reps.items():
        dH = int(np.sum(np.array(key) != s_obs))
        w = q**dH * (1.0 - q) ** (n_syn - dH)
        if model == "exact":
            pp, pm = ghz_code_probs(a, c, blocks, E, V=V, q=q)
        else:
            p, phi = np.abs(c) ** 2, -np.angle(a)
            pp, pm = ghz_code_probs_theorem(p, phi, E, V=V, q=q)
        p_plus = p_plus + w * pp
        p_minus = p_minus + w * pm
    return p_plus, p_minus


# --------------------------------------------------------------------------
# Shot-to-shot signal offsets
# --------------------------------------------------------------------------
def marg_log_visibility(M, N_clean, sigma):
    """Analytic Gaussian marginalisation of shot-to-shot offsets eps_k ~ N(0, sigma^2)
    on N_clean qubits at depth M: E[cos(2M sum_k (omega+eps_k))] = cos(2 M N_clean omega)
    e^{-2 M^2 N_clean sigma^2} (leading order in the offsets)."""
    return -2.0 * M**2 * N_clean * sigma**2


def hwindow_prob(omega, N, sigma, theta, outcome):
    """Heisenberg-window likelihood at fixed N, M = 1, gamma = 0, readout phase
    theta: P(+|omega) = (1 + e^{-2 N sigma^2} cos(2 N omega - theta))/2."""
    sign = 1.0 if outcome == 0 else -1.0
    return 0.5 * (1.0 + sign * math.exp(-2.0 * N * sigma**2) * np.cos(2.0 * N * np.asarray(omega) - theta))


def hwindow_true_prob(omega, N, S, theta, outcome):
    """Truth side of the window sweep: realized offset sum S = sum_k eps_k,
    P(+) = (1 + cos(2 (N omega + S) - theta))/2."""
    sign = 1.0 if outcome == 0 else -1.0
    return 0.5 * (1.0 + sign * np.cos(2.0 * (N * omega + S) - theta))


def ghz_code_probs_mixture(p, phi, blocks, pattern, V=1.0, q_parity_n=None, q=0.0):
    """Exact joint likelihood in closed form for odd N_code and an odd number of
    blocks (every configuration in the paper), equal to ``ghz_code_probs`` to
    machine precision there (checked 2026-09-02 for 1x3, 1x5, 1x7, 3x3, 3x5;
    it fails for even N or an even block count, where an extra interference
    term survives):

        P(s, v) = sum_{x in S(s)} P_cl(x) [1 + v V cos(2 sum_{k not in x} phi_k)] / 2,

    with S(s) the 2^{n_blocks} error patterns consistent with the syndrome
    (each block's pattern or its complement) and
    P_cl(x) = prod_{k not in x} (1 - p_k) prod_{k in x} p_k.  The syndrome
    identifies each block's pattern only up to the logical X, and the two
    cosets do not interfere in the parity statistics, so the likelihood is a
    classical mixture of the theorem's closed form over S(s).  The theorem's
    form keeps the minimum-weight representative only; the dropped terms have
    relative probability weight (p/(1-p))^{N_code - 2 d_b} per block.
    """
    p = np.asarray(p)
    phi = np.asarray(phi)
    pattern = np.asarray(pattern, bool)
    p_plus = 0.0
    p_minus = 0.0
    for x in _consistent_patterns(pattern, blocks):
        pp, pm = ghz_code_probs_theorem(p, phi, x, V=V, q_parity_n=q_parity_n, q=q)
        p_plus = p_plus + pp
        p_minus = p_minus + pm
    return p_plus, p_minus
