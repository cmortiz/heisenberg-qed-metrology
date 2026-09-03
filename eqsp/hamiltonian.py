"""JAX kernels for the Hamiltonian-noise likelihoods, plus float64 NumPy truth-side helpers.

For odd N_code and an odd number of blocks the exact circuit likelihood is the
classical mixture over the syndrome coset (closed_forms.ghz_code_probs_mixture):

    P(s, +/-) = 1/2 [ prod_b A_b  +/-  V (1-2q)^N  Re prod_b Z_b ]
    A_b = P_b(E_b) + P_b(Ebar_b)
    Z_b = P_b(E_b) e^{2i Phi_b(E_b)} + P_b(Ebar_b) e^{2i Phi_b(Ebar_b)}

with P_b(x) = prod_{k in b} p_k^{x_k} (1-p_k)^{1-x_k} and Phi_b(x) = sum of
phi_k over the qubits of b that are clean in x.  The q-aware estimator extends
each block factor to a weighted sum over all 2^{N_code} patterns.  These kernels
are computed in log space with a per-grid-point shift so that tiny P_b never
underflow float32.

All kernels are float32 (jax_enable_x64 off) and take M as a traced float32.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import math

import numpy as np
from jax import jit
from jax.scipy.special import logsumexp

from . import closed_forms as cf

LIK_FLOOR = 1e-30      # relative floor on a likelihood before log
P_MAX = 1.0 - 1e-7     # upper clip of a syndrome probability: 1 - 1e-30 == 1 in float32, so p = 1 (reached where the
                       # prior window crosses omega = 0 and sin^2(M Omega) = 1) gave log(1 - p) = -inf and 0 * -inf = NaN
HIGHEST = jax.lax.Precision.HIGHEST


# ==========================================================================
# Bare GHZ at fixed odd N
# ==========================================================================
@jit
def ghz_x_parity_grid_complex(omega_grid, gammas, M_evol):
    """(2, G) array [Re, Im] of prod_k (A_k - i B_k) over the omega grid: the GHZ
    coherence whose rotation by the preparation phase gives the parity
    Re[e^{-i theta} prod_k(...)]."""
    Omega_k = jnp.sqrt(omega_grid[:, None]**2 + gammas[None, :]**2 + 1e-30)
    MO = M_evol * Omega_k
    ratio = omega_grid[:, None] / Omega_k
    s = jnp.sin(MO)
    alpha = 1.0 - 2.0 * ratio**2 * s**2
    beta = 2.0 * ratio * s * jnp.cos(MO)

    def cplx_mul(carry, pair):
        cr, ci = carry
        pr, pi = pair
        return (cr * pr - ci * pi, cr * pi + ci * pr)
    # (alpha, beta) = (A, -B) of closed_forms.bare_bloch (U^dag X U = A X + B Y + C Z), so the scan
    # builds prod_k (A_k - i B_k); the conjugate would be harmless for the real part at theta = 0 but
    # wrong for the rotated parity.
    init = (jnp.ones_like(omega_grid), jnp.zeros_like(omega_grid))
    final, _ = jax.lax.scan(lambda carry, p: (cplx_mul(carry, p), None), init, (alpha.T, beta.T))
    return jnp.stack([final[0], final[1]])


@jit
def bare_parity_update_theta(logpk, coh, theta, log_vis, outcome):
    """Bare update, odd N: P(+ | omega) = (1 + e^{log_vis} Re[e^{-i theta} Z])/2
    with Z = coh[0] + i coh[1] from ghz_x_parity_grid_complex."""
    x = jnp.exp(log_vis) * (coh[0] * jnp.cos(theta) + coh[1] * jnp.sin(theta))
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * x) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


def nyquist_cap(m: int, N: int, period: float) -> int:
    """Largest depth M such that the grid resolves the M-th fringe:
    pi 2^m / (2 N period) on a prior window of width ``period`` (2^(m-2) // N on
    a 2 pi grid)."""
    return max(1, int(math.pi * 2**m / (2.0 * N * period)))


# ==========================================================================
# Homogeneous single-block grids (ED)
# ==========================================================================
@jit
def p_grid_homogeneous_exact(omega_grid, gamma, M_evol):
    """p_k(omega) = sin^2(M Omega) gamma^2 / Omega^2 on the grid, written so that
    float32 does not cancel: the algebraically equal form sin^2 (1 - omega^2/Omega^2)
    subtracts two numbers near 1 and, where p < 1e-7, misses p by up to 2%
    (measured against the float64 closed form)."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    sin2 = jnp.sin(M_Omega) ** 2
    return sin2 * (gamma ** 2 / (omega_grid ** 2 + gamma ** 2 + 1e-30))


@jit
def phi_grid_homogeneous(omega_grid, gamma, M_evol):
    """Single-qubit phi over the grid, homogeneous gamma."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    c = jnp.cos(M_Omega)
    s = jnp.sin(M_Omega)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    return jnp.arctan2(sign_c * s * omega_grid, sign_c * Omega * c)


# ==========================================================================
# Per-qubit grids (combined: per-qubit offsets omega + eps_k)
# ==========================================================================
@jit
def p_grid_per_qubit(omega_grid, gammas, epsilons, M_evol):
    """Per-qubit p_k(omega_j + eps_k) over the grid, (G, N_total)."""
    omega_k = omega_grid[:, None] + epsilons[None, :]
    Omega_k = jnp.sqrt(omega_k**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    sin2 = jnp.sin(M_Omega_k) ** 2
    ratio_g2 = gammas[None, :] ** 2 / (omega_k ** 2 + gammas[None, :] ** 2 + 1e-30)
    return sin2 * ratio_g2


@jit
def phi_grid_per_qubit(omega_grid, gammas, epsilons, M_evol):
    """Per-qubit phi_k(omega_j + eps_k), (G, N_total)."""
    omega_k = omega_grid[:, None] + epsilons[None, :]
    Omega_k = jnp.sqrt(omega_k**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    c = jnp.cos(M_Omega_k)
    s = jnp.sin(M_Omega_k)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    return jnp.arctan2(sign_c * s * omega_k, sign_c * Omega_k * c)


# ==========================================================================
# Exact-model updates (odd N_code, odd block count)
# ==========================================================================
def _finish(logpk, shift, A, Zre, log_vis, outcome):
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    lik = 0.5 * (A + sign * jnp.exp(log_vis) * Zre)
    # A and Zre carry the per-grid-point shift (A = O(1) at the dominant pattern), so the floor acts on the
    # likelihood relative to its own scale at that grid point: the same relative floor at every omega_j.
    log_lik = shift + jnp.log(jnp.maximum(lik, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


@jit
def exact_ed_update(logpk, p_grid, phi_grid, logc, log_vis, outcome, theta=0.0):
    """ED update, homogeneous block of N = len(logc) - 1 qubits.

    Block factors summed over pattern weight w = 0..N with log coefficients
    ``logc`` (float32, -inf excluded): the mismatched estimator sets logc = 0 at
    w in {d, N-d} (the two coset members); the q-aware estimator sets
    logc[w] = log sum_{|x|=w} q^{d_H(s_obs, dx)} (1-q)^{N-1-d_H}.
    log_vis = log of the inference visibility on the coherent term; theta the
    per-shot preparation phase (cos(2 Phi - theta)).
    """
    N = logc.shape[0] - 1
    w = jnp.arange(N + 1, dtype=jnp.float32)
    p_safe = jnp.clip(p_grid, 1e-30, P_MAX)
    lp = jnp.log(p_safe)[:, None]
    l1p = jnp.log(1.0 - p_safe)[:, None]
    la = w[None, :] * lp + (N - w)[None, :] * l1p + logc[None, :]        # (G, N+1)
    shift = jnp.max(la, axis=1)
    e = jnp.exp(la - shift[:, None])
    A = jnp.sum(e, axis=1)
    Zre = jnp.sum(e * jnp.cos(2.0 * (N - w)[None, :] * phi_grid[:, None] - theta), axis=1)
    return _finish(logpk, shift, A, Zre, log_vis, outcome)


@jit
def exact_blocks_update(logpk, p_grid, phi_grid, X, logw, log_vis, marg_coef, outcome, theta=0.0):
    """Update for B contiguous blocks of N_code qubits (combined).

    p_grid, phi_grid: (G, B*N_code) per-qubit grids.
    X: (B, P, N_code) float32 patterns per block (P = 2: E_b and its complement;
       q-aware: all 2^N_code patterns).  logw: (B, P) float32 log weights
       (-inf excludes a pattern).  marg_coef = 2 M^2 sigma^2 damps each
       pattern's coherent term by exp(-marg_coef * n_clean) (marg inference;
       0 otherwise).  log_vis = log of the inference visibility.
    """
    B, P, Nc = X.shape
    p_safe = jnp.clip(p_grid, 1e-30, P_MAX)
    lp = jnp.log(p_safe)
    l1p = jnp.log(1.0 - p_safe)
    shift = 0.0
    A = 1.0
    Z = 1.0 + 0.0j
    for b in range(B):
        sl = slice(b * Nc, (b + 1) * Nc)
        Xb = X[b]                                                        # (P, Nc)
        Xc = 1.0 - Xb
        # HIGHEST: on GPU the default matmul precision uses TF32 (10-bit mantissa) for float32 dots; these dots
        # sum log-probabilities that are exponentiated below, so a 1e-3 relative error would leak into P_b.
        la = (jnp.dot(lp[:, sl], Xb.T, precision=HIGHEST)
              + jnp.dot(l1p[:, sl], Xc.T, precision=HIGHEST) + logw[b][None, :])   # (G, P)
        Ph = jnp.dot(phi_grid[:, sl], Xc.T, precision=HIGHEST)          # clean-qubit phase sums
        n_clean = jnp.sum(Xc, axis=1)                                    # (P,)
        sb = jnp.max(la, axis=1)
        e = jnp.exp(la - sb[:, None])
        damp = jnp.exp(-marg_coef * n_clean)[None, :]
        Ab = jnp.sum(e, axis=1)
        Zb = jnp.sum(e * damp * jnp.exp(2.0j * Ph), axis=1)
        shift = shift + sb
        A = A * Ab
        Z = Z * Zb
    Zre = jnp.real(Z) * jnp.cos(theta) + jnp.imag(Z) * jnp.sin(theta)      # Re[e^{-i theta} Z]
    return _finish(logpk, shift, A, Zre, log_vis, outcome)


# ==========================================================================
# Bare GHZ, adaptive n, fixed-point angle
# ==========================================================================
def bare_adaptive_tables(m: int, gamma: float, omega=None):
    """Float64 setup for the fixed-point bare-adaptive kernel on the [0, pi) grid
    omega_j = pi j / 2^m.  Returns jnp arrays (q0, q1, q2, logr, logC): the
    three 16-bit limbs of q_j = round(theta_j / 2pi * 2^48), log r_j and
    log |C_j| (float32) with (r, theta, C) = closed_forms.bare_polar at M = 1.
    At gamma = 0, q_j = j << 32 exactly, r = 1 and C = 0."""
    G = 2**m
    j = np.arange(G, dtype=np.float64)
    om = np.pi * j / G if omega is None else np.asarray(omega, np.float64)
    if gamma == 0.0:
        u = j / G
        logr = np.zeros(G)
        logC = np.full(G, np.log(1e-30))
    else:
        r, theta, C = cf.bare_polar(om, gamma, 1)
        u = np.mod(theta / (2.0 * np.pi), 1.0)
        logr = np.log(np.minimum(np.maximum(r, 1e-30), 1.0))
        logC = np.log(np.maximum(np.abs(C), 1e-30))
    q = np.rint(u * 2.0**48).astype(np.uint64) & np.uint64((1 << 48) - 1)
    q0 = (q & np.uint64(0xFFFF)).astype(np.uint32)
    q1 = ((q >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint32)
    q2 = ((q >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint32)
    return (jnp.asarray(q0), jnp.asarray(q1), jnp.asarray(q2),
            jnp.asarray(logr.astype(np.float32)), jnp.asarray(logC.astype(np.float32)))


@jit
def bare_adaptive_update(logpk, q0, q1, q2, logr, logC, n, log_ro, log_V, outcome, theta=0.0):
    """P(+ | omega_j) = (1 + e^{log_ro} [e^{log_V} r_j^n cos(n psi_j - theta) + [n even] C_j^n])/2,
    theta the per-shot preparation phase.

    The angle n theta_j is computed mod 2pi in 48-bit fixed point (three 16-bit
    limbs, n <= 2^14, all products < 2^31 exact); at gamma = 0 this is the exact
    integer-mod angle.  log_ro = n log(1-2q) scales the whole parity
    (readout flips); log_V = log(k^(n-1) F) scales only the coherent term (a
    dephased GHZ keeps the even-n population term C^n); both 0 for the
    mismatched estimator."""
    n_u = jnp.asarray(n).astype(jnp.uint32)
    A = n_u * q0
    B = n_u * q1
    C = n_u * q2
    top = (A >> 16) + B + (C << 16)                       # top 32 bits of n q mod 2^48
    angle = 2.0 * jnp.pi * top.astype(jnp.float32) / 4294967296.0
    cos_n = jnp.cos(angle - theta)
    rn = jnp.exp(n * logr)
    cn = jnp.where(n % 2 == 0, jnp.exp(n * logC), 0.0)
    x_n = jnp.exp(log_V) * (rn * cos_n) + cn
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * jnp.exp(log_ro) * x_n) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


# ==========================================================================
# Heisenberg window
# ==========================================================================
@jit
def hwindow_update(logpk, xs, N, log_lambda, theta, outcome):
    """P(+ | omega) = (1 + lambda_N cos(2 N omega - theta))/2, lambda_N = e^{-2 N sigma^2}."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * jnp.exp(log_lambda) * jnp.cos(2.0 * N * xs - theta)) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


# ==========================================================================
# Float64 NumPy truth-side helpers
# ==========================================================================
def exact_params(omegas, gammas, M_evols):
    """Batched (p_k, phi_k) from closed_forms.bitflip_params.  ``omegas`` is (N,)
    or (S, N); ``M_evols`` is (S,); returns two (S, N) arrays."""
    omegas = np.asarray(omegas, np.float64)
    omegas_row = omegas[None, :] if omegas.ndim == 1 else omegas
    M_col = np.asarray(M_evols, np.float64)[:, None]
    gammas_row = np.asarray(gammas, np.float64)[None, :]
    Om = np.sqrt(omegas_row**2 + gammas_row**2)
    Om_safe = np.where(Om == 0, 1.0, Om)
    s, co = np.sin(M_col * Om), np.cos(M_col * Om)
    a = co - 1j * s * omegas_row / Om_safe
    c = -1j * s * gammas_row / Om_safe
    return np.abs(c) ** 2, -np.angle(a)


def exact_block_terms(p, phi, pattern, n_blocks):
    """Per-block (A_b, Z_b) of the exact mixture likelihood for shot-batched arrays
    p, phi, pattern of shape (S, B*N_code), N_code odd.  Returns (A (S,B), Z (S,B) complex)."""
    S, N = p.shape
    Nc = N // n_blocks
    if Nc % 2 == 0:
        raise ValueError("exact mixture form requires odd N_code")
    x = np.asarray(pattern, bool).reshape(S, n_blocks, Nc)
    p3 = p.reshape(S, n_blocks, Nc)
    ph3 = phi.reshape(S, n_blocks, Nc)
    PE = np.prod(np.where(x, p3, 1.0 - p3), axis=2)
    PEbar = np.prod(np.where(x, 1.0 - p3, p3), axis=2)
    PhE = np.sum(np.where(x, 0.0, ph3), axis=2)      # clean qubits of E
    PhEbar = np.sum(np.where(x, ph3, 0.0), axis=2)   # clean qubits of Ebar = errored in E
    A = PE + PEbar
    Z = PE * np.exp(2j * PhE) + PEbar * np.exp(2j * PhEbar)
    return A, Z


def exact_p_plus(p, phi, pattern, n_blocks, V=1.0, theta=0.0):
    """P(+ | syndrome of pattern) = (1 + V Re[e^{-i theta} prod_b Z_b] / prod_b A_b)/2, shot-batched."""
    A, Z = exact_block_terms(p, phi, pattern, n_blocks)
    ph = np.exp(-1j * np.asarray(theta, np.float64))
    return 0.5 * (1.0 + V * np.real(ph * np.prod(Z, axis=1)) / np.prod(A, axis=1))


def block_syndromes(pattern, n_blocks):
    """Z_i Z_{i+1} bits per block: (S, B*N_code) bool -> (S, B, N_code-1) bool."""
    S, N = pattern.shape
    Nc = N // n_blocks
    x = np.asarray(pattern, bool).reshape(S, n_blocks, Nc)
    return x[:, :, :-1] ^ x[:, :, 1:]


def decode_min_weight(syndromes):
    """Min-weight pattern per block from (S, B, N_code-1) syndromes -> (S, B*N_code) bool.
    For odd N_code exactly one coset member has weight <= L."""
    S, B, Nm1 = syndromes.shape
    Nc = Nm1 + 1
    x = np.zeros((S, B, Nc), bool)
    x[:, :, 1:] = (np.cumsum(syndromes.astype(np.int64), axis=2) % 2).astype(bool)
    heavy = x.sum(axis=2) > Nc // 2
    x = np.where(heavy[:, :, None], ~x, x)
    return x.reshape(S, B * Nc)


def all_patterns(Nc):
    """(2^Nc, Nc) bool patterns, index = binary code (bit k = qubit k)."""
    idx = np.arange(2**Nc)
    return ((idx[:, None] >> np.arange(Nc)[None, :]) & 1).astype(bool)


def pattern_log_weights(s_obs_block, q, Nc):
    """q-aware log weights for all 2^Nc patterns of one block given its observed
    syndrome bits (S, Nc-1): log[q^{d_H}(1-q)^{Nc-1-d_H}], shape (S, 2^Nc)."""
    X = all_patterns(Nc)                                   # (P, Nc)
    syn = (X[:, :-1] ^ X[:, 1:]).astype(np.int64)         # (P, Nc-1)
    s = np.asarray(s_obs_block, np.int64)                  # (S, Nc-1)
    dH = np.sum(syn[None, :, :] != s[:, None, :], axis=2)  # (S, P)
    with np.errstate(divide="ignore"):
        logw = dH * np.log(q) + (Nc - 1 - dH) * np.log1p(-q)
    return logw


def weight_log_coeffs(s_obs_block, q, Nc):
    """ED q-aware coefficients by pattern weight: log sum_{|x|=w} q^{d_H}(1-q)^{Nc-1-d_H},
    shape (S, Nc+1) (homogeneous block, so only the weight matters on the grid)."""
    X = all_patterns(Nc)
    logw = pattern_log_weights(s_obs_block, q, Nc)               # (S, P)
    wts = X.sum(1)
    out = np.full((logw.shape[0], Nc + 1), -np.inf)
    for w in range(Nc + 1):
        sel = wts == w
        if sel.any():
            with np.errstate(divide="ignore"):
                out[:, w] = np.log(np.sum(np.exp(logw[:, sel]), axis=1))
    return out


def bare_parity_batch(omega, gamma, M, ns, thetas=0.0):
    """(coherent, population) parts of the bare parity for an int array ns
    (float64): r^n cos(n psi - theta) and [n even] C^n.  <X^n> = V * coherent + population
    under a dephased GHZ of visibility V (closed_forms.bare_parity_prob)."""
    r, psi, C = cf.bare_polar(omega, gamma, M)
    ns = np.asarray(ns, np.int64)
    coherent = r**ns * np.cos(ns * psi - np.asarray(thetas, np.float64))
    population = np.where(ns % 2 == 0, C**ns, 0.0)
    return coherent, population


def parity_flip_prob(q, n):
    """Probability that the parity of n bits flips when each flips with prob q."""
    return 0.5 * (1.0 - (1.0 - 2.0 * q) ** np.asarray(n, np.float64))
