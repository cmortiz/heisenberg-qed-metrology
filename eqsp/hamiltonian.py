"""JAX kernels for the Hamiltonian-noise likelihoods, plus float64 NumPy truth-side helpers.

Two families:

* Legacy-verbatim theorem-model kernels (Theorem thm:bitflip closed forms), lifted
  from simulation_bare_hamiltonian.py (bh), simulation_error_detected.py (ed) and
  simulation_combined.py (cb) after the closed-form check in
  checks/check_kernels.py.  Bodies are unchanged (only the function names
  differ) so XLA fusion and float32 rounding match the committed CSVs
  (plan, legacy convention 19).

* New exact-model kernels.  For odd N_code and an odd number of blocks the exact
  circuit likelihood is the classical mixture over the syndrome coset
  (closed_forms.ghz_code_probs_mixture):

      P(s, +/-) = 1/2 [ prod_b A_b  +/-  V (1-2q)^N  Re prod_b Z_b ]
      A_b = P_b(E_b) + P_b(Ebar_b)
      Z_b = P_b(E_b) e^{2i Phi_b(E_b)} + P_b(Ebar_b) e^{2i Phi_b(Ebar_b)}

  with P_b(x) = prod_{k in b} p_k^{x_k} (1-p_k)^{1-x_k} and Phi_b(x) = sum of
  phi_k over the qubits of b that are clean in x.  The theorem kernel keeps the
  E_b term only.  The q-aware estimator extends each block factor to a weighted
  sum over all 2^{N_code} patterns.  These kernels are computed in log space
  with a per-grid-point shift so that tiny P_b never underflow float32.

All kernels are float32 (jax_enable_x64 off) and take M as a traced float32
(bh:186, ed:251, cb:348; legacy convention 21).
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

LIK_FLOOR = 1e-30      # sim:71, bh:121, ed:75, ed:89, cb:77, cb:95, cb:118
P_MAX = 1.0 - 1e-7     # upper clip of a syndrome probability: 1 - 1e-30 == 1 in float32, so p = 1 (reached where the
                       # prior window crosses omega = 0 and sin^2(M Omega) = 1) gave log(1 - p) = -inf and 0 * -inf = NaN
HIGHEST = jax.lax.Precision.HIGHEST


# ==========================================================================
# Legacy-verbatim kernels: bare GHZ, fixed odd N (bh:58-124)
# ==========================================================================
@jit
def ghz_x_parity_grid(omega_grid, gammas, M_evol):
    """<X^N>_GHZ over the omega grid, odd N (bh:58-88, verbatim)."""
    # Shape: (grid_size, N)
    Omega_k = jnp.sqrt(omega_grid[:, None]**2 + gammas[None, :]**2 + 1e-30)
    MO = M_evol * Omega_k
    ratio = omega_grid[:, None] / Omega_k
    s = jnp.sin(MO)
    alpha = 1.0 - 2.0 * ratio**2 * s**2
    beta = 2.0 * ratio * s * jnp.cos(MO)  # sin(2 MO) * ratio / 1 = 2 sin(MO) cos(MO) ratio
    # Complex product over qubits: prod_k (alpha_k - i beta_k)
    re = alpha
    im = -beta
    # Accumulate product via log is numerically unstable (magnitudes near 1); do direct product.
    # For N up to a few tens, direct complex product is fine.
    def cplx_mul(carry, pair):
        cr, ci = carry
        pr, pi = pair
        return (cr * pr - ci * pi, cr * pi + ci * pr)
    init = (jnp.ones_like(omega_grid), jnp.zeros_like(omega_grid))
    # scan over the qubit axis
    pairs = (re.T, im.T)  # each shape (N, grid_size)
    final, _ = jax.lax.scan(
        lambda carry, p: (cplx_mul(carry, p), None),
        init,
        pairs,
    )
    return final[0]  # Re part


@jit
def ghz_x_parity_grid_complex(omega_grid, gammas, M_evol):
    """(2, G) array [Re, Im] of prod_k (alpha_k - i beta_k) over the omega grid: the
    GHZ coherence whose rotation by the preparation phase gives the exact-model
    parity Re[e^{-i theta} prod_k(...)] (new kernel; same recurrence as
    ghz_x_parity_grid)."""
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
    # Legacy beta = -B of closed_forms.bare_bloch (U^dag X U = A X + B Y + C Z): the legacy scan over
    # (alpha, -beta) therefore builds the conjugate of prod_k (A_k - i B_k), harmless for the real
    # part at theta = 0 but wrong for the rotated parity.  Use (alpha, +beta) = (A, -B).
    init = (jnp.ones_like(omega_grid), jnp.zeros_like(omega_grid))
    final, _ = jax.lax.scan(lambda carry, p: (cplx_mul(carry, p), None), init, (alpha.T, beta.T))
    return jnp.stack([final[0], final[1]])


@jit
def bare_parity_update_theta(logpk, coh, theta, log_vis, outcome):
    """Exact-model bare update, odd N: P(+ | omega) = (1 + e^{log_vis} Re[e^{-i theta} Z])/2
    with Z = coh[0] + i coh[1] from ghz_x_parity_grid_complex."""
    x = jnp.exp(log_vis) * (coh[0] * jnp.cos(theta) + coh[1] * jnp.sin(theta))
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * x) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


def nyquist_cap(m: int, N: int, period: float) -> int:
    """Largest depth M such that the grid resolves the M-th fringe: legacy rule
    2^(m-2) // N on a 2 pi grid (bh:146, ed:163, cb:210), generalised to a prior
    window of width ``period``: pi 2^m / (2 N period)."""
    return max(1, int(math.pi * 2**m / (2.0 * N * period)))


@jit
def ghz_x_parity_true(omega, gammas, M_evol):
    """Scalar <X^N>_GHZ at the true omega, float32 (bh:91-113, verbatim; legacy convention 10)."""
    Omega_k = jnp.sqrt(omega**2 + gammas**2 + 1e-30)
    MO = M_evol * Omega_k
    ratio = omega / Omega_k
    s = jnp.sin(MO)
    alpha = 1.0 - 2.0 * ratio**2 * s**2
    beta = 2.0 * ratio * s * jnp.cos(MO)
    re = alpha
    im = -beta
    # Product over qubits
    def cplx_mul(carry, pair):
        cr, ci = carry
        pr, pi = pair
        return (cr * pr - ci * pi, cr * pi + ci * pr)
    init = (jnp.float32(1.0), jnp.float32(0.0))
    final, _ = jax.lax.scan(
        lambda carry, p: (cplx_mul(carry, p), None),
        init,
        (re, im),
    )
    return final[0]


@jit
def bare_parity_update(logpk, x_parity_grid, outcome):
    """Log-space update with P(+1 | omega) = (1 + <X^N>)/2 (bh:116-124, verbatim)."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * x_parity_grid) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def bare_parity_update_vis(logpk, x_parity_grid, outcome, log_visibility):
    """bare_parity_update with an inference visibility exp(log_visibility) on
    <X^N> (aware estimator: (1-2q)^N k^(N-1) F).  New kernel."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    likelihood = (1.0 + sign * jnp.exp(log_visibility) * x_parity_grid) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


# ==========================================================================
# Legacy-verbatim kernels: ED (ed:68-127)
# ==========================================================================
@jit
def phase_update(logpk, phi_eff_grid, outcome):
    """P(v | omega, syndrome) = (1 + sign cos(2 phi_eff))/2 (ed:68-78 == cb:70-80, verbatim)."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    cos_val = jnp.cos(2.0 * phi_eff_grid)
    likelihood = (1.0 + sign * cos_val) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def syndrome_update_homogeneous(logpk, p_k_grid, d, N_code):
    """log P(d | omega, M) = d log p + (N-d) log(1-p) + const (ed:81-93, verbatim)."""
    p_safe = jnp.clip(p_k_grid, 1e-30, P_MAX)
    log_lik = d * jnp.log(p_safe) + (N_code - d) * jnp.log(1.0 - p_safe)
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def p_grid_homogeneous(omega_grid, gamma, M_evol):
    """p_k(omega) on the grid, homogeneous gamma (ed:96-103, verbatim)."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    sin2 = jnp.sin(M_Omega) ** 2
    ratio2 = omega_grid ** 2 / (omega_grid ** 2 + gamma ** 2 + 1e-30)
    return sin2 * (1.0 - ratio2)


@jit
def p_grid_homogeneous_exact(omega_grid, gamma, M_evol):
    """p_k(omega) on the grid without the float32 cancellation of the legacy
    form (1 - omega^2/Omega^2 loses up to 2% of p where p < 1e-7; finding of
    checks/check_kernels.py B4x).  Used by the exact model; the theorem model
    keeps p_grid_homogeneous for the bitwise regression."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    sin2 = jnp.sin(M_Omega) ** 2
    return sin2 * (gamma ** 2 / (omega_grid ** 2 + gamma ** 2 + 1e-30))


@jit
def phi_eff_grid_ed(omega_grid, gammas, M_evol):
    """phi_eff = sum_k phi_k over the grid, d = 0 (ed:106-116, verbatim)."""
    Omega_k = jnp.sqrt(omega_grid[:, None]**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    # sin/cos form avoids tan overflow near poles
    c = jnp.cos(M_Omega_k)
    s = jnp.sin(M_Omega_k)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    phi_k = jnp.arctan2(sign_c * s * omega_grid[:, None], sign_c * Omega_k * c)
    return jnp.sum(phi_k, axis=1)


@jit
def phi_grid_homogeneous(omega_grid, gamma, M_evol):
    """Single-qubit phi over the grid, homogeneous (ed:119-127, verbatim)."""
    Omega = jnp.sqrt(omega_grid**2 + gamma**2 + 1e-30)
    M_Omega = M_evol * Omega
    c = jnp.cos(M_Omega)
    s = jnp.sin(M_Omega)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    return jnp.arctan2(sign_c * s * omega_grid, sign_c * Omega * c)


# ==========================================================================
# Legacy-verbatim kernels: combined (cb:84-163)
# ==========================================================================
@jit
def phase_update_vis(logpk, phi_eff_grid, outcome, log_visibility):
    """P(v | omega) = (1 + sign lambda cos(2 Phi_eff))/2, lambda = exp(log_visibility)
    (cb:84-97 bayesian_update_marg, verbatim).  Also carries the aware-estimator
    visibility log((1-2q)^N k^(N-1) F) for theorem-model runs."""
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    cos_val = jnp.cos(2.0 * phi_eff_grid) * jnp.exp(log_visibility)
    likelihood = (1.0 + sign * cos_val) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def p_grid_per_qubit(omega_grid, gammas, epsilons, M_evol):
    """Per-qubit p_k(omega_j + eps_k) over the grid, (G, N_total) (cb:100-108, verbatim)."""
    omega_k = omega_grid[:, None] + epsilons[None, :]
    Omega_k = jnp.sqrt(omega_k**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    sin2 = jnp.sin(M_Omega_k) ** 2
    ratio_g2 = gammas[None, :] ** 2 / (omega_k ** 2 + gammas[None, :] ** 2 + 1e-30)
    return sin2 * ratio_g2


@jit
def syndrome_update_per_qubit(logpk, p_k_per_qubit_grid, error_mask):
    """Per-qubit Bernoulli syndrome factor (cb:111-123, verbatim)."""
    p_safe = jnp.clip(p_k_per_qubit_grid, 1e-30, P_MAX)
    log_lik = error_mask[None, :] * jnp.log(p_safe) + (1.0 - error_mask[None, :]) * jnp.log(1.0 - p_safe)
    log_lik_sum = jnp.sum(log_lik, axis=1)
    logpk = logpk + log_lik_sum
    logpk = logpk - logsumexp(logpk)
    return logpk


@jit
def phi_eff_grid_cb(omega_grid, gammas, epsilons, M_evol):
    """d = 0 Phi_eff(omega_j) = sum_k phi_k(omega_j + eps_k), (G,) (cb:126-138, verbatim)."""
    # Per-qubit omega_k = omega_j + epsilon_k: (G,1) + (1,N_total) -> (G, N_total)
    omega_k = omega_grid[:, None] + epsilons[None, :]
    Omega_k = jnp.sqrt(omega_k**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    # sin/cos form avoids tan overflow near pi/2 poles
    c = jnp.cos(M_Omega_k)
    s = jnp.sin(M_Omega_k)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    phi_k = jnp.arctan2(sign_c * s * omega_k, sign_c * Omega_k * c)
    return jnp.sum(phi_k, axis=1)


@jit
def phi_grid_per_qubit(omega_grid, gammas, epsilons, M_evol):
    """Per-qubit phi_k(omega_j + eps_k), (G, N_total) (cb:141-150, verbatim)."""
    omega_k = omega_grid[:, None] + epsilons[None, :]
    Omega_k = jnp.sqrt(omega_k**2 + gammas[None, :]**2 + 1e-30)
    M_Omega_k = M_evol * Omega_k
    c = jnp.cos(M_Omega_k)
    s = jnp.sin(M_Omega_k)
    sign_c = jnp.where(c >= 0, 1.0, -1.0)
    return jnp.arctan2(sign_c * s * omega_k, sign_c * Omega_k * c)


@jit
def phi_eff_with_errors(phi_per_qubit, error_mask):
    """phi_eff = sum_{k clean} phi_k = sum_k phi_k - sum_{errored k} phi_k (cb:153-163, verbatim)."""
    phi_eff_d0 = jnp.sum(phi_per_qubit, axis=1)
    correction = phi_per_qubit @ error_mask
    return phi_eff_d0 - correction


# ==========================================================================
# New kernels: exact model (odd N_code, odd block count)
# ==========================================================================
def _finish(logpk, shift, A, Zre, log_vis, outcome):
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    lik = 0.5 * (A + sign * jnp.exp(log_vis) * Zre)
    log_lik = shift + jnp.log(jnp.maximum(lik, LIK_FLOOR))
    logpk = logpk + log_lik
    return logpk - logsumexp(logpk)


@jit
def exact_ed_update(logpk, p_grid, phi_grid, logc, log_vis, outcome, theta=0.0):
    """Exact-model ED update, homogeneous block of N = len(logc) - 1 qubits.

    Block factors summed over pattern weight w = 0..N with log coefficients
    ``logc`` (float32, -inf excluded):  plain exact model logc = 0 at w in
    {d, N-d}; q-aware estimator logc[w] = log sum_{|x|=w} q^{d_H(s_obs, dx)}
    (1-q)^{N-1-d_H}; theorem-model q-aware additionally masks w > L.
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
    """Exact-model update for B contiguous blocks of N_code qubits (combined).

    p_grid, phi_grid: (G, B*N_code) per-qubit grids (cb precomputes).
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
# New kernel: bare GHZ, adaptive n, fixed-point angle (plan, 'Precision')
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
    theta the per-shot preparation phase (0 in the legacy model).

    The angle n theta_j is computed mod 2pi in 48-bit fixed point (three 16-bit
    limbs, n <= 2^14, all products < 2^31 exact).  At gamma = 0 the angle
    expression equals the legacy integer-mod angle sim:68-69 bitwise (checked in
    checks/check_kernels.py).  log_ro = n log(1-2q) scales the whole parity
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
# New kernel: Heisenberg window
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
def legacy_params_ed(omega_true, gammas, M_evols):
    """Batched (p_k, phi_k) exactly as ed:184-200: p = sin^2 (1 - omega^2/(omega^2+gamma^2+1e-30))."""
    M_evols_col = np.asarray(M_evols, np.float64)[:, None]
    gammas_row = np.asarray(gammas, np.float64)[None, :]
    Omega_k_all = np.sqrt(omega_true**2 + gammas_row**2)
    M_Omega_k_all = M_evols_col * Omega_k_all
    sin2_all = np.sin(M_Omega_k_all)**2
    denom_all = omega_true**2 + gammas_row**2 + 1e-30
    ratio2_all = omega_true**2 / denom_all
    p_k_all = sin2_all * (1.0 - ratio2_all)
    c_all = np.cos(M_Omega_k_all)
    s_all = np.sin(M_Omega_k_all)
    sign_c_all = np.where(c_all >= 0, 1.0, -1.0)
    phi_k_all = np.arctan2(sign_c_all * s_all * omega_true, sign_c_all * Omega_k_all * c_all)
    return p_k_all, phi_k_all


def legacy_params_cb(omegas, gammas, M_evols):
    """Batched (p_k, phi_k) exactly as cb:249-264 (quenched) with per-qubit omegas:
    p = sin^2 gamma^2/(omega^2+gamma^2+1e-30).  Also matches the per-shot form
    cb:287-298 when omegas has a leading shot axis."""
    omegas = np.asarray(omegas, np.float64)
    if omegas.ndim == 1:
        omegas_row = omegas[None, :]
    else:
        omegas_row = omegas
    M_evols_col = np.asarray(M_evols, np.float64)[:, None]
    gammas_row = np.asarray(gammas, np.float64)[None, :]
    Omega_k_all = np.sqrt(omegas_row**2 + gammas_row**2)
    M_Omega_k_all = M_evols_col * Omega_k_all
    sin2_all = np.sin(M_Omega_k_all)**2
    denom_all = omegas_row**2 + gammas_row**2 + 1e-30
    p_k_all = sin2_all * (gammas_row**2 / denom_all)
    c_all = np.cos(M_Omega_k_all)
    s_all = np.sin(M_Omega_k_all)
    sign_c_all = np.where(c_all >= 0, 1.0, -1.0)
    phi_k_all = np.arctan2(sign_c_all * s_all * omegas_row, sign_c_all * Omega_k_all * c_all)
    return p_k_all, phi_k_all


def exact_params(omegas, gammas, M_evols):
    """Batched exact-model (p_k, phi_k) from closed_forms.bitflip_params; shapes as legacy_params_cb."""
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


def pattern_log_weights(s_obs_block, q, Nc, theorem_L=None):
    """q-aware log weights for all 2^Nc patterns of one block given its observed
    syndrome bits (S, Nc-1): log[q^{d_H}(1-q)^{Nc-1-d_H}], shape (S, 2^Nc).
    theorem_L: mask patterns of weight > L to -inf (theorem-model q-aware)."""
    X = all_patterns(Nc)                                   # (P, Nc)
    syn = (X[:, :-1] ^ X[:, 1:]).astype(np.int64)         # (P, Nc-1)
    s = np.asarray(s_obs_block, np.int64)                  # (S, Nc-1)
    dH = np.sum(syn[None, :, :] != s[:, None, :], axis=2)  # (S, P)
    with np.errstate(divide="ignore"):
        logw = dH * np.log(q) + (Nc - 1 - dH) * np.log1p(-q)
    if theorem_L is not None:
        logw = np.where(X.sum(1)[None, :] > theorem_L, -np.inf, logw)
    return logw


def weight_log_coeffs(s_obs_block, q, Nc, theorem_L=None):
    """ED q-aware coefficients by pattern weight: log sum_{|x|=w} q^{d_H}(1-q)^{Nc-1-d_H},
    shape (S, Nc+1) (homogeneous block, so only the weight matters on the grid)."""
    X = all_patterns(Nc)
    logw = pattern_log_weights(s_obs_block, q, Nc, theorem_L)     # (S, P)
    wts = X.sum(1)
    out = np.full((logw.shape[0], Nc + 1), -np.inf)
    for w in range(Nc + 1):
        sel = wts == w
        if sel.any():
            with np.errstate(divide="ignore"):
                out[:, w] = np.log(np.sum(np.exp(logw[:, sel]), axis=1))
    return out


def bare_parity_batch(omega, gamma, M, ns, thetas=0.0):
    """(coherent, population) parts of the exact bare parity for an int array ns
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
