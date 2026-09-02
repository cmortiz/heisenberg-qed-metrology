#!/usr/bin/env python3
"""Scratch verification (workstream A, trust rule): every JAX kernel in
eqsp/hamiltonian.py against the float64 closed forms in eqsp/closed_forms.py,
the exact-model per-block identities, and the fixed-point bare-adaptive angle
against the legacy integer-mod kernel (simulation.py:59-74) at gamma = 0.

Run:  cd numerics && uv run python checks/check_kernels.py
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import itertools
import numpy as np
import jax

X64 = "--x64" in sys.argv          # formula-isolation mode: float64 kernels, tight tolerances

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp

from eqsp import closed_forms as cf
from eqsp import hamiltonian as H
from eqsp import engine

jax.config.update("jax_enable_x64", X64)   # after the imports: engine.py forces x64 off at import (sim:22)

DT = jnp.float64 if X64 else jnp.float32
def jf(x):
    return jnp.asarray(x, dtype=DT)

RESULTS = []
print(f"mode: {'x64 (formula check, tol 1e-9)' if X64 else 'float32 (legacy precision; tolerance scales with N*M)'}")


def rec(name, err, tol, note=""):
    if err is None:
        print(f"skip  {name:70s} (identically zero likelihood, e.g. d = N with v = -)")
        return
    ok = bool(err <= tol)
    RESULTS.append((name, err, tol, ok, note))
    print(f"{'PASS' if ok else 'FAIL'}  {name:70s} err={err:.3e}  tol={tol:.0e}  {note}")


def post_err(post_kernel, ref_lik):
    """Kernel posterior (from a uniform prior, so proportional to the likelihood)
    against the reference likelihood, both scaled to unit peak.  Returns the
    max absolute deviation relative to the peak; in x64 mode the max of that
    and the relative error on the region above 1e-3 of the peak (the float32
    kernels clip likelihoods at 1e-30 near their zeros, where a relative metric
    is meaningless)."""
    ref = np.asarray(ref_lik, np.float64)
    k = np.asarray(post_kernel, np.float64)
    if not np.isfinite(ref).all() or ref.max() <= 1e-200:
        return None                                   # degenerate reference (identically zero likelihood): skipped
    ref = ref / ref.max()
    k = k / k.max()
    e_peak = float(np.max(np.abs(k - ref)))
    if not X64:
        return e_peak
    mask = ref >= 1e-3
    e_rel = float(np.max(np.abs(k[mask] - ref[mask]) / ref[mask]))
    return max(e_peak, e_rel)


def tol32(N, M, base=1e-5):
    """Expected float32 error scale: angles M*Omega up to ~2 pi M with 2^-24
    relative rounding, accumulated over N factors (legacy convention 10)."""
    return 1e-9 if X64 else base * max(1, N) * max(1, M)


OMEGA = 0.3
G = 4096
grid = np.linspace(0, 2 * np.pi, G, endpoint=False, dtype=np.float64)
grid32 = jf(grid)
prior = jnp.full(G, -np.log(G), dtype=DT)
rng = np.random.default_rng(12345)

# ==========================================================================
print("\n=== A. bare_fixed kernels (bh:58-124) vs closed_forms.bare_parity ===")
for N, gamma, M in itertools.product([3, 5, 7], [0.01, 0.05, 0.1, 0.3], [1, 7, 123]):
    par = np.asarray(H.ghz_x_parity_grid(grid32, jnp.full(N, gamma, DT), jf(M)))
    ref = cf.bare_parity(grid, gamma, M, N)
    rec(f"A1 ghz_x_parity_grid N={N} g={gamma} M={M}", float(np.max(np.abs(par - ref))), tol32(N, M, 4e-6))
for N, gamma, M in itertools.product([2, 4], [0.1, 0.3], [1, 7]):
    par = np.asarray(H.ghz_x_parity_grid(grid32, jnp.full(N, gamma, DT), jf(M)))
    ref = cf.bare_parity(grid, gamma, M, N)
    A_, B_, C_ = cf.bare_bloch(grid, gamma, M)
    print(f"      even N={N} g={gamma} M={M}: kernel-vs-exact max|diff|={np.max(np.abs(par-ref)):.3e}, "
          f"max|C^N|={np.max(np.abs(C_**N)):.3e} (kernel omits C^N; bare_fixed uses odd N only)")
for N, gamma, M in (itertools.product([3, 7], [0.05, 0.1], [1, 7, 123, 1000]) if not X64 else []):
    xt = float(H.ghz_x_parity_true(jnp.float32(OMEGA), jnp.full(N, gamma, jnp.float32), jnp.float32(M)))
    ref = float(cf.bare_parity(OMEGA, gamma, M, N))
    print(f"      A2 ghz_x_parity_true N={N} g={gamma} M={M}: float32 truth {xt:+.7f} vs float64 {ref:+.7f} "
          f"diff={xt-ref:+.2e}")
# posterior of the update kernel = normalized likelihood
par = H.ghz_x_parity_grid(grid32, jnp.full(3, 0.1, DT), jf(7))
for outcome in (0, 1):
    post = np.exp(np.asarray(H.bare_parity_update(prior, par, outcome)))
    ref = cf.bare_parity_prob(grid, 0.1, 7, 3, outcome)
    rec(f"A3 bare_parity_update outcome={outcome}", post_err(post, ref), tol32(3, 7))
    post = np.exp(np.asarray(H.bare_parity_update_vis(prior, par, outcome, jf(np.log(0.8 * 0.98**3)))))
    ref = cf.bare_parity_prob(grid, 0.1, 7, 3, outcome, q=0.01, V=0.8)   # odd N: population term is 0
    rec(f"A4 bare_parity_update_vis outcome={outcome} (V=0.8,q=0.01)", post_err(post, ref), tol32(3, 7))

# ==========================================================================
print("\n=== B. ED theorem kernels (ed:68-127) vs closed_forms (theorem model) ===")
for N, gamma, M in itertools.product([3, 5, 7], [0.01, 0.1, 0.3], [1, 7, 123]):
    p_ref, phi_ref = cf.bitflip_params(grid, gamma, M)
    p_k = np.asarray(H.p_grid_homogeneous(grid32, jf(gamma), jf(M)))
    phi_k = np.asarray(H.phi_grid_homogeneous(grid32, jf(gamma), jf(M)))
    phi_eff = np.asarray(H.phi_eff_grid_ed(grid32, jnp.full(N, gamma, DT), jf(M)))
    rec(f"B1 p_grid_homogeneous g={gamma} M={M} (abs)", float(np.max(np.abs(p_k - p_ref))), tol32(1, M, 2e-6))
    relp = np.abs(p_k - p_ref) / np.maximum(p_ref, 1e-30)
    print(f"      B1r p_grid_homogeneous g={gamma} M={M}: max relative error {relp.max():.2e} at omega={grid[relp.argmax()]:.3f}, "
          f"p={p_ref[relp.argmax()]:.2e} (ed:102-103 computes gamma^2/Omega^2 as 1 - omega^2/Omega^2)")
    rec(f"B2 phi_grid_homogeneous cos2phi g={gamma} M={M}",
        float(np.max(np.abs(np.cos(2 * phi_k) - np.cos(2 * phi_ref)))), tol32(1, M))
    rec(f"B3 phi_eff_grid_ed cos(2 N phi) N={N} g={gamma} M={M}",
        float(np.max(np.abs(np.cos(2 * phi_eff) - np.cos(2 * N * phi_ref)))), tol32(N, M))
    # float32 mode: reference built from the kernel's own p (isolates composition
    # rounding from the ed:102-103 cancellation, which B1r quantifies); x64: exact p
    p_used = p_ref if X64 else np.asarray(p_k, np.float64)
    pN = np.repeat(p_used[:, None], N, axis=1)
    phiN = np.repeat(phi_ref[:, None], N, axis=1)
    pg = H.p_grid_homogeneous(grid32, jf(gamma), jf(M))
    ps = H.phi_grid_homogeneous(grid32, jf(gamma), jf(M))
    for d in range(0, N + 1):
        pattern = np.zeros(N, bool); pattern[:d] = True
        for outcome in (0, 1):
            lp = H.syndrome_update_homogeneous(prior, pg, d, N)
            phi_eff_grid = (N - d) * ps                                 # ed:253, outside jit
            post = np.exp(np.asarray(H.phase_update(lp, phi_eff_grid, outcome)))
            ref = cf.ghz_code_probs_theorem(pN, phiN, pattern)[outcome]
            rec(f"B4 ED-FL composed N={N} g={gamma} M={M} d={d} v={outcome}", post_err(post, ref), tol32(N, M))
            if not X64 and d > 0 and outcome == 0:
                ref_exact_p = cf.ghz_code_probs_theorem(np.repeat(p_ref[:, None], N, axis=1), phiN, pattern)[outcome]
                print(f"      B4x  same vs exact-p reference: {post_err(post, ref_exact_p):.2e} (float32 p cancellation, ed:102-103)")
    pe = H.phi_eff_grid_ed(grid32, jnp.full(N, gamma, DT), jf(M))
    for outcome in (0, 1):
        post = np.exp(np.asarray(H.phase_update(prior, pe, outcome)))
        ref = cf.ghz_code_probs_theorem(pN, phiN, np.zeros(N, bool))[outcome]
        ref = ref / np.prod(1 - pN, axis=1)                             # PS: conditional on d = 0
        rec(f"B5 ED-PS phase_update N={N} g={gamma} M={M} v={outcome}", post_err(post, ref), tol32(N, M))
        post = np.exp(np.asarray(H.phase_update_vis(prior, pe, outcome, jf(np.log(0.9)))))
        ref = cf.ghz_code_probs_theorem(pN, phiN, np.zeros(N, bool), V=0.9)[outcome]
        ref = ref / np.prod(1 - pN, axis=1)
        rec(f"B6 ED-PS phase_update_vis V=0.9 N={N} g={gamma} M={M} v={outcome}", post_err(post, ref), tol32(N, M))

# ==========================================================================
print("\n=== C. combined theorem kernels (cb:84-163) vs closed_forms (theorem model) ===")
for Nc, gamma, M, sigma in itertools.product([3, 5], [0.01, 0.1], [1, 7], [0.01, 0.05]):
    Nt = 3 * Nc
    eps_k = rng.normal(0, sigma, Nt)
    om_k = grid[:, None] + eps_k[None, :]
    p_ref, phi_ref = cf.bitflip_params(om_k, gamma, M)
    gam32 = jnp.full(Nt, gamma, DT)
    eps32 = jf(eps_k)
    pq = H.p_grid_per_qubit(grid32, gam32, eps32, jf(M))
    phq = H.phi_grid_per_qubit(grid32, gam32, eps32, jf(M))
    pe = H.phi_eff_grid_cb(grid32, gam32, eps32, jf(M))
    rec(f"C1 p_grid_per_qubit Nc={Nc} g={gamma} M={M} s={sigma}", float(np.max(np.abs(np.asarray(pq) - p_ref))), tol32(1, M, 2e-6))
    rec(f"C2 phi_grid_per_qubit cos2phi Nc={Nc} g={gamma} M={M}",
        float(np.max(np.abs(np.cos(2 * np.asarray(phq)) - np.cos(2 * phi_ref)))), tol32(1, M))
    rec(f"C3 phi_eff_grid_cb cos Nc={Nc} g={gamma} M={M}",
        float(np.max(np.abs(np.cos(2 * np.asarray(pe)) - np.cos(2 * phi_ref.sum(1))))), tol32(Nt, M))
    for trial in range(2):
        mask = rng.random(Nt) < 0.3
        maskj = jf(mask.astype(np.float64))
        for outcome in (0, 1):
            lp = H.syndrome_update_per_qubit(prior, pq, maskj)
            phe = H.phi_eff_with_errors(phq, maskj) if mask.any() else pe
            post = np.exp(np.asarray(H.phase_update(lp, phe, outcome)))
            ref = cf.ghz_code_probs_theorem(p_ref, phi_ref, mask)[outcome]
            rec(f"C4 comb-FL composed Nc={Nc} g={gamma} M={M} d={mask.sum()} v={outcome}", post_err(post, ref), tol32(Nt, M))
            # marg: visibility exp(-2 M^2 N_clean sigma^2) on the coherent term (cb:376)
            N_clean = Nt - mask.sum()
            lv = -2.0 * M**2 * N_clean * sigma**2
            post = np.exp(np.asarray(H.phase_update_vis(lp, phe, outcome, jf(lv))))
            ref = cf.ghz_code_probs_theorem(p_ref, phi_ref, mask, V=np.exp(lv))[outcome]
            rec(f"C5 comb-FL marg composed Nc={Nc} g={gamma} M={M} d={mask.sum()} v={outcome}", post_err(post, ref), tol32(Nt, M))

# ==========================================================================
print("\n=== D. exact model: identities and new kernels vs closed_forms.ghz_code_probs / _mixture ===")
# D1: orchestrator's D/X per-block identity and the reduced (A, Z) form, float64, random points
def dx_identity(a, c, blocks, pattern, V):
    """P(s,+/-) = 1/4[prod D00 + prod D11 + 2V Re prod D01] +/- 1/4[Re prod X00 + Re prod X11 + 2V Re prod X01]."""
    pattern = np.asarray(pattern, bool)
    D00 = D11 = D01 = X00 = X11 = X01 = 1.0
    for b in blocks:
        b = np.asarray(b)
        E = pattern[b]; Eb = ~E
        def T0(x): return np.prod(np.where(x, c[b], a[b]))
        def T1(x): return np.prod(np.where(x, np.conj(a[b]), c[b]))
        d00 = d11 = d01 = x00 = x11 = x01 = 0.0
        for x, xb in ((E, Eb), (Eb, E)):
            d00 += abs(T0(x))**2; d11 += abs(T1(x))**2; d01 += np.conj(T0(x)) * T1(x)
            x00 += np.conj(T0(x)) * T0(xb); x11 += np.conj(T1(x)) * T1(xb); x01 += np.conj(T0(x)) * T1(xb)
        D00 *= d00; D11 *= d11; D01 *= d01; X00 *= x00; X11 *= x11; X01 *= x01
    diag = 0.25 * (D00 + D11 + 2 * V * np.real(D01))
    cross = 0.25 * (np.real(X00) + np.real(X11) + 2 * V * np.real(X01))
    return diag + cross, diag - cross

def reduced(p, phi, blocks, pattern, V):
    """Odd-N reduced form: 1/2 [prod A_b +/- V Re prod Z_b]."""
    pattern = np.asarray(pattern, bool)
    A = 1.0; Z = 1.0 + 0j
    for b in blocks:
        b = np.asarray(b); E = pattern[b]
        PE = np.prod(np.where(E, p[b], 1 - p[b])); PEb = np.prod(np.where(E, 1 - p[b], p[b]))
        PhE = np.sum(np.where(E, 0.0, phi[b])); PhEb = np.sum(np.where(E, phi[b], 0.0))
        A *= PE + PEb; Z *= PE * np.exp(2j * PhE) + PEb * np.exp(2j * PhEb)
    return 0.5 * (A + V * Z.real), 0.5 * (A - V * Z.real)

err_dx = err_red = err_mix = 0.0
for nb, Nc in itertools.product([1, 3], [3, 5]):
    Nt = nb * Nc
    blocks = [list(range(b * Nc, (b + 1) * Nc)) for b in range(nb)]
    for trial in range(20):
        om = OMEGA + rng.normal(0, 0.1, Nt)
        gamma = rng.choice([0.01, 0.1, 0.3]); M = int(rng.integers(1, 200)); V = rng.choice([1.0, 0.7])
        a, c = cf.single_qubit_amplitudes(om, gamma, M)
        p, phi = cf.bitflip_params(om, gamma, M)
        pattern = rng.random(Nt) < 0.4
        ref = cf.ghz_code_probs(a, c, blocks, pattern, V=V)
        dx = dx_identity(a, c, blocks, pattern, V)
        rd = reduced(p, phi, blocks, pattern, V)
        mix = cf.ghz_code_probs_mixture(p, phi, blocks, pattern, V=V)
        err_dx = max(err_dx, abs(dx[0] - ref[0]), abs(dx[1] - ref[1]))
        err_red = max(err_red, abs(rd[0] - ref[0]), abs(rd[1] - ref[1]))
        err_mix = max(err_mix, abs(mix[0] - ref[0]), abs(mix[1] - ref[1]))
rec("D1a orchestrator D/X identity vs ghz_code_probs (1,3 blocks; N=3,5; float64)", err_dx, 1e-13)
rec("D1b reduced (A,Z) odd-N form vs ghz_code_probs", err_red, 1e-13)
rec("D1c ghz_code_probs_mixture vs ghz_code_probs", err_mix, 1e-13)
# even N: the reduced form must fail (documented)
blocks = [[0, 1, 2, 3]]
om = OMEGA + rng.normal(0, 0.1, 4); a, c = cf.single_qubit_amplitudes(om, 0.1, 7); p, phi = cf.bitflip_params(om, 0.1, 7)
pat = np.array([True, False, False, False])
print(f"      even N=4 sanity: reduced-vs-exact diff = {abs(reduced(p, phi, blocks, pat, 1.0)[0] - cf.ghz_code_probs(a, c, blocks, pat)[0]):.3e} (expected nonzero)")

def kernel_p_amplitudes(a, c, pg, N):
    """float32 mode: rescale |a|, |c| to the kernel's own p (phases kept), so the
    reference isolates composition rounding from the ed:102-103 p cancellation."""
    if X64:
        return a, c
    p_k = np.asarray(pg, np.float64)[:, None] * np.ones((1, N))
    p_ex = np.abs(c) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = np.where(p_ex > 0, np.sqrt(p_k / np.maximum(p_ex, 1e-300)), 1.0)
        fa = np.sqrt((1 - p_k) / (1 - p_ex))
    return a * fa, c * fc


# D2: exact_ed_update
for N, gamma, M in itertools.product([3, 5, 7], [0.01, 0.1, 0.3], [1, 7, 123]):
    L = N // 2
    a, c = cf.single_qubit_amplitudes(grid[:, None] * np.ones((1, N)), gamma, M)
    pg = H.p_grid_homogeneous(grid32, jf(gamma), jf(M))
    ps = H.phi_grid_homogeneous(grid32, jf(gamma), jf(M))
    a, c = kernel_p_amplitudes(a, c, pg, N)
    tol = tol32(N, M)
    for d, (V, q) in itertools.product(range(L + 1), [(1.0, 0.0), (0.9, 0.01)]):
        pattern = np.zeros(N, bool); pattern[:d] = True
        logc = np.full(N + 1, -np.inf); logc[d] = 0.0; logc[N - d] = 0.0
        log_vis = N * np.log1p(-2 * q) + np.log(V)
        for outcome in (0, 1):
            post = np.exp(np.asarray(H.exact_ed_update(prior, pg, ps, jf(logc), jf(log_vis), outcome)))
            ref = cf.ghz_code_probs(a, c, [list(range(N))], pattern, V=V, q=q)[outcome]
            rec(f"D2 exact_ed_update N={N} g={gamma} M={M} d={d} V={V} q={q} v={outcome}", post_err(post, ref), tol)

# D3/D4: exact_blocks_update, 3 blocks, per-qubit offsets, with and without marg damping
for Nc, gamma, M, sigma in itertools.product([3, 5], [0.05, 0.3], [1, 7], [0.02]):
    Nt = 3 * Nc
    blocks = [list(range(b * Nc, (b + 1) * Nc)) for b in range(3)]
    eps_k = rng.normal(0, sigma, Nt)
    om_k = grid[:, None] + eps_k[None, :]
    a, c = cf.single_qubit_amplitudes(om_k, gamma, M)
    p_ref, phi_ref = cf.bitflip_params(om_k, gamma, M)
    gam32 = jnp.full(Nt, gamma, DT); eps32 = jf(eps_k)
    pq = H.p_grid_per_qubit(grid32, gam32, eps32, jf(M))
    phq = H.phi_grid_per_qubit(grid32, gam32, eps32, jf(M))
    for trial in range(2):
        pattern = rng.random(Nt) < 0.35
        X = np.stack([np.stack([pattern[b], ~pattern[b]]) for b in blocks]).astype(np.float64)   # (3, 2, Nc)
        logw = np.zeros((3, 2))
        for (V, q), outcome in itertools.product([(1.0, 0.0), (0.8, 0.01)], (0, 1)):
            log_vis = Nt * np.log1p(-2 * q) + np.log(V)
            post = np.exp(np.asarray(H.exact_blocks_update(prior, pq, phq, jf(X), jf(logw), jf(log_vis), jf(0.0), outcome)))
            ref = cf.ghz_code_probs(a, c, blocks, pattern, V=V, q=q)[outcome]
            rec(f"D3 exact_blocks_update Nc={Nc} g={gamma} M={M} d={pattern.sum()} V={V} q={q} v={outcome}",
                post_err(post, ref), tol32(Nt, M))
        # D4 marg damping: reference = mixture with exp(-coef * n_clean(x)) per consistent pattern
        coef = 2.0 * M**2 * sigma**2
        for outcome in (0, 1):
            post = np.exp(np.asarray(H.exact_blocks_update(prior, pq, phq, jf(X), jf(logw), jf(0.0), jf(coef), outcome)))
            ref = 0.0
            for x in cf._consistent_patterns(pattern, blocks):
                Pcl = np.prod(np.where(x, p_ref, 1 - p_ref), axis=1)
                Phc = np.sum(np.where(x, 0.0, phi_ref), axis=1)
                sgn = 1.0 if outcome == 0 else -1.0
                ref = ref + 0.5 * Pcl * (1 + sgn * np.exp(-coef * (Nt - x.sum())) * np.cos(2 * Phc))
            rec(f"D4 exact_blocks_update marg coef={coef:.3g} Nc={Nc} g={gamma} M={M} v={outcome}", post_err(post, ref), tol32(Nt, M))

# D5: q-aware ED (exact and theorem masks) vs closed_forms.qaware_probs
for N, gamma, M, q in itertools.product([3, 5], [0.05, 0.3], [1, 7], [0.01, 0.03]):
    L = N // 2
    a, c = cf.single_qubit_amplitudes(grid[:, None] * np.ones((1, N)), gamma, M)
    pg = H.p_grid_homogeneous(grid32, jf(gamma), jf(M))
    ps = H.phi_grid_homogeneous(grid32, jf(gamma), jf(M))
    a, c = kernel_p_amplitudes(a, c, pg, N)
    for trial in range(2):
        s_obs = (rng.random(N - 1) < 0.5).astype(int)
        for model in ("exact", "theorem"):
            logc = H.weight_log_coeffs(s_obs[None, :], q, N, theorem_L=(L if model == "theorem" else None))[0]
            V = 0.95
            log_vis = N * np.log1p(-2 * q) + np.log(V)
            for outcome in (0, 1):
                post = np.exp(np.asarray(H.exact_ed_update(prior, pg, ps, jf(logc), jf(log_vis), outcome)))
                ref = cf.qaware_probs(a, c, [list(range(N))], s_obs, q, V=V, model=model)[outcome]
                rec(f"D5 q-aware ED {model} N={N} g={gamma} M={M} q={q} s={s_obs.tolist()} v={outcome}", post_err(post, ref), tol32(N, M))

# D6: q-aware combined (3 blocks of N_code = 3) vs closed_forms.qaware_probs
for gamma, M, q in itertools.product([0.1], [1, 7], [0.01]):
    Nc, Nt = 3, 9
    blocks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    eps_k = rng.normal(0, 0.02, Nt)
    om_k = grid[:, None] + eps_k[None, :]
    a, c = cf.single_qubit_amplitudes(om_k, gamma, M)
    gam32 = jnp.full(Nt, gamma, DT); eps32 = jf(eps_k)
    pq = H.p_grid_per_qubit(grid32, gam32, eps32, jf(M))
    phq = H.phi_grid_per_qubit(grid32, gam32, eps32, jf(M))
    Xall = np.broadcast_to(H.all_patterns(Nc)[None], (3, 2**Nc, Nc)).astype(np.float64)
    s_obs = (rng.random(Nt - 3) < 0.4).astype(int)
    s_blocks = s_obs.reshape(3, Nc - 1)
    for model in ("exact", "theorem"):
        logw = np.stack([H.pattern_log_weights(s_blocks[b][None, :], q, Nc, theorem_L=(1 if model == "theorem" else None))[0]
                         for b in range(3)])
        V = 0.9; log_vis = Nt * np.log1p(-2 * q) + np.log(V)
        for outcome in (0, 1):
            post = np.exp(np.asarray(H.exact_blocks_update(prior, pq, phq, jf(Xall), jf(logw), jf(log_vis), jf(0.0), outcome)))
            ref = cf.qaware_probs(a, c, blocks, s_obs, q, V=V, model=model)[outcome]
            rec(f"D6 q-aware combined {model} g={gamma} M={M} q={q} v={outcome}", post_err(post, ref), tol32(Nt, M))

# D7: truth-side helpers vs closed_forms at random true parameters
err7 = 0.0
for nb, Nc in itertools.product([1, 3], [3, 5, 7]):
    Nt = nb * Nc
    blocks = [list(range(b * Nc, (b + 1) * Nc)) for b in range(nb)]
    S = 50
    Ms = rng.integers(1, 500, S)
    om = OMEGA + rng.normal(0, 0.03, Nt)
    gamma = 0.1
    p, phi = H.exact_params(om, np.full(Nt, gamma), Ms)
    pats = rng.random((S, Nt)) < 0.3
    pp = H.exact_p_plus(p, phi, pats, nb, V=0.9)
    for i in range(S):
        a, c = cf.single_qubit_amplitudes(om, gamma, int(Ms[i]))
        r = cf.ghz_code_probs(a, c, blocks, pats[i], V=0.9)
        err7 = max(err7, abs(pp[i] - r[0] / (r[0] + r[1])))
rec("D7 exact_p_plus (truth side) vs ghz_code_probs P(+|s)", err7, 1e-12)
# decode / syndromes
pats = rng.random((200, 15)) < 0.5
syn = H.block_syndromes(pats, 3)
dec = H.decode_min_weight(syn)
ok = np.array_equal(H.block_syndromes(dec, 3), syn) and np.all(dec.reshape(200, 3, 5).sum(2) <= 2)
rec("D8 decode_min_weight reproduces syndromes with weight <= L", 0.0 if ok else 1.0, 0.0)
lp = H.legacy_params_ed(OMEGA, np.full(3, 0.1), np.array([1, 7, 123]))
ex = H.exact_params(np.full(3, OMEGA), np.full(3, 0.1), np.array([1, 7, 123]))
rec("D9 legacy_params_ed p vs exact_params p", float(np.max(np.abs(lp[0] - ex[0]))), 1e-14)
rec("D9 legacy_params_ed cos(2phi) vs exact_params", float(np.max(np.abs(np.cos(2 * lp[1]) - np.cos(2 * ex[1])))), 1e-13)

# ==========================================================================
print("\n=== E. bare_adaptive fixed-point kernel ===")
m = 16; Gb = 2**m
q0, q1, q2, logr, logC = H.bare_adaptive_tables(m, 0.0)
qfull = (np.asarray(q2).astype(np.uint64) << np.uint64(32)) | (np.asarray(q1).astype(np.uint64) << np.uint64(16)) | np.asarray(q0).astype(np.uint64)
rec("E1 tables gamma=0: q_j == j << 32", 0.0 if np.array_equal(qfull, np.arange(Gb, dtype=np.uint64) << np.uint64(32)) else 1.0, 0.0)
# general path at gamma = 0 (no special case) must give the same limbs
r_, th_, C_ = cf.bare_polar(np.pi * np.arange(Gb) / Gb, 0.0, 1)
u_ = np.mod(th_ / (2 * np.pi), 1.0)
qg = np.rint(u_ * 2.0**48).astype(np.uint64) & np.uint64((1 << 48) - 1)
rec("E1b general polar path at gamma=0 rounds to j << 32", 0.0 if np.array_equal(qg, qfull) else 1.0, 0.0)


# legacy sim:40-74 verbatim
@jit
def ghz_log_visibility(gammas, n):
    mask = jnp.arange(gammas.shape[0]) < n
    log_terms = jnp.where(mask, jnp.log(jnp.maximum(1.0 - gammas, 1e-30)), 0.0)
    return jnp.sum(log_terms)


@jit
def ghz_bayesian_update(logpk, j_grid, gammas, n, outcome, grid_size):
    log_vn = ghz_log_visibility(gammas, n)
    vn = jnp.exp(log_vn)
    sign = jnp.where(outcome == 0, 1.0, -1.0)
    phase_idx = (n * j_grid) & (grid_size - 1)
    cos_2n = jnp.cos(2.0 * jnp.pi * phase_idx / grid_size)
    likelihood = (1.0 + sign * vn * cos_2n) / 2.0
    log_lik = jnp.log(jnp.maximum(likelihood, 1e-30))
    logpk = logpk + log_lik
    logpk = logpk - logsumexp(logpk)
    return logpk


j_grid = jnp.arange(Gb, dtype=jnp.int32)
gammas0 = jnp.zeros(16384, dtype=jnp.float32)
logpk0 = jnp.full(Gb, -m * jnp.log(2.0), dtype=jnp.float32)
if not X64:
    n_bad = 0
    ns_test = np.concatenate([[1, 2, 3, 10, 16383, 16384], rng.integers(1, 16385, 200)])
    for n in ns_test:
        n = int(n)
        for outcome in (0, 1):
            a_ = np.asarray(ghz_bayesian_update(logpk0, j_grid, gammas0, n, outcome, Gb))
            b_ = np.asarray(H.bare_adaptive_update(logpk0, q0, q1, q2, logr, logC, n, 0.0, 0.0, outcome))
            if not np.array_equal(a_.view(np.uint32), b_.view(np.uint32)):
                n_bad += 1
    rec("E2a bitwise vs legacy sim:59-74 at gamma=0, uniform prior, 206 n x 2 outcomes", float(n_bad), 0.0)
    # chained posterior
    la = lb = logpk0
    n_bad = 0
    for step in range(300):
        n = int(rng.integers(1, 16385)); outcome = int(rng.integers(0, 2))
        la = ghz_bayesian_update(la, j_grid, gammas0, n, outcome, Gb)
        lb = H.bare_adaptive_update(lb, q0, q1, q2, logr, logC, n, 0.0, 0.0, outcome)
        if not np.array_equal(np.asarray(la).view(np.uint32), np.asarray(lb).view(np.uint32)):
            n_bad += 1
    rec("E2b bitwise vs legacy, 300-step chained posterior", float(n_bad), 0.0)
    # E3: gamma > 0 likelihood vs closed form (the kernel is float32 by design: fixed-point angle, float32 exp)
    xs_pi = np.pi * np.arange(Gb) / Gb
    for gamma in (0.01, 0.05, 0.1):
        q0g, q1g, q2g, logrg, logCg = H.bare_adaptive_tables(m, gamma)
        for n in (1, 2, 3, 4, 100, 1001, 4096, 16384):
            for outcome in (0, 1):
                post = np.exp(np.asarray(H.bare_adaptive_update(logpk0, q0g, q1g, q2g, logrg, logCg, n, 0.0, 0.0, outcome)))
                ref = cf.bare_parity_prob(xs_pi, gamma, 1, n, outcome)
                if ref.max() < 1e-12:
                    continue
                rec(f"E3 bare_adaptive_update g={gamma} n={n} v={outcome} vs bare_parity_prob", post_err(post, ref), 2e-5)
        # aware visibilities: readout on the whole parity, prep on the coherent term only (even n has C^n)
        for n in (49, 50):
            post = np.exp(np.asarray(H.bare_adaptive_update(logpk0, q0g, q1g, q2g, logrg, logCg, n,
                                                            float(n * np.log(0.98)), float(np.log(0.9)), 0)))
            ref = cf.bare_parity_prob(xs_pi, gamma, 1, n, 0, q=0.01, V=0.9)
            rec(f"E4 bare_adaptive_update aware g={gamma} n={n} (V=0.9,q=0.01)", post_err(post, ref), 2e-5)
err = 0.0
for gamma in (0.0, 0.01, 0.1):
    for n in (1, 2, 3, 4, 5, 100, 16384):
        coh, pop = H.bare_parity_batch(OMEGA, gamma, 1, np.array([n]))
        err = max(err, abs(coh[0] + pop[0] - cf.bare_parity(OMEGA, gamma, 1, n)),
                  abs(coh[0] - cf.bare_parity_coherent(OMEGA, gamma, 1, n)),
                  abs(pop[0] - float(np.asarray(cf.bare_parity_population(OMEGA, gamma, 1, n)))))
rec("E5 bare_parity_batch (coherent, population) vs closed_forms (float64)", err, 1e-10)
pf = H.parity_flip_prob(0.01, np.array([1, 3, 100]))
rec("E6 parity_flip_prob n=1 equals q", abs(pf[0] - 0.01), 1e-15)

# ==========================================================================
print("\n=== F. hwindow kernel ===")
for N, sigma in itertools.product([12, 200, 800], [0.05, 0.1]):
    xsN = np.linspace(0, np.pi / N, Gb, endpoint=False)
    xsN32 = jf(xsN)
    prior0 = jnp.full(Gb, -m * np.log(2.0), dtype=DT)
    for theta, outcome in itertools.product([0.3, 4.0], (0, 1)):
        post = np.exp(np.asarray(H.hwindow_update(prior0, xsN32, N, jf(-2 * N * sigma**2), jf(theta), outcome)))
        ref = cf.hwindow_prob(xsN, N, sigma, theta, outcome)
        if ref.max() < 1e-12:
            continue
        rec(f"F1 hwindow_update N={N} s={sigma} theta={theta} v={outcome}", post_err(post, ref), tol32(1, 2 * N, 2e-7))

# ==========================================================================
print("\n=== SUMMARY ===")
n_fail = sum(1 for r in RESULTS if not r[3])
print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed")
for name, err, tol, ok, note in RESULTS:
    if not ok:
        print(f"  FAIL {name}: err={err:.3e} tol={tol:.0e} {note}")
worst = {}
for name, err, tol, ok, note in RESULTS:
    key = name.split(" ")[0]
    worst[key] = max(worst.get(key, 0.0), err)
print("worst error per check group:")
for k in sorted(worst):
    print(f"  {k:6s} {worst[k]:.3e}")
