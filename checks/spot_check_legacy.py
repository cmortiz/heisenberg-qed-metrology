"""Verification 3: exact-tier spot check of the v2 engine against the committed
legacy CSVs (scripts/results/*.csv), seed 1, model="theorem", q=0, k=1, F=1.

JAX protocols: 3 largest eps, grid m taken from the full legacy eps range
(bh:50-55) so the grid matches the legacy run.  Sequential: all 60 eps, all 9
committed configs (must match exactly).  CPU run; the legacy CSVs were GPU runs
(last-bit XLA differences are possible for the JAX protocols).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from eqsp.config import RunSpec, epsilon_grid
from eqsp.io import make_row
from eqsp import engine
from eqsp.protocols import get_protocol
from eqsp.protocols.sequential import Sequential

SEED = 1
EPS_ALL = epsilon_grid()
EPS3 = EPS_ALL[-3:]
RES = os.path.join(REPO, "scripts", "results")
pd.set_option("display.width", 250)


def legacy_rows(csv, **filt):
    d = pd.read_csv(os.path.join(RES, csv))
    d = d[d.seed == SEED]
    for k, v in filt.items():
        d = d[np.isclose(d[k], v)] if isinstance(v, float) else d[d[k] == v]
    return d.drop_duplicates(subset=["epsilon"], keep="last").sort_values("epsilon")


def run_jax(spec, eps_list):
    p = get_protocol(spec, "cpu")
    m = p.grid_m(float(EPS_ALL.min()))            # legacy: m from the full range
    p.begin_run(SEED, m)
    rows = []
    for eps in eps_list:
        res = engine.run_epsilon(p, float(eps), p.rng_for(SEED, float(eps)))
        rows.append(make_row(spec, SEED, float(eps), res, "spot", "cpu"))
    return pd.DataFrame(rows)


def compare(name, leg, new, cols):
    print(f"\n=== {name} ===")
    print(f"{"epsilon":>10} | " + " | ".join(f"{c[1]:>34}" for c in cols) + " | match")
    n_bad = 0
    for _, r in new.iterrows():
        l = leg[np.isclose(leg.epsilon, r.epsilon)]
        if len(l) == 0:
            print(f"{r.epsilon:10.4e} | legacy row missing"); n_bad += 1; continue
        l = l.iloc[0]
        cells = []
        ok = True
        for c_new, c_leg in cols:
            a, b = r[c_new], l[c_leg]
            if isinstance(b, (bool, np.bool_)) or c_leg == "converged":
                same = bool(a) == bool(b)
            elif c_leg in ("final_error", "mean_d", "sum_F_obs", "sum_M_sq"):
                same = np.isclose(float(a), float(b), rtol=1e-5, atol=1e-7)
            else:
                same = np.isclose(float(a), float(b), rtol=0, atol=0.5)
            ok &= bool(same)
            cells.append(f"{str(b)[:16]:>16} -> {str(a)[:15]:>15}")
        n_bad += (not ok)
        print(f"{r.epsilon:10.4e} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'}")
    print(f"--> {len(new) - n_bad}/{len(new)} rows match")
    return n_bad


bad = 0
C_BASE = [("total_resources", "total_resources"), ("converged", "converged"), ("final_error", "final_error"), ("n_rounds", "n_rounds")]
C_ED = C_BASE[:3] + [("n_accepted", "n_accepted"), ("n_rejected", "n_rejected"), ("n_syndrome_used", "n_syndrome_used"),
                     ("mean_d", "mean_d"), ("sum_F_obs", "sum_F_obs"), ("sum_M_sq", "sum_M_sq")]   # legacy ED/combined CSVs have no n_rounds

# bare_adaptive vs bare_ghz.csv gamma=0 (legacy has no n_rounds column)
spec = RunSpec("bare_adaptive", gamma=0.0, model="theorem")
bad += compare("bare_adaptive gamma=0 vs bare_ghz.csv", legacy_rows("bare_ghz.csv", gamma=0.0, h=0.0), run_jax(spec, EPS3), C_BASE[:3])

# bare_fixed vs bare_hamiltonian.csv
spec = RunSpec("bare_fixed", gamma=0.05, L=1, model="theorem")
bad += compare("bare_fixed g=0.05 L=1 vs bare_hamiltonian.csv", legacy_rows("bare_hamiltonian.csv", gamma=0.05, L=1), run_jax(spec, EPS3), C_BASE)

# ed fl / ps
spec = RunSpec("ed", gamma=0.05, L=1, inference="fl", model="theorem")
bad += compare("ed fl g=0.05 L=1 vs ed_full_likelihood.csv", legacy_rows("ed_full_likelihood.csv", gamma=0.05, L=1), run_jax(spec, EPS3), C_ED)
spec = RunSpec("ed", gamma=0.05, L=1, inference="ps", model="theorem")
bad += compare("ed ps g=0.05 L=1 vs ed_postselect.csv", legacy_rows("ed_postselect.csv", gamma=0.05, L=1), run_jax(spec, EPS3), C_ED)

# combined fl
spec = RunSpec("combined", gamma=0.05, L=1, sigma=0.01, inference="fl", model="theorem")
bad += compare("combined fl g=0.05 L=1 s=0.01 vs combined_full_likelihood.csv",
               legacy_rows("combined_full_likelihood.csv", gamma=0.05, L=1, sigma_epsilon=0.01), run_jax(spec, EPS3), C_ED)

# sequential: all configs, all eps, exact
for L in (1, 2, 3):
    for sigma in (0.0, 0.01, 0.05):
        spec = RunSpec("sequential", L=L, sigma=sigma, model="theorem")
        new = pd.DataFrame(Sequential(spec).run_seed(SEED, EPS_ALL, "spot"))
        leg = legacy_rows("sequential.csv", L=L, sigma_epsilon=sigma)
        cols = C_BASE + [("M_max", "M_max_used")]
        n_bad = 0
        for _, r in new.iterrows():
            l = leg[np.isclose(leg.epsilon, r.epsilon)].iloc[0]
            ok = all((bool(r[a]) == bool(l[b])) if b == "converged" else np.isclose(float(r[a]), float(l[b]), rtol=1e-9, atol=1e-12) for a, b in cols)
            n_bad += (not ok)
            if not ok:
                print("   mismatch", dict(eps=r.epsilon, **{b: (l[b], r[a]) for a, b in cols}))
        print(f"sequential L={L} sigma={sigma}: {len(new) - n_bad}/{len(new)} rows identical (total_resources, converged, final_error, n_rounds, M_max_used)")
        bad += n_bad
print(f"\nTOTAL mismatching rows: {bad}")
