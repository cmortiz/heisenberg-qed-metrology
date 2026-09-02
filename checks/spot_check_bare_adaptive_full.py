"""bare_adaptive gamma=0, seed 1, all 60 eps, model=theorem: compare with BOTH
legacy runs recorded in scripts/results/bare_ghz.csv (two rows per eps)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from eqsp.config import RunSpec, epsilon_grid
from eqsp import engine
from eqsp.protocols import get_protocol

d = pd.read_csv(os.path.join(REPO, "scripts/results/bare_ghz.csv"))
d = d[(d.seed == 1) & (d.gamma == 0) & (d.h == 0)].sort_index()  # each legacy run is a contiguous 60-row block
runs = [d.iloc[:60], d.iloc[60:]]
assert all(len(g) == 60 and g.epsilon.nunique() == 60 for g in runs), [len(g) for g in runs]
print("legacy runs (row blocks):", [(int(g.index[0]), int(g.index[-1]), g.timestamp.min()[:19], g.timestamp.max()[:19]) for g in runs])
spec = RunSpec("bare_adaptive", gamma=0.0, model="theorem", stop="legacy")
rows = engine.run_seed(get_protocol(spec, "cpu"), spec, 1, epsilon_grid(), "spot", "cpu", progress=True)
new = pd.DataFrame(rows)
stats = {"A": 0, "B": 0, "both": 0, "neither": 0}
for _, r in new.iterrows():
    a = runs[0][np.isclose(runs[0].epsilon, r.epsilon)].iloc[0]
    b = runs[1][np.isclose(runs[1].epsilon, r.epsilon)].iloc[0]
    ma = int(a.total_resources) == int(r.total_resources) and np.isclose(a.final_error, r.final_error, rtol=1e-5, atol=1e-7)
    mb = int(b.total_resources) == int(r.total_resources) and np.isclose(b.final_error, r.final_error, rtol=1e-5, atol=1e-7)
    key = "both" if (ma and mb) else "A" if ma else "B" if mb else "neither"
    stats[key] += 1
    if key in ("neither", "A", "B"):
        print(f"eps={r.epsilon:.4e} N={int(r.N):5d}: legacy A {int(a.total_resources):8d} {a.final_error:.4e} | legacy B {int(b.total_resources):8d} {b.final_error:.4e} | new {int(r.total_resources):8d} {r.final_error:.4e} -> {key}")
print("legacy A vs B agree on", int(sum(int(runs[0].sort_values('epsilon').total_resources.values[i]) == int(runs[1].sort_values('epsilon').total_resources.values[i]) for i in range(60))), "/ 60 eps")
print("new matches:", stats)
