"""Regression harness against the committed v1 CSVs (diagnostic, never gating).

Two tiers, both write ``scripts/results/v2/regression_report.md``:

* exact tier (``-m regression``): seeds 1 and 2 of one cell per protocol,
  ``--model theorem``, at the 3 largest eps of the grid; row equality of
  ``total_resources, converged, final_error`` with the committed CSV.
  Bitwise agreement is expected only for sequential (pure NumPy); the float32
  XLA kernels were committed from a GPU build, so CPU differences are recorded,
  not judged.
* statistical tier (``-m regression``): runs only when merged v2 CSVs exist
  under ``scripts/results/v2/`` (or ``results/v2/``); per-seed alpha within
  2 SEM and converged fraction within 2 points, using a plain OLS of log T on
  log eps over converged rows of the full range with at least 10 points.

The always-on test at the bottom checks that the sequential protocol
reproduces committed ``sequential.csv`` rows exactly (seed 1, 3 largest eps).

The tests assert only that the harness ran and the report was written.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import platform
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from eqsp.config import RunSpec, epsilon_grid, grid_resolution  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
RUN_PY = os.path.join(SCRIPTS, "run.py")
COMMITTED = os.path.join(SCRIPTS, "results")
V2_DIRS = [os.path.join(SCRIPTS, "results", "v2"), os.path.join(REPO, "results", "v2")]
REPORT_DIR = V2_DIRS[0]
REPORT = os.path.join(REPORT_DIR, "regression_report.md")

EPS_GRID = epsilon_grid()
TOP3 = [float(e) for e in sorted(EPS_GRID)[-3:]]
SEEDS = [1, 2]
KEY_COLS = ["total_resources", "converged", "final_error"]

# one cell per protocol / legacy mode: (RunSpec, committed csv, committed row filter)
CELLS = [
    ("bare_adaptive_g0", RunSpec("bare_adaptive", gamma=0.0, model="theorem"), "bare_ghz.csv",
     dict(gamma=0.0, h=0.0, mode="ghz")),
    ("bare_fixed_L1_g0.05", RunSpec("bare_fixed", gamma=0.05, L=1, model="theorem"), "bare_hamiltonian.csv",
     dict(gamma=0.05, h=0.0, L=1, mode="bare_hamiltonian")),
    ("ed_fl_L1_g0.01", RunSpec("ed", gamma=0.01, L=1, inference="fl", model="theorem"), "ed_full_likelihood.csv",
     dict(gamma=0.01, h=0.0, L=1, mode="full_likelihood")),
    ("ed_ps_L1_g0.01", RunSpec("ed", gamma=0.01, L=1, inference="ps", model="theorem"), "ed_postselect.csv",
     dict(gamma=0.01, h=0.0, L=1, mode="error_detected")),
    ("combined_fl_L1_g0.01_s0.01", RunSpec("combined", gamma=0.01, L=1, sigma=0.01, inference="fl", model="theorem"),
     "combined_full_likelihood.csv", dict(gamma=0.01, h=0.0, sigma_epsilon=0.01, L=1, mode="combined_full_likelihood")),
    ("combined_ps_L1_g0.01_s0.01", RunSpec("combined", gamma=0.01, L=1, sigma=0.01, inference="ps", model="theorem"),
     "combined_postselect.csv", dict(gamma=0.01, h=0.0, sigma_epsilon=0.01, L=1, mode="combined")),
    ("combined_marg_L1_g0.01_s0.01", RunSpec("combined", gamma=0.01, L=1, sigma=0.01, inference="fl", marg=True, model="theorem"),
     "combined_full_likelihood_marg.csv", dict(gamma=0.01, h=0.0, sigma_epsilon=0.01, L=1, mode="combined_full_likelihood_marg")),
    ("sequential_L1_s0", RunSpec("sequential", L=1, sigma=0.0, model="theorem"), "sequential.csv",
     dict(sigma_epsilon=0.0, L=1, mode="sequential")),
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _spec_cli_args(spec: RunSpec):
    args = ["--gamma", repr(spec.gamma), "--L", str(spec.L), "--sigma-eps", repr(spec.sigma),
            "--model", spec.model, "--q", repr(spec.q), "--k", repr(spec.k), "--F", repr(spec.F),
            "--estimator", spec.estimator]
    if spec.protocol in ("ed", "combined"):
        args += ["--inference", spec.inference]
    if spec.marg:
        args += ["--marg"]
    if spec.h:
        args += ["--h", repr(spec.h)]
    return args


_HELP_CACHE = {}


def _run_py_help():
    if "help" not in _HELP_CACHE:
        if not os.path.exists(RUN_PY):
            _HELP_CACHE["help"] = ""
        else:
            r = subprocess.run([sys.executable, RUN_PY, "--help"], capture_output=True, text=True,
                               cwd=REPO, env={**os.environ, "JAX_PLATFORMS": "cpu"})
            _HELP_CACHE["help"] = r.stdout + r.stderr
    return _HELP_CACHE["help"]


def _eps_selection_flag():
    """Flag of run.py that restricts the eps grid, if it advertises one."""
    help_text = _run_py_help()
    m = re.search(r"(--eps-(?:idx|index|indices|subset|only|select|top|list|values?))\b", help_text)
    return m.group(1) if m else None


def _committed_rows(csv_name, match, seed, eps_list):
    df = pd.read_csv(os.path.join(COMMITTED, csv_name))
    for k, v in match.items():
        if k in df.columns:
            df = df[np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v)]
    df = df[df.seed == seed]
    out = {}
    for eps in eps_list:
        sub = df[np.isclose(df.epsilon, eps, rtol=1e-9, atol=0)]
        if len(sub):
            out[eps] = sub.iloc[-1]
    return out


def _rows_via_subprocess(spec, seed, eps_list, flag):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.csv")
        idx = [int(np.argmin(np.abs(EPS_GRID - e))) for e in eps_list]
        cmd = [sys.executable, RUN_PY, spec.protocol, "--seed", str(seed), *_spec_cli_args(spec),
               "--device", "cpu", "--out", out, "--quiet", flag, ",".join(str(i) for i in idx)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"}, timeout=1800)
        if r.returncode != 0 or not os.path.exists(out):
            return None, f"run.py failed (rc={r.returncode}): {r.stderr[-800:]}"
        df = pd.read_csv(out)
        rows = {}
        for eps in eps_list:
            sub = df[np.isclose(df.epsilon, eps, rtol=1e-9, atol=0)]
            if len(sub):
                rows[eps] = sub.iloc[-1]
        return rows, "subprocess run.py " + " ".join(cmd[2:])


def _rows_in_process(spec, seed, eps_list):
    """Fallback: drive the adapter with the engine loop directly (same per-eps
    path run.py takes)."""
    protocols = pytest.importorskip("eqsp.protocols")
    eio = pytest.importorskip("eqsp.io")
    p = protocols.get_protocol(spec, "cpu")
    m = p.grid_m(float(EPS_GRID.min()))
    p.begin_run(seed, m)
    rows = {}
    for eps in eps_list:
        rng = eio.eps_rng(seed, eps)
        if hasattr(p, "run_epsilon"):
            res = p.run_epsilon(eps, rng)
        else:
            engine = pytest.importorskip("eqsp.engine")
            res = engine.run_epsilon(p, eps, rng)
        ctr = getattr(res, "ctr", getattr(res, "counters", res))
        row = None
        if hasattr(p, "row"):
            try:
                row = p.row(ctr, bool(res.converged), float(res.error))
            except Exception:
                row = None
        if row is None:
            row = dict(total_resources=getattr(ctr, "resources", np.nan), converged=bool(res.converged),
                       final_error=float(res.error))
        rows[eps] = pd.Series(row)
    return rows, "in-process engine.run_epsilon"


def _compare(v1, v2):
    """Per-column equality flags for one (seed, eps) row pair."""
    out = {}
    out["total_resources"] = int(v1["total_resources"]) == int(v2["total_resources"])
    out["converged"] = bool(v1["converged"]) == bool(v2["converged"])
    e1, e2 = float(v1["final_error"]), float(v2["final_error"])
    out["final_error"] = math.isclose(e1, e2, rel_tol=1e-9, abs_tol=1e-12)
    return out


def _ols_alpha(df):
    """alpha = -slope of log T on log eps over converged rows (full range, min 10 points)."""
    sub = df[df.converged.astype(bool)]
    if len(sub) < 10:
        return np.nan
    x, y = np.log(sub.epsilon.values.astype(float)), np.log(sub.total_resources.values.astype(float))
    slope = np.polyfit(x, y, 1)[0]
    return -slope


def _cell_stats(df):
    alphas = np.array([_ols_alpha(g) for _, g in df.groupby("seed")])
    alphas = alphas[np.isfinite(alphas)]
    return dict(n_seeds=len(alphas), alpha=float(alphas.mean()) if len(alphas) else np.nan,
                sem=float(alphas.std(ddof=1) / math.sqrt(len(alphas))) if len(alphas) > 1 else np.nan,
                conv_frac=float(df.converged.astype(bool).mean()))


def _write_report(sections):
    os.makedirs(REPORT_DIR, exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    except Exception:
        sha = "unknown"
    try:
        import jax
        jv = jax.__version__
    except Exception:
        jv = "unavailable"
    head = [
        "# Regression report (diagnostic, not gating)",
        "",
        f"- generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- numerics HEAD: {sha}; jax: {jv}; python: {platform.python_version()}; platform: {platform.platform()}",
        "- baseline: committed CSVs under scripts/results/ (tag v1-jax-suite, GPU float32 XLA, May 2026)",
        "- exact tier compares total_resources (int), converged (bool), final_error (rel 1e-9)",
        "- statistical tier: per-seed alpha = -OLS slope of log T on log eps over converged rows, full range, >= 10 points;",
        "  cell alpha = mean over seeds, SEM = std/sqrt(n); flag |delta alpha| > 2 sqrt(SEM_v1^2 + SEM_v2^2) or |delta conv frac| > 0.02",
        "",
    ]
    with open(REPORT, "w") as f:
        f.write("\n".join(head + sections) + "\n")


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------
def exact_tier(cells=CELLS, seeds=SEEDS, eps_list=TOP3):
    flag = _eps_selection_flag()
    lines = ["## Exact tier", ""]
    if flag:
        lines.append(f"runner: `scripts/run.py` via subprocess (CPU), eps restricted with `{flag}`")
    else:
        lines.append("runner: in-process `engine.run_epsilon` (run.py advertises no eps-selection flag, or does not exist yet); "
                     "same per-eps code path as run.py")
    lines += ["", "| cell | seed | eps | v1 T | v2 T | v1 conv | v2 conv | v1 err | v2 err | match (T, conv, err) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    n_pairs = n_match = 0
    notes = []
    for name, spec, csv_name, match in cells:
        for seed in seeds:
            v1 = _committed_rows(csv_name, match, seed, eps_list)
            if not v1:
                notes.append(f"- {name} seed {seed}: no committed rows at the requested eps")
                continue
            try:
                if flag and os.path.exists(RUN_PY):
                    v2, how = _rows_via_subprocess(spec, seed, eps_list, flag)
                    if v2 is None:
                        notes.append(f"- {name} seed {seed}: {how}; falling back to in-process")
                        v2, how = _rows_in_process(spec, seed, eps_list)
                else:
                    v2, how = _rows_in_process(spec, seed, eps_list)
            except Exception as exc:  # adapter missing or interface mismatch: record, do not fail
                notes.append(f"- {name} seed {seed}: [BLOCKED: {type(exc).__name__}: {str(exc)[:200]}]")
                continue
            for eps in eps_list:
                if eps not in v1 or eps not in v2:
                    notes.append(f"- {name} seed {seed} eps {eps:.4g}: row missing on {'v1' if eps not in v1 else 'v2'} side")
                    continue
                cmp = _compare(v1[eps], v2[eps])
                n_pairs += 1
                n_match += all(cmp.values())
                lines.append(f"| {name} | {seed} | {eps:.4g} | {int(v1[eps]['total_resources'])} | {int(v2[eps]['total_resources'])} | "
                             f"{bool(v1[eps]['converged'])} | {bool(v2[eps]['converged'])} | {float(v1[eps]['final_error']):.6e} | "
                             f"{float(v2[eps]['final_error']):.6e} | {cmp['total_resources']}, {cmp['converged']}, {cmp['final_error']} |")
    lines += ["", f"rows compared: {n_pairs}; rows identical in all three columns: {n_match}", ""]
    if notes:
        lines += ["notes:"] + notes + [""]
    return lines, n_pairs


def statistical_tier(v2_dirs=V2_DIRS, cells=CELLS):
    lines = ["## Statistical tier", ""]
    found = False
    lines += ["| cell | v1 n_seeds | v1 alpha | v1 SEM | v2 n_seeds | v2 alpha | v2 SEM | delta | 2 sigma | v1 conv | v2 conv | flag |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    results = []
    for name, spec, csv_name, match in cells:
        v2_path = None
        for d in v2_dirs:
            cand = os.path.join(d, f"{spec.protocol}.csv")
            if os.path.exists(cand):
                v2_path = cand
                break
        if v2_path is None:
            continue
        found = True
        df2 = pd.read_csv(v2_path)
        sel = pd.Series(True, index=df2.index)
        for col, val in [("gamma", spec.gamma), ("L", spec.L), ("sigma_epsilon", spec.sigma), ("q", spec.q), ("k", spec.k), ("F", spec.F)]:
            if col in df2.columns:
                sel &= np.isclose(df2[col].astype(float), val)
        for col, val in [("mode", spec.mode_label()), ("model", spec.model), ("estimator", spec.estimator)]:
            if col in df2.columns:
                sel &= df2[col] == val
        df2 = df2[sel]
        if df2.empty:
            lines.append(f"| {name} | | | | 0 | | | | | | | no v2 rows |")
            continue
        df1 = pd.read_csv(os.path.join(COMMITTED, csv_name))
        for k, v in match.items():
            if k in df1.columns:
                df1 = df1[np.isclose(df1[k], v) if isinstance(v, float) else (df1[k] == v)]
        s1, s2 = _cell_stats(df1), _cell_stats(df2)
        delta = s2["alpha"] - s1["alpha"]
        two_sigma = 2 * math.sqrt(np.nan_to_num(s1["sem"]) ** 2 + np.nan_to_num(s2["sem"]) ** 2)
        dconv = s2["conv_frac"] - s1["conv_frac"]
        flag = []
        if abs(delta) > two_sigma:
            flag.append("alpha")
        if abs(dconv) > 0.02:
            flag.append("conv")
        results.append(dict(cell=name, s1=s1, s2=s2, delta=delta, two_sigma=two_sigma, dconv=dconv, flag=flag))
        lines.append(f"| {name} | {s1['n_seeds']} | {s1['alpha']:.4f} | {s1['sem']:.4f} | {s2['n_seeds']} | {s2['alpha']:.4f} | "
                     f"{s2['sem']:.4f} | {delta:+.4f} | {two_sigma:.4f} | {s1['conv_frac']:.3f} | {s2['conv_frac']:.3f} | "
                     f"{', '.join(flag) if flag else 'ok'} |")
    if not found:
        lines = ["## Statistical tier", "", "no merged v2 CSVs found under " + " or ".join(v2_dirs) + "; tier not run", ""]
    else:
        lines.append("")
    return lines, results


# --------------------------------------------------------------------------
# tests (regression marker)
# --------------------------------------------------------------------------
@pytest.mark.regression
def test_exact_tier_writes_report():
    pytest.importorskip("eqsp.protocols")
    lines, n_pairs = exact_tier()
    stat_lines, _ = statistical_tier()
    _write_report(lines + stat_lines)
    assert os.path.exists(REPORT)
    with open(REPORT) as f:
        text = f.read()
    assert "## Exact tier" in text and "## Statistical tier" in text
    print(f"\nregression report written to {REPORT} ({n_pairs} row pairs compared)")


@pytest.mark.regression
def test_statistical_tier_runs():
    lines, results = statistical_tier()
    assert lines[0] == "## Statistical tier"
    if not os.path.exists(REPORT):
        _write_report(["## Exact tier", "", "not run in this session", ""] + lines)
    print("\n".join(lines))


# --------------------------------------------------------------------------
# always-on: sequential reproduces committed rows exactly (pure NumPy)
# --------------------------------------------------------------------------
def test_sequential_reproduces_committed_rows_exactly():
    pytest.importorskip("eqsp.protocols.sequential")
    name, spec, csv_name, match = CELLS[-1]
    v1 = _committed_rows(csv_name, match, 1, TOP3)
    assert len(v1) == 3, "committed sequential.csv lacks seed 1 rows at the 3 largest eps"
    flag = _eps_selection_flag()
    if flag and os.path.exists(RUN_PY):
        v2, how = _rows_via_subprocess(spec, 1, TOP3, flag)
        if v2 is None:
            v2, how = _rows_in_process(spec, 1, TOP3)
    else:
        v2, how = _rows_in_process(spec, 1, TOP3)
    for eps in TOP3:
        assert eps in v2, (how, eps)
        cmp = _compare(v1[eps], v2[eps])
        assert all(cmp.values()), (eps, dict(v1[eps][KEY_COLS]), dict(v2[eps][KEY_COLS]), how)
