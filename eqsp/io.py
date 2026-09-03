"""RNG streams, shard paths, CSV rows (no jax import).

RNG conventions (plan, 'RNG plan'):
* per-eps stream   eps_rng(seed, eps)  = default_rng(seed + (int(eps*1e8) % 2**31))
                   (sim:239, bh:235, ed:363, cb:507, sq:66; note the precedence)
* run-level stream seed_rng(seed)      = default_rng(seed)               (cb:465, sq:141)
* new streams      stream_rng(seed, eps, *extra) = default_rng(SeedSequence([seed, eps_key, *extra]))
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CSV_COLUMNS, RunSpec


# --------------------------------------------------------------------------
# RNG
# --------------------------------------------------------------------------
def eps_key(eps: float) -> int:
    """int(eps*1e8) % 2**31, the legacy per-eps seed offset."""
    return int(eps * 1e8) % (2**31)


def eps_rng(seed: int, eps: float) -> np.random.Generator:
    return np.random.default_rng(seed + eps_key(eps))


def seed_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stream_rng(seed: int, eps: float, *extra: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), eps_key(eps), *[int(e) for e in extra]]))


# --------------------------------------------------------------------------
# Shard naming
# --------------------------------------------------------------------------
def _g(x: float) -> str:
    return format(float(x), "g")


def config_key(spec: RunSpec) -> str:
    """Filesystem-safe key of everything that identifies a cell except protocol
    and seed (hwindow: the N grid varies inside one shard, so N_fixed is excluded)."""
    parts = [
        spec.mode_label(),
        f"g{_g(spec.gamma)}", f"h{_g(spec.h)}", f"s{_g(spec.sigma)}", f"L{spec.L}",
        f"q{_g(spec.q)}", f"k{_g(spec.k)}", f"F{_g(spec.F)}",
        spec.estimator, spec.model,
    ]
    if spec.extended:
        parts.append("ext")
    if spec.stop == "legacy":
        parts.append("legacy")
    if spec.protocol == "hwindow":
        parts.append(f"eps{_g(spec.eps_tar)}")
    return "_".join(parts)


def shard_path(results_dir, spec: RunSpec, seed: int) -> Path:
    return Path(results_dir) / "shards" / spec.protocol / f"{config_key(spec)}_seed{seed}.csv"


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------
def make_row(spec: RunSpec, seed: int, eps: float, result, engine_version: str, jax_version: str) -> dict:
    """One CSV row with every CSV_COLUMNS key.

    ``result`` is an ``engine.EpsResult`` (converged, error, ctr, extra); the
    protocol's ``extra`` dict (N, M_max, grid_m, visibility, ...) overrides the
    defaults.  Grouping keys are always written from the spec (never NaN, so
    pandas groupby keeps the rows); other columns are NaN where not applicable.
    """
    nan = float("nan")
    row = {c: nan for c in CSV_COLUMNS}
    ctr = result.ctr
    row.update({
        "protocol": spec.protocol,
        "mode": spec.mode_label(),
        "seed": int(seed),
        "epsilon": float(eps),
        "gamma": float(spec.gamma),
        "h": float(spec.h),
        "sigma_epsilon": float(spec.sigma),
        "L": int(spec.L),
        "q": float(spec.q),
        "k": float(spec.k),
        "F": float(spec.F),
        "estimator": spec.estimator,
        "model": spec.model,
        "total_resources": int(ctr.resources),
        "converged": bool(result.converged),
        "final_error": float(result.error),
        "n_rounds": int(ctr.n_rounds),
        "n_updates": int(ctr.n_updates),
        "n_accepted": int(ctr.n_accepted),
        "n_rejected": int(ctr.n_rejected),
        "acceptance_rate": (ctr.n_accepted / (ctr.n_accepted + ctr.n_rejected)) if (ctr.n_accepted + ctr.n_rejected) > 0 else nan,
        "n_syndrome_used": int(ctr.n_syndrome_used),
        "mean_d": (ctr.sum_d / ctr.n_updates) if ctr.n_updates > 0 else nan,   # NaN, like every column with no shots behind it
        "sum_F_obs": float(ctr.sum_F_obs),
        "sum_M_sq": float(ctr.sum_M_sq),
        "M_exec_max": int(ctr.M_exec_max) if ctr.M_exec_max > 0 else nan,
        "stop_rule": spec.stop,
        "posterior_std": ctr.posterior_std,
        "engine_version": engine_version,
        "jax_version": jax_version,
        "timestamp": pd.Timestamp.now().isoformat(),
    })
    for key, val in result.extra.items():
        if key not in row:
            raise KeyError(f"row field {key!r} is not a CSV column")
        row[key] = val
    return row


def write_rows(path, rows: list[dict]) -> None:
    """Append rows under an exclusive flock; header only on an empty file;
    float format %.10e (sim:291-292)."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            needs_header = f.tell() == 0
            df.to_csv(f, header=needs_header, index=False, float_format="%.10e")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def git_short_sha(repo_dir) -> str:
    """Numerics git short SHA for the engine_version column ('unknown' if unavailable)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_dir),
                             capture_output=True, text=True, timeout=10)   # 10 s: git on a warm repo answers in ms; the column is provenance, never worth blocking a run
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=str(repo_dir), capture_output=True, text=True, timeout=10)
            return sha + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:   # no git binary, not a repo, or the timeout: the column records "unknown" rather than aborting the run
        pass
    return "unknown"
