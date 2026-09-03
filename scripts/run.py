#!/usr/bin/env python3
"""One CLI for every protocol (plan, 'CLI and suite').

  run.py PROTOCOL (--seed S | --seeds 1-40) [--gamma] [--h] [--L] [--sigma-eps]
         [--inference ps|fl] [--marg] [--marg-cap] [--q] [--k] [--F] [--estimator mismatched|aware]
         [--model exact|theorem] [--extended] [--eps-tar] [--N-grid auto|N1,N2,...]
         [--out FILE | --results-dir DIR] [--device auto|cpu|gpu] [--max-experiments N] [--quiet]

Default output: one shard per (protocol, config, seed) at
results/v2/shards/<protocol>/<config_key>_seed<S>.csv (repo-root results/).
``--device cpu`` sets JAX_PLATFORMS=cpu before any jax import; ``sequential``
never imports jax.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eqsp.config import RunSpec, epsilon_grid, window_N_grid   # noqa: E402  (no jax)
from eqsp.io import git_short_sha, shard_path, write_rows       # noqa: E402  (no jax)
from eqsp.protocols import PROTOCOLS, set_device                # noqa: E402  (no jax)

DEFAULT_RESULTS_DIR = REPO / "results" / "v2"


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EQSP numerics: run one protocol for one or more seeds")
    ap.add_argument("protocol", choices=PROTOCOLS)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--seeds", type=str, help="e.g. 1-40 or 1,2,7")
    ap.add_argument("--gamma", type=float, default=0.0, help="transverse field strength (Hamiltonian noise)")
    ap.add_argument("--h", type=float, default=0.0, help="relative std of heterogeneous gamma_k (legacy; 0)")
    ap.add_argument("--L", type=int, default=1, help="repetition-code half-distance, N_code = 2L+1")
    ap.add_argument("--sigma-eps", type=float, default=0.0, help="signal offset std (combined, sequential, hwindow)")
    ap.add_argument("--inference", choices=["fl", "ps"], default="fl")
    ap.add_argument("--marg", action="store_true", help="combined: per-shot offsets, marginalised inference")
    ap.add_argument("--marg-cap", action="store_true",
                    help="combined marg: cap the depth at the information-optimal M = 1/(2 sigma sqrt(N_total))")
    ap.add_argument("--q", type=float, default=0.0, help="readout flip probability per measured bit")
    ap.add_argument("--k", type=float, default=1.0, help="entangling-gate fidelity (prep visibility k^(n-1) F)")
    ap.add_argument("--F", type=float, default=1.0, help="single-qubit init fidelity")
    ap.add_argument("--estimator", choices=["mismatched", "aware"], default="mismatched")
    ap.add_argument("--model", choices=["exact", "theorem"], default="exact")
    ap.add_argument("--stop", choices=["posterior", "legacy"], default="posterior",
                    help="stopping rule: posterior (truth-free, 2 std <= eps, every update) or legacy (truth-based at legacy cadence)")
    ap.add_argument("--extended", action="store_true", help="extended eps range (legacy --extended)")
    ap.add_argument("--eps-tar", type=float, default=1e-4, help="hwindow: single target eps")
    ap.add_argument("--N-grid", type=str, default="auto", help="hwindow: 'auto' (12 log-spaced in [N*/16, 4N*]) or N1,N2,...")
    ap.add_argument("--out", type=str, default=None, help="append all rows to this CSV instead of per-seed shards")
    ap.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--max-experiments", type=int, default=None, help="override the protocol's max_experiments (tests)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def make_spec(args) -> RunSpec:
    return RunSpec(
        protocol=args.protocol, gamma=args.gamma, h=args.h, L=args.L, sigma=args.sigma_eps,
        inference=args.inference, marg=args.marg, marg_cap=args.marg_cap, q=args.q, k=args.k, F=args.F,
        estimator=args.estimator, model=args.model, extended=args.extended, stop=args.stop,
        eps_tar=(args.eps_tar if args.protocol == "hwindow" else None),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_device(args.device)                                 # before the first jax import
    from eqsp.protocols import get_protocol                 # noqa: E402

    spec = make_spec(args)
    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
    engine_version = git_short_sha(REPO)
    results_dir = Path(args.results_dir)
    log = (lambda *a, **k: None) if args.quiet else print

    def out_path(seed: int) -> Path:
        return Path(args.out) if args.out else shard_path(results_dir, spec, seed)

    t0 = time.time()
    n_rows = 0
    n_conv = 0

    def on_done(seed: int, rows: list[dict]) -> None:
        nonlocal n_rows, n_conv
        write_rows(out_path(seed), rows)
        n_rows += len(rows)
        c = sum(1 for r in rows if r["converged"])
        n_conv += c
        log(f"  seed {seed}: {c}/{len(rows)} converged, {time.time() - t0:.1f}s -> {out_path(seed)}")

    log(f"run.py {spec.protocol} mode={spec.mode_label()} seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) "
        f"gamma={spec.gamma} h={spec.h} L={spec.L} sigma={spec.sigma} q={spec.q} k={spec.k} F={spec.F} "
        f"estimator={spec.estimator} model={spec.model} engine={engine_version}")

    if spec.protocol == "sequential":
        p = get_protocol(spec, args.device)
        epsilons = epsilon_grid(spec.extended, spec.protocol)
        p.run_seeds(seeds, epsilons, engine_version, "none", progress=False, on_seed_done=on_done)
    else:
        import jax
        from eqsp import engine
        jax_version = jax.__version__
        log(f"  jax {jax_version} on {jax.devices()[0].platform}")
        if spec.protocol == "hwindow":
            Ns = window_N_grid(spec.sigma) if args.N_grid == "auto" else [int(x) for x in args.N_grid.split(",")]
            log(f"  N grid: {Ns}")
            for seed in seeds:
                rows: list[dict] = []
                for N in Ns:
                    spec_N = replace(spec, N_fixed=N)
                    p = get_protocol(spec_N, args.device)
                    if args.max_experiments:
                        p.cfg = replace(p.cfg, max_experiments=args.max_experiments)
                    rows.extend(engine.run_seed(p, spec_N, seed, [spec.eps_tar], engine_version, jax_version))
                on_done(seed, rows)
        else:
            p = get_protocol(spec, args.device)
            if args.max_experiments:
                p.cfg = replace(p.cfg, max_experiments=args.max_experiments)
            epsilons = epsilon_grid(spec.extended, spec.protocol)
            engine.run_seeds(p, spec, seeds, epsilons, engine_version, jax_version,
                             progress=not args.quiet, on_seed_done=on_done)
    log(f"done: {n_rows} rows, {n_conv} converged, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
