#!/usr/bin/env python3
"""Suite runner: expands the suite table (groups 1-8), runs one process per
(protocol, config, seed) in two thread pools, resumes by skipping existing
shards, and merges shards into results/v2/<protocol>.csv.

  suite.py [--seeds 1-40] [--only 1,3,7] [--quick] [--dry-run] [--merge] [--label NAME]
           [--stamp-engine-version] [--gpu-slots 22] [--cpu-slots 64] [--n-gpus auto|N]
           [--mem-fraction 0.08] [--cpu-protocols sequential[,bare_adaptive,hwindow]] [--results-dir DIR]

GPU pool: slot s runs on CUDA_VISIBLE_DEVICES = s % n_gpus with
XLA_PYTHON_CLIENT_PREALLOCATE=true, XLA_PYTHON_CLIENT_MEM_FRACTION=0.08 (11 slots per
card = 88% preallocated; 12 slots per card OOM at CUBIN load, chupacabras 2026-09-02) and a
shared JAX_COMPILATION_CACHE_DIR; one warm-up process per GPU per distinct
(protocol, grid m) populates the cache before the pool opens.  CPU pool:
OMP_NUM_THREADS=1, single-thread XLA, --device cpu.  Without GPUs everything
runs in the CPU pool.  Logs: scripts/logs/<label>.log, scripts/logs/<label>/<job>.log,
scripts/logs/<label>.joblog.tsv (exit codes).  No jax import in this process.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eqsp.config import GROUP_KEYS, RunSpec, grid_resolution, window_N_grid   # noqa: E402
from eqsp.io import config_key, shard_path                                    # noqa: E402

RUN_PY = REPO / "scripts" / "run.py"
LOG_DIR = REPO / "scripts" / "logs"
DEFAULT_RESULTS_DIR = REPO / "results" / "v2"
N_EPS = 60

# Per-group parameter grids
ED_GRID = [(1, 0.01), (1, 0.05), (1, 0.10), (2, 0.01), (2, 0.05), (2, 0.10), (3, 0.01), (3, 0.05), (3, 0.10)]
COMB_GRID = [(1, 0.01, 0.01), (1, 0.05, 0.01), (1, 0.10, 0.01), (1, 0.01, 0.05), (1, 0.05, 0.05), (1, 0.10, 0.05),
             (2, 0.01, 0.01), (2, 0.05, 0.01), (2, 0.10, 0.01), (2, 0.01, 0.05), (2, 0.05, 0.05), (2, 0.10, 0.05)]
BARE_HAM_GRID = ED_GRID
SEQ_GRID = [(1, 0.0), (1, 0.01), (1, 0.05), (2, 0.0), (2, 0.01), (2, 0.05), (3, 0.0), (3, 0.01), (3, 0.05)]
MARG_CONFIGS = [(1, 0.01, 0.01)]                     # group 4: marginalised inference, uncapped and depth-capped
BARE_ADAPTIVE_GAMMAS = [0.0, 0.01, 0.05, 0.10]
WINDOW_SIGMAS = [0.05, 0.10]
# hwindow target: eps_tar = 4e-2 sigma^2 = 2e-2 / N*, so the sweep is the same in the scaled variable N/N*
# (shots needed ~ (N* eps)^-2 at both ends) and one period pi/N stays >= 20 x tolerance up to N = 4 N*.
# sigma 0.05 -> 1e-4 (N* = 200), sigma 0.10 -> 4e-4 (N* = 50).
def WINDOW_EPS(sigma: float) -> float:
    return 4e-2 * sigma**2
READOUT_QS = [1e-3, 1e-2, 3e-2]
PREP_KS = [0.99, 0.995, 0.999]
PREP_F = 0.99
NOISE_GAMMA = 0.10
NOISE_SIGMA = 0.01
ESTIMATORS = ["mismatched", "aware"]


@dataclass
class Job:
    spec: RunSpec
    seed: int
    group: int
    n_runs: int          # rows this job produces (60 eps, or the window N count)

    @property
    def name(self) -> str:
        return f"g{self.group}_{self.spec.protocol}_{config_key(self.spec)}_seed{self.seed}"

    @property
    def grid_m(self) -> int:
        s = self.spec
        if s.protocol in ("bare_adaptive", "hwindow"):
            return 16
        eps_min = 1e-6 if s.extended else 1e-4
        return grid_resolution(s.N_total, eps_min)

    def shard(self, results_dir: Path) -> Path:
        return shard_path(results_dir, self.spec, self.seed)


# --------------------------------------------------------------------------
# Suite expansion
# --------------------------------------------------------------------------
def suite_configs(quick: bool) -> dict[int, list[RunSpec]]:
    """Group number -> list of RunSpec (one per (protocol, config))."""
    g: dict[int, list[RunSpec]] = {}
    g[1] = [RunSpec("bare_adaptive", gamma=x) for x in BARE_ADAPTIVE_GAMMAS]
    g[2] = [RunSpec("bare_fixed", gamma=x, L=L) for L, x in BARE_HAM_GRID]
    # matched-depth baseline (2026-10-16): the legacy cells above draw M ~ U{1..floor(1/eps)}
    # while ED/combined draw floor(1/(eps N)) at the same per-shot charge N*M, so the published
    # prefactor gap carried an N-fold schedule inflation.  These cells rerun the baseline on the
    # ED schedule; gamma = 0 is the calibration row, where the ratio to ED-FL should be ~ 1.
    g[2] += [RunSpec("bare_fixed", gamma=x, L=L, matched_depth=True) for L, x in BARE_HAM_GRID]
    g[2] += [RunSpec("bare_fixed", gamma=0.0, L=L, matched_depth=True) for L in (1, 2, 3)]
    g[3] = [RunSpec("ed", gamma=x, L=L, inference=inf) for L, x in ED_GRID for inf in ("fl", "ps")]
    g[3] += [RunSpec("ed", gamma=0.0, L=L, inference="fl") for L in (1, 2, 3)]   # gamma = 0 calibration row
    g[4] = ([RunSpec("combined", gamma=x, L=L, sigma=s, inference=inf) for L, x, s in COMB_GRID for inf in ("fl", "ps")]
            + [RunSpec("combined", gamma=x, L=L, sigma=s, inference="fl", marg=True) for L, x, s in MARG_CONFIGS]
            + [RunSpec("combined", gamma=x, L=L, sigma=s, inference="fl", marg=True, marg_cap=True)
               for L, x, s in MARG_CONFIGS])
    g[5] = [RunSpec("sequential", L=L, sigma=s) for L, s in SEQ_GRID]
    g[6] = [RunSpec("hwindow", sigma=s, eps_tar=WINDOW_EPS(s)) for s in WINDOW_SIGMAS]
    g[7] = []
    for q in READOUT_QS:
        for est in ESTIMATORS:
            g[7].append(RunSpec("bare_adaptive", gamma=NOISE_GAMMA, q=q, estimator=est))
            g[7] += [RunSpec("ed", gamma=NOISE_GAMMA, L=L, inference="fl", q=q, estimator=est) for L in (1, 2, 3)]
            g[7] += [RunSpec("combined", gamma=NOISE_GAMMA, L=L, sigma=NOISE_SIGMA, inference="fl", q=q, estimator=est)
                     for L in (1, 2)]
        g[7] += [RunSpec("sequential", L=L, sigma=0.0, q=q) for L in (1, 2, 3)]          # majority vote: estimator-agnostic
    g[8] = []
    for k in PREP_KS:
        for est in ESTIMATORS:
            g[8].append(RunSpec("bare_adaptive", gamma=NOISE_GAMMA, k=k, F=PREP_F, estimator=est))
            g[8] += [RunSpec("ed", gamma=NOISE_GAMMA, L=L, inference="fl", k=k, F=PREP_F, estimator=est) for L in (1, 2, 3)]
            g[8] += [RunSpec("combined", gamma=NOISE_GAMMA, L=L, sigma=NOISE_SIGMA, inference="fl", k=k, F=PREP_F, estimator=est)
                     for L in (1, 2)]
    g[8] += [RunSpec("sequential", L=L, sigma=0.0, F=PREP_F) for L in (1, 2, 3)]         # F only (product state)
    if quick:
        g[1] = [RunSpec("bare_adaptive", gamma=0.05)]
        g[2] = [RunSpec("bare_fixed", gamma=0.05, L=1)]
        g[3] = [RunSpec("ed", gamma=0.05, L=1, inference="fl")]
        g[4] = [RunSpec("combined", gamma=0.05, L=1, sigma=0.01, inference="fl")]
        g[5] = [RunSpec("sequential", L=1, sigma=0.0)]
        g[6] = [RunSpec("hwindow", sigma=0.05, eps_tar=WINDOW_EPS(0.05))]
        g[7] = [RunSpec("ed", gamma=NOISE_GAMMA, L=1, inference="fl", q=1e-2, estimator="aware")]
        g[8] = [RunSpec("ed", gamma=NOISE_GAMMA, L=1, inference="fl", k=0.99, F=PREP_F, estimator="aware")]
    return g


def expand_jobs(groups: list[int], seeds: list[int], quick: bool) -> list[Job]:
    cfgs = suite_configs(quick)
    jobs: list[Job] = []
    for grp in groups:
        for spec in cfgs[grp]:
            n_runs = len(window_N_grid(spec.sigma)) if spec.protocol == "hwindow" else N_EPS
            for seed in seeds:
                jobs.append(Job(spec, seed, grp, n_runs))
    return jobs


def job_args(job: Job) -> list[str]:
    s = job.spec
    args = [s.protocol, "--seed", str(job.seed), "--gamma", repr(s.gamma), "--L", str(s.L),
            "--sigma-eps", repr(s.sigma), "--inference", s.inference, "--q", repr(s.q), "--k", repr(s.k),
            "--F", repr(s.F), "--estimator", s.estimator]
    if s.marg:
        args.append("--marg")
    if s.marg_cap:
        args.append("--marg-cap")
    if s.matched_depth:
        args.append("--matched-depth")
    if s.extended:
        args.append("--extended")
    if s.protocol == "hwindow":
        args += ["--eps-tar", repr(s.eps_tar), "--N-grid", "auto"]
    return args


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def detect_gpus() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except Exception:
        return 0


class Runner:
    def __init__(self, args, label: str, n_gpus: int, cache_dir: Path):
        self.args = args
        self.label = label
        self.n_gpus = n_gpus
        self.cache_dir = cache_dir
        self.results_dir = Path(args.results_dir)
        self.job_log_dir = LOG_DIR / label
        self.job_log_dir.mkdir(parents=True, exist_ok=True)
        self.main_log = open(LOG_DIR / f"{label}.log", "a")
        self.joblog = open(LOG_DIR / f"{label}.joblog.tsv", "a")
        if self.joblog.tell() == 0:
            self.joblog.write("job\tpool\tslot\texit_code\tseconds\tshard\n")
        self.lock = threading.Lock()
        self.failed: list[str] = []

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
        with self.lock:
            print(line, flush=True)
            self.main_log.write(line + "\n")
            self.main_log.flush()

    def env_for(self, pool: str, slot: int) -> dict:
        env = os.environ.copy()
        env["JAX_COMPILATION_CACHE_DIR"] = str(self.cache_dir)
        if pool == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(slot % self.n_gpus)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.args.mem_fraction)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
        return env

    def run_job(self, job: Job, pool: str, slot: int, extra: list[str] | None = None,
                results_dir: Path | None = None, tag: str = "") -> int:
        cmd = [sys.executable, str(RUN_PY)] + job_args(job) + [
            "--device", "gpu" if pool == "gpu" else "cpu",
            "--results-dir", str(results_dir or self.results_dir), "--quiet"] + (extra or [])
        t0 = time.time()
        with open(self.job_log_dir / f"{job.name}{tag}.log", "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=self.env_for(pool, slot), cwd=str(REPO)).returncode
        secs = time.time() - t0
        with self.lock:
            self.joblog.write(f"{job.name}{tag}\t{pool}\t{slot}\t{rc}\t{secs:.1f}\t{job.shard(results_dir or self.results_dir)}\n")
            self.joblog.flush()
        return rc

    def run_pool(self, jobs: list[Job], pool: str, n_slots: int) -> None:
        if not jobs:
            return
        slots: Queue = Queue()
        for s in range(n_slots):
            slots.put(s)
        done = [0]
        total = len(jobs)

        def work(job: Job) -> None:
            slot = slots.get()
            try:
                rc = self.run_job(job, pool, slot)
            finally:
                slots.put(slot)
            with self.lock:
                done[0] += 1
                n = done[0]
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            if rc != 0:
                with self.lock:
                    self.failed.append(job.name)
            self.log(f"{pool} slot {slot} [{n}/{total}] {job.name}: {status}")

        self.log(f"=== {pool} pool: {total} jobs on {n_slots} slots ===")
        with ThreadPoolExecutor(max_workers=n_slots) as ex:
            list(ex.map(work, jobs))

    def warm_up(self, gpu_jobs: list[Job]) -> None:
        """One short process per GPU per distinct (protocol, grid m) to populate the
        compilation cache before the pool opens."""
        classes: dict[tuple, Job] = {}
        for job in gpu_jobs:
            classes.setdefault((job.spec.protocol, job.grid_m), job)
        if not classes:
            return
        tmp = self.results_dir / "_warmup"
        self.log(f"=== warm-up: {len(classes)} (protocol, m) classes x {self.n_gpus} GPUs ===")

        def warm(gpu: int) -> None:
            for key, job in classes.items():
                # 20 experiments: enough to trigger every jitted kernel of the adapter and fill the persistent
                # compilation cache; the shard goes to _warmup and is discarded
                rc = self.run_job(job, "gpu", gpu, extra=["--max-experiments", "20"], results_dir=tmp, tag=f".warmup_gpu{gpu}")
                self.log(f"warm-up gpu {gpu} {key}: {'ok' if rc == 0 else f'FAILED rc={rc}'}")

        with ThreadPoolExecutor(max_workers=self.n_gpus) as ex:
            list(ex.map(warm, range(self.n_gpus)))
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Provenance: engine_version reconstruction
# --------------------------------------------------------------------------
# The chupacabras deploy is an rsync of the working tree, so the runs had no git
# metadata and every row of the v2 suite carries engine_version = "unknown".  The
# table below is the commit history of the engine (commits touching eqsp/,
# scripts/run.py or scripts/suite.py) over the run, in the server's clock zone
# (-0600), which is also the zone of the naive ISO timestamps in the CSV rows.
# ``engine_version_at`` maps a row timestamp to the last such commit at or before
# it; the assumption is that a deploy followed each commit, so a row computed at
# time t ran the code of the newest engine commit at or before t.
ENGINE_VERSION_BY_TIME = [
    ("2026-09-02 01:33:51", "f4c574e"),   # engine, protocol adapters, run.py, suite.py
    ("2026-09-02 01:56:29", "a18a0ea"),   # posterior stopping rule
    ("2026-09-02 02:14:39", "7361f24"),   # hwindow eps_tar, analyze keys
    ("2026-09-02 02:29:44", "715a7f9"),   # hwindow device-side chunked loop
    ("2026-09-02 03:02:42", "870441a"),   # per-shot preparation phase, one-period priors
    ("2026-09-02 16:32:53", "41738d1"),   # P_MAX float32 clip (NaN posterior fix)
    ("2026-09-02 19:01:32", "c4daba1"),   # combined marg_cap
    ("2026-09-02 21:09:03", "b020022"),   # analyze/plot/paper_numbers audit fixes
]
UNKNOWN = "unknown"


def engine_version_at(timestamp: str) -> str:
    """Last engine commit at or before ``timestamp`` (naive ISO, server zone);
    UNKNOWN for a timestamp older than the first entry or unparseable."""
    try:
        t = dt.datetime.fromisoformat(str(timestamp))
    except ValueError:
        return UNKNOWN
    out = UNKNOWN
    for stamp, sha in ENGINE_VERSION_BY_TIME:
        if t >= dt.datetime.fromisoformat(stamp):
            out = sha
        else:
            break
    return out


def stamp_engine_version(df) -> int:
    """Fill engine_version in place wherever it is UNKNOWN; returns the row count changed."""
    if "engine_version" not in df.columns or "timestamp" not in df.columns:
        return 0
    mask = df["engine_version"].astype(str) == UNKNOWN
    n = int(mask.sum())
    if n:
        df.loc[mask, "engine_version"] = df.loc[mask, "timestamp"].map(engine_version_at)
    return n


def stamp_engine_version_files(results_dir: Path, log=print) -> None:
    """Rewrite results_dir/<protocol>.csv in place, replacing only the
    engine_version field of rows that carry UNKNOWN (every other byte is copied
    through verbatim)."""
    import csv
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        if not rows or "engine_version" not in rows[0]:
            continue
        iv, it = rows[0].index("engine_version"), rows[0].index("timestamp")
        n = 0
        for r in rows[1:]:
            if r[iv] == UNKNOWN:
                r[iv] = engine_version_at(r[it])
                n += 1
        with path.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        log(f"stamped {n} rows of {path} ({len(rows) - 1} total)")


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------
def merge(results_dir: Path, log=print) -> None:
    import pandas as pd
    shards_root = results_dir / "shards"
    if not shards_root.exists():
        log(f"no shards under {shards_root}")
        return
    for proto_dir in sorted(p for p in shards_root.iterdir() if p.is_dir()):
        files = sorted(proto_dir.glob("*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        keys = GROUP_KEYS + ["seed", "epsilon"] + (["N"] if proto_dir.name == "hwindow" else [])
        n0 = len(df)
        df = df.sort_values(keys + ["timestamp"], kind="mergesort")
        df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys, kind="mergesort")
        n_stamped = stamp_engine_version(df)     # rows run from a copy without git metadata
        out = results_dir / f"{proto_dir.name}.csv"
        df.to_csv(out, index=False, float_format="%.10e")
        log(f"merged {len(files)} shards -> {out}: {len(df)} rows ({n0 - len(df)} duplicates dropped, "
            f"{n_stamped} engine_version stamped by commit time)")


# --------------------------------------------------------------------------
def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EQSP v2 suite runner")
    ap.add_argument("--seeds", default="1-40")
    ap.add_argument("--only", default="1,2,3,4,5,6,7,8", help="comma-separated group numbers")
    ap.add_argument("--quick", action="store_true", help="4 seeds, one config per group")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge shards into results/v2/<protocol>.csv and exit")
    ap.add_argument("--stamp-engine-version", action="store_true",
                    help="fill engine_version = unknown in results/v2/<protocol>.csv from the row timestamp and exit")
    ap.add_argument("--label", default=None)
    ap.add_argument("--gpu-slots", type=int, default=22)
    ap.add_argument("--cpu-slots", type=int, default=64)
    ap.add_argument("--n-gpus", default="auto")
    ap.add_argument("--mem-fraction", type=float, default=0.08)
    ap.add_argument("--cpu-protocols", default="sequential", help="protocols routed to the CPU pool")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    if args.stamp_engine_version:
        stamp_engine_version_files(results_dir)
        return 0
    if args.merge:
        merge(results_dir)
        return 0

    seeds = [1, 2, 3, 4] if args.quick else parse_seeds(args.seeds)
    groups = [int(x) for x in args.only.split(",") if x.strip()]
    jobs = expand_jobs(groups, seeds, args.quick)
    n_gpus = detect_gpus() if args.n_gpus == "auto" else int(args.n_gpus)
    cpu_protocols = set(x.strip() for x in args.cpu_protocols.split(",") if x.strip())
    if n_gpus == 0:
        cpu_protocols = {"bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow"}

    pending = [j for j in jobs if not j.shard(results_dir).exists()]
    skipped = len(jobs) - len(pending)
    gpu_jobs = [j for j in pending if j.spec.protocol not in cpu_protocols]
    cpu_jobs = [j for j in pending if j.spec.protocol in cpu_protocols]

    n_cfg = {g: len(set(config_key(j.spec) for j in jobs if j.group == g)) for g in groups}
    n_runs = sum(j.n_runs for j in jobs)
    print(f"suite: groups {groups}; seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}); quick={args.quick}")
    grids = "; ".join(f"sigma={s:g}: {window_N_grid(s)}" for s in WINDOW_SIGMAS)
    print(f"  configs per group: {n_cfg}  total configs {sum(n_cfg.values())} "
          f"(hwindow counts one config per sigma; N grids {grids})")
    print(f"  jobs: {len(jobs)} total, {skipped} already have shards, {len(pending)} pending "
          f"({len(gpu_jobs)} GPU pool, {len(cpu_jobs)} CPU pool); rows to produce: {n_runs}")
    print(f"  GPUs: {n_gpus}; gpu slots {args.gpu_slots}; cpu slots {args.cpu_slots}; results {results_dir}")
    if args.dry_run:
        for j in jobs:
            pool = "cpu" if j.spec.protocol in cpu_protocols else "gpu"
            print(f"  [{pool}] {'skip ' if j.shard(results_dir).exists() else ''}{j.name}  ->  run.py {' '.join(job_args(j))}")
        return 0

    label = args.label or f"suite_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = LOG_DIR / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args, label, n_gpus, cache_dir)
    runner.log(f"suite {label}: {len(pending)} pending jobs ({len(gpu_jobs)} gpu, {len(cpu_jobs)} cpu), {skipped} skipped")
    t0 = time.time()
    if gpu_jobs and n_gpus > 0 and not args.no_warmup:
        runner.warm_up(gpu_jobs)
    threads = []
    if gpu_jobs:
        threads.append(threading.Thread(target=runner.run_pool, args=(gpu_jobs, "gpu", args.gpu_slots)))
    if cpu_jobs:
        threads.append(threading.Thread(target=runner.run_pool, args=(cpu_jobs, "cpu", args.cpu_slots)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    runner.log(f"suite {label} finished in {(time.time() - t0) / 3600:.2f} h; {len(runner.failed)} failed jobs")
    for name in runner.failed:
        runner.log(f"  FAILED {name}")
    merge(results_dir, runner.log)
    return 1 if runner.failed else 0


if __name__ == "__main__":
    sys.exit(main())
