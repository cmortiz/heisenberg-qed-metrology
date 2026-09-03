"""Protocol adapters.  Modules import lazily so ``sequential`` never loads jax
and ``--device cpu`` can set JAX_PLATFORMS before the first jax import."""
from __future__ import annotations

import importlib
import os
import sys

from ..config import RunSpec

PROTOCOLS = ("bare_adaptive", "bare_fixed", "ed", "combined", "sequential", "hwindow")   # every protocol but sequential runs on the JAX engine


def set_device(device: str) -> None:
    """Set JAX_PLATFORMS before jax is imported (no-op for 'auto')."""
    if device == "auto":
        return
    if "jax" in sys.modules and os.environ.get("JAX_PLATFORMS") != {"cpu": "cpu", "gpu": "cuda"}[device]:
        raise RuntimeError("jax is already imported; set --device before the first jax import")
    os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda"}[device]


def get_protocol(spec: RunSpec, device: str = "auto"):
    """Adapter instance for ``spec``.  Bayesian protocols return a
    ``engine.BayesianProtocol``; ``sequential`` returns an object with
    ``run_seeds(seeds, epsilons, engine_version, jax_version, progress, on_seed_done)``."""
    if spec.protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {spec.protocol!r}")
    if spec.protocol == "sequential":
        mod = importlib.import_module(".sequential", __name__)
        return mod.Sequential(spec)
    set_device(device)
    mod = importlib.import_module(f".{spec.protocol}", __name__)
    return mod.make(spec)
