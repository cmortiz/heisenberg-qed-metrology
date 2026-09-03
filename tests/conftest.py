"""pytest configuration: make the repo root importable (``eqsp`` is not an
installed package, ``[tool.uv] package = false``)."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep every JAX-backed test on the CPU (setdefault: an explicit JAX_PLATFORMS in the
# environment wins).  tests/test_regression.py passes the same variable to the run.py
# subprocesses it spawns.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
