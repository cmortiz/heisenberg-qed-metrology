#!/usr/bin/env python3
"""Fill the paper's \\NUM{key} placeholders from results/v2/numbers.json.

    uv run python scripts/paper_numbers.py --tex ../sensing.tex --numbers results/v2/numbers.json --dry-run
    uv run python scripts/paper_numbers.py --tex ../sensing.tex --numbers results/v2/numbers.json --write

Every number printed in the manuscript's numerics section comes from this
substitution, so the text never carries arithmetic of its own.  Keys are the
dotted names of results/numbers_schema.md; the paper uses the short aliases of
``ALIASES`` (``conv`` = ``converged_pct``, ``alpha_asym`` = ``alpha_asymptotic``,
``accept`` = ``accept_pct``, ``comb_`` = ``combined_``, ``seq_`` = ``sequential_``,
``per_block_accept_`` = ``per_block_accept_pct_``, ``g0`` = ``g0.00``, and
``alpha_full`` = ``alpha`` for tables whose headline window is the full range) plus
``EXTRA`` (``<cell>.accept`` = the rejection-mode ``accept_pct`` of that cell).
Formatting is decided by the key suffix (see ``fmt``).  With --write the file is
rewritten in place and a ``\\NUMSRC`` comment line lists the numbers.json path and
the git SHA of the data, so the substitution is reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

NUM_RE = re.compile(r"\\NUM\{([^}]+)\}")

ALIASES = [
    (".conv", ".converged_pct"),
    (".alpha_asym", ".alpha_asymptotic"),
    (".accept", ".accept_pct"),
    (".comb_", ".combined_"),
    (".seq_", ".sequential_"),
    (".per_block_accept_", ".per_block_accept_pct_"),
    (".g0.", ".g0.00."),
    (".alpha_full", ".alpha"),            # tables whose headline window is the full range
]
# joint acceptance of a table cell = the rejection-mode entry
EXTRA = [(".accept", ".rej.accept_pct")]


def candidates(key: str):
    yield key
    k = key
    for a, b in ALIASES:
        k = k.replace(a, b)
    yield k
    # single-alias variants
    for a, b in ALIASES:
        yield key.replace(a, b)
    for a, b in EXTRA:
        if key.endswith(a):
            yield key[: -len(a)] + b


FIELDS = ("n_seeds", "sem", "n_rows")     # "<key>.<field>" reads a field of the entry at <key>


def lookup(numbers: dict, key: str):
    for c in candidates(key):
        if c in numbers:
            return c, numbers[c]
    head, _, field = key.rpartition(".")
    if field in FIELDS:
        c, e = lookup(numbers, head)
        if isinstance(e, dict) and e.get(field) is not None:
            return f"{c}.{field}", {"value": e[field]}
    return None, None


def _val(entry):
    if isinstance(entry, dict):
        return entry.get("value"), entry.get("sem")
    return entry, None


def fmt(key: str, entry) -> str:
    v, sem = _val(entry)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        if isinstance(entry, dict) and entry.get("n_seeds") == 0:
            return "--"                                # cell ran but no seed retained min_points converged targets in the window
        return r"\textcolor{red}{[NaN " + key + "]}"
    if isinstance(v, (list, tuple)):                  # [min, max] range: "a" if equal after rounding, else "a--b"
        base = key.replace("_range", "")
        if base.rsplit(".", 1)[-1].startswith("alpha"):
            lo, hi = (f"{x:.3f}" for x in v)          # exponent ranges keep three decimals (0.995--0.996)
        else:
            lo, hi = (fmt(base, {"value": x}) for x in v)
        return lo if lo == hi else f"{lo}\\text{{--}}{hi}"
    last = key.rsplit(".", 1)[-1]
    if key.startswith("text.fisher_saturation"):
        return f"{v:.3f}"                              # saturation ratios sit within a few percent of 1
    if last.startswith("alpha"):
        if sem is not None and not math.isnan(sem):
            return f"{v:.2f} \\pm {sem:.2f}" if sem >= 0.005 else f"{v:.3f} \\pm {sem:.3f}"
        return f"{v:.2f}"
    if last.startswith("conv"):
        return f"{v:.1f}" if last.endswith("_min") or "_min_" in last else f"{v:.0f}"   # a minimum keeps one decimal so "at least 99.6%" never rounds to 100
    if last.startswith("accept"):
        return f"{v:.1f}"
    if last.startswith("N_argmin") or last.startswith("M_exec") or last.startswith("M_max") or last.startswith("n_"):
        return f"{int(round(v))}"
    if "rel_rms" in last:
        return f"{100.0 * v:.0f}"                      # printed as a percentage
    if last.startswith("logfit_"):
        return f"{v:.1f}"
    if last.startswith("window_shift") or last.startswith("M2sigma2") or last.startswith("visibility"):
        return f"{v:.2f}" if abs(v) >= 0.1 else f"{v:.3f}"   # two significant figures for these O(0.01) to O(1) quantities
    if last.startswith("nonconv_error"):
        return f"{v:.0f}"
    if last.startswith("prefactor_ratio") or last.startswith("bare_over") or last.startswith("edfl_over") or last.startswith("rej_over"):
        return f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    if last.endswith("_C") or last == "C":
        return f"{int(round(v))}"
    if last.startswith("per_block_accept"):
        return f"{v:.1f}"
    if isinstance(v, (int,)) or float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.3g}"


def data_sha(numbers_path: Path) -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", numbers_path.name],
                              capture_output=True, text=True, cwd=numbers_path.parent).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--numbers", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    numbers = json.loads(args.numbers.read_text())
    tex = args.tex.read_text()
    keys = NUM_RE.findall(tex)
    resolved, missing = {}, []
    for k in dict.fromkeys(keys):
        c, e = lookup(numbers, k)
        if c is None:
            missing.append(k)
        else:
            resolved[k] = (c, fmt(k, e))
    print(f"{len(set(keys))} unique placeholders, {len(resolved)} resolved, {len(missing)} missing")
    for k in missing:
        print("  MISSING", k)
    if args.dry_run or not args.write:
        for k, (c, s) in sorted(resolved.items()):
            print(f"  {k:60s} <- {c:60s} = {s}")
        return 1 if missing else 0
    if missing:
        print("refusing to write with missing keys", file=sys.stderr)
        return 1
    out = NUM_RE.sub(lambda m: resolved[m.group(1)][1], tex)
    stamp = f"% \\NUMSRC numbers.json {args.numbers} @ {data_sha(args.numbers)}\n"
    if "% \\NUMSRC" in out:
        out = re.sub(r"% \\NUMSRC[^\n]*\n", lambda _m: stamp, out, count=1)   # lambda: the stamp holds backslashes
    else:
        out = out.replace("\\newcommand{\\NUM}", stamp + "\\newcommand{\\NUM}", 1)
    args.tex.write_text(out)
    print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
