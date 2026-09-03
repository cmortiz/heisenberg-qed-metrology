#!/usr/bin/env python3
"""Paper figures from numbers.json (every printed alpha) and the suite CSVs (data points).

Usage
    uv run python scripts/plot.py                       # results/v2 -> ../figures/*.pdf (+ a 300 dpi .png of each)
    uv run python scripts/plot.py --figs fig1 fig4

No fit is performed here.  Slopes, intercepts and CI widths
come from numbers.json entries written by analyze.py; the CSVs supply only the per-epsilon
data markers (geometric mean of T over converged seeds).  Each figure uses the window of the
table it accompanies: fig1, fig4 full range; fig2, fig3, fig5 asymptotic (eps < 1e-2).
Fit lines are drawn over their window only; the 95% CI fan is centred on the window's
log-midpoint (a band pivoted at eps = 1 would be inflated at small eps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze import FIG1_BARE_GAMMAS, FIG3_CELLS, V2_DIR, clean, gk, kk, load_results, qk, sk  # noqa: E402
from eqsp import fit  # noqa: E402

FIGURES_DIR = REPO.parent / "figures"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": "#3c3836", "axes.labelcolor": "#3c3836", "xtick.color": "#504945", "ytick.color": "#504945",
    "font.family": "serif", "font.size": 8, "mathtext.fontset": "cm",
    "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": "#504945", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "in", "ytick.direction": "in", "axes.grid": False,
    "legend.frameon": False, "legend.fontsize": 7, "legend.labelcolor": "#3c3836",
    "lines.linewidth": 1.2, "lines.markersize": 4.0, "figure.dpi": 300,
    "hatch.linewidth": 0.8, "hatch.color": "#504945",
})
GAMMA_COLORS = {0.01: "#458588", 0.05: "#b57614", 0.10: "#cc241d"}   # gruvbox blue, dark yellow, red
ED_COLOR = "#504945"
REF_COLOR = "#7c6f64"
EDGE_COLOR = "#504945"
THREE_COLORS = ["#458588", "#b57614", "#cc241d"]


# --------------------------------------------------------------------------- helpers
def load_numbers(results_dir: Path) -> dict:
    path = results_dir / "numbers.json"
    if not path.exists():
        sys.exit(f"{path} missing: run scripts/analyze.py first")
    return json.loads(path.read_text())


def entry(nums: dict, key: str) -> dict | None:
    e = nums.get(key)
    if e is None or e.get("value") is None:
        print(f"  skip: numbers.json has no usable '{key}'")
        return None
    return e


def rows(df: pd.DataFrame, **cond) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, val in cond.items():
        mask &= np.isclose(df[col], val) if isinstance(val, (int, float)) else (df[col] == val).to_numpy()
    return df[mask]


def bare_adaptive(df: pd.DataFrame) -> pd.DataFrame:
    """Bare GHZ with adaptive N, no readout or prep noise."""
    return clean(df[df["protocol"] == "bare_adaptive"])


def gmean_T_per_eps(cell: pd.DataFrame) -> pd.DataFrame:
    conv = cell[cell["converged"]]
    g = conv.groupby("epsilon")["total_resources"].apply(lambda t: float(np.exp(np.log(t.astype(float)).mean())))
    return g.reset_index().rename(columns={"total_resources": "T"})


def window_range(cell: pd.DataFrame, window: str) -> tuple[float, float]:
    eps = cell.loc[cell["converged"], "epsilon"]
    eps = eps[eps < fit.WINDOWS[window][0]]
    return float(eps.min()), float(eps.max())


def draw_fit(ax, e: dict, eps_lo: float, eps_hi: float, color, ls, lw=1.2, band_alpha=0.22):
    """Line exp(intercept - alpha log eps) over [eps_lo, eps_hi] with a 95% CI fan on alpha."""
    eps = np.logspace(np.log10(eps_lo), np.log10(eps_hi), 200)
    T = np.exp(e["intercept"] - e["value"] * np.log(eps))
    ax.plot(eps, T, color=color, ls=ls, lw=lw, zorder=4)
    if e.get("ci95"):
        centre = 0.5 * (np.log(eps_lo) + np.log(eps_hi))
        fan = e["ci95"] * np.abs(np.log(eps) - centre)
        ax.fill_between(eps, T * np.exp(-fan), T * np.exp(fan), alpha=band_alpha, color=color, lw=0, zorder=2)
    return eps, T


def draw_data(ax, cell: pd.DataFrame, color, marker="o", ms=2.6, alpha=0.9):
    d = gmean_T_per_eps(cell)
    ax.plot(d["epsilon"], d["T"], ls="none", marker=marker, ms=ms, color=color, alpha=alpha,
            markeredgewidth=0.4, markeredgecolor="white", zorder=3)


def alpha_text(e: dict, nd=2) -> str:
    return rf"$\alpha = {e['value']:.{nd}f}$"


def add_alpha_annotation(ax, eps, T, text, color, x_frac=0.5, offset_pts=6):
    """Annotation rotated to the line slope (call after axis limits are set)."""
    idx = max(1, min(int(len(eps) * x_frac), len(eps) - 2))
    pts = ax.transData.transform(np.column_stack((eps[[idx - 1, idx + 1]], T[[idx - 1, idx + 1]])))
    angle = np.degrees(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0]))
    offset = mtransforms.offset_copy(ax.transData, fig=ax.figure, y=offset_pts, units="points")
    ax.text(eps[idx], T[idx], text, color=color, fontsize=7, ha="center", va="bottom", rotation=angle,
            rotation_mode="anchor", transform=offset, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def setup_loglog_axes(ax, xlabel=r"Target precision $\varepsilon$", ylabel=r"Total resources $T$"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(8e-5, 1.5e-1)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out / name}.pdf")


# --------------------------------------------------------------------------- figure 1
def figure1(df, nums, out):
    """Within-model: bare GHZ (adaptive N) at gamma in {0.01, 0.05, 0.10} and ED-FL L=1 at gamma=0.10. Window: full."""
    print("fig1_error_detection")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bare = bare_adaptive(df)
    styles = {0.01: "-", 0.05: "--", 0.10: "-."}
    legend, annotate = [], []
    for g in FIG1_BARE_GAMMAS:
        cell, e = rows(bare, gamma=g), entry(nums, f"fig1.alpha.bare_{gk(g)}")
        if cell.empty or e is None:
            continue
        draw_data(ax, cell, GAMMA_COLORS[g])
        eps, T = draw_fit(ax, e, *window_range(cell, e["window"]), GAMMA_COLORS[g], styles[g])
        annotate.append((eps, T, alpha_text(e), GAMMA_COLORS[g], 0.35 if g < 0.10 else 0.25))
        legend.append(Line2D([0], [0], color=GAMMA_COLORS[g], ls=styles[g], marker="o", ms=2.6,
                             label=rf"Bare GHZ, $\gamma{{=}}{100 * g:g}\%$"))
    ed = rows(clean(df[df["mode"] == "ed_full_likelihood"]), L=1, gamma=0.10)
    e = entry(nums, "fig1.alpha.edfl_L1_g0.10")
    if not ed.empty and e is not None:
        draw_data(ax, ed, ED_COLOR, marker="s")
        eps, T = draw_fit(ax, e, *window_range(ed, e["window"]), ED_COLOR, (0, (4, 2)), lw=1.4)
        annotate.append((eps, T, alpha_text(e), ED_COLOR, 0.80))
        legend.append(Line2D([0], [0], color=ED_COLOR, ls=(0, (4, 2)), lw=1.4, marker="s", ms=2.6,
                             label=r"ED-FL, $L{=}1$ ($N{=}3$), $\gamma{=}10\%$"))
    setup_loglog_axes(ax)
    for eps, T, text, color, xf in annotate:
        add_alpha_annotation(ax, eps, T, text, color, x_frac=xf)
    ax.legend(handles=legend, loc="upper right", fontsize=6.5, handlelength=2.0, handletextpad=0.5, labelspacing=0.3)
    save(fig, out, "fig1_error_detection")


# --------------------------------------------------------------------------- figure 2
def figure2(df, nums, out):
    """Grouped bars: asymptotic alpha (a) and Converged (%) (b) for bare GHZ and ED L=1,2,3, PS solid / FL hatched."""
    print("fig2_inference_comparison")
    gammas = [0.01, 0.05, 0.10]
    groups = []   # (label, [(gamma, ps_alpha, fl_alpha, ps_conv, fl_conv)])
    bare_items = []
    for g in gammas:
        a, c = entry(nums, f"fig2.alpha.bare_{gk(g)}"), entry(nums, f"fig2.conv.bare_{gk(g)}")
        if a and c:
            bare_items.append((g, a, None, c, None))
    if bare_items:
        groups.append(("Bare GHZ", bare_items))
    for L in (1, 2, 3):
        items = []
        for g in gammas:
            ps, fl = entry(nums, f"fig2.alpha.ed_L{L}_{gk(g)}_rej"), entry(nums, f"fig2.alpha.ed_L{L}_{gk(g)}_edfl")
            psc, flc = entry(nums, f"fig2.conv.ed_L{L}_{gk(g)}_rej"), entry(nums, f"fig2.conv.ed_L{L}_{gk(g)}_edfl")
            if ps and fl and psc and flc:
                items.append((g, ps, fl, psc, flc))
        if items:
            groups.append((f"ED $L\\!={L}$", items))
    if not groups:
        print("  skip: no fig2 keys")
        return

    fig, (ax_a, ax_c) = plt.subplots(2, 1, figsize=(3.6, 4.2), height_ratios=[2, 1], sharex=True)
    fig.subplots_adjust(hspace=0.25)
    bw, gap_pair, gap_gamma, gap_group = 0.35, 0.05, 0.55, 0.85
    x, ticks, tick_labels, ranges = 0.0, [], [], []
    for label, items in groups:
        start = x
        for g, ps, fl, psc, flc in items:
            color = GAMMA_COLORS[g]
            bar_kw = dict(color=color, edgecolor=EDGE_COLOR, linewidth=0.4, zorder=3)
            err_kw = dict(capsize=2, error_kw={"linewidth": 0.8, "color": EDGE_COLOR})
            ax_a.bar(x, ps["value"], bw, yerr=ps["sem"], **bar_kw, **err_kw)
            ax_c.bar(x, psc["value"], bw, **bar_kw)
            if fl is None:
                ticks.append(x + bw / 2)
                x += bw + gap_gamma + 0.35
            else:
                xf = x + bw + gap_pair
                ax_a.bar(xf, fl["value"], bw, yerr=fl["sem"], hatch="//////", **bar_kw, **err_kw)
                ax_c.bar(xf, flc["value"], bw, hatch="//////", **bar_kw)
                ticks.append((x + xf + bw) / 2)
                x = xf + bw + gap_gamma
            tick_labels.append(f"{100 * g:g}%")
        ranges.append((start, x - gap_gamma, label))
        x += gap_group
    for a, y, txt in ((ax_a, 1.0, "HL"), (ax_a, 2.0, "SQL")):
        a.axhline(y, color=REF_COLOR, ls="--" if y == 1 else ":", lw=0.7, zorder=1)
        a.text(1.02, y, txt, fontsize=7, color=REF_COLOR, va="center", ha="left", transform=a.get_yaxis_transform(), clip_on=False)
    ax_c.set_xticks(ticks)
    ax_c.set_xticklabels(tick_labels, fontsize=6.5)
    for start, end, label in ranges:
        ax_c.text((start + end) / 2, -0.35, label, ha="center", va="top", fontsize=7.5, fontweight="bold",
                  transform=ax_c.get_xaxis_transform(), clip_on=False)
    for i in range(1, len(ranges)):
        for a in (ax_a, ax_c):
            a.axvline(ranges[i][0] - gap_group / 2, color="#d5c4a1", lw=0.5, ls=":", zorder=0)
    ax_a.text(-0.14, 1.02, "(a)", transform=ax_a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax_c.text(-0.14, 1.02, "(b)", transform=ax_c.transAxes, fontweight="bold", fontsize=9, va="bottom")
    ax_c.set_xlabel(r"Noise rate $\gamma$", fontsize=9, labelpad=22)
    ax_a.set_ylabel(r"Asymptotic exponent $\alpha$")
    ax_a.set_ylim(0.8, 2.2)
    ax_a.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_c.set_ylabel("Converged (%)")
    ax_c.set_ylim(50, 105)
    handles = [Patch(facecolor=GAMMA_COLORS[0.01], edgecolor=EDGE_COLOR, linewidth=0.4, label=r"$\gamma{=}1\%$"),
               Patch(facecolor="white", edgecolor=EDGE_COLOR, linewidth=0.4, label="Rejection sampling"),
               Patch(facecolor=GAMMA_COLORS[0.05], edgecolor=EDGE_COLOR, linewidth=0.4, label=r"$\gamma{=}5\%$"),
               Patch(facecolor="white", edgecolor=EDGE_COLOR, linewidth=0.4, hatch="//////", label="ED-FL"),
               Patch(facecolor=GAMMA_COLORS[0.10], edgecolor=EDGE_COLOR, linewidth=0.4, label=r"$\gamma{=}10\%$")]
    ax_a.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=6.5, columnspacing=0.6,
                handletextpad=0.3, handlelength=1.2, labelspacing=0.3, borderaxespad=0, frameon=True, fancybox=False,
                edgecolor="none", facecolor="white")
    save(fig, out, "fig2_inference_comparison")


# --------------------------------------------------------------------------- figure 3
def figure3(df, nums, out):
    """Combined ED-FL at (L=1, 1%), (L=1, 10%), (L=2, 10%), sigma = 0.01. Window: asymptotic."""
    print("fig3_combined_protocol")
    comb = clean(df[df["mode"] == "combined_full_likelihood"])
    styles = [(GAMMA_COLORS[0.01], "-", 1.2), (GAMMA_COLORS[0.10], (0, (3, 1, 1, 1)), 1.2), (ED_COLOR, (0, (4, 2)), 1.4)]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    legend = []
    for (L, g, s), (color, ls, lw) in zip(FIG3_CELLS, styles):
        cell, e = rows(comb, L=L, gamma=g, sigma_epsilon=s), entry(nums, f"fig3.alpha.combined_L{L}_{gk(g)}_{sk(s)}")
        if cell.empty or e is None:
            continue
        draw_data(ax, cell, color, marker="o" if L == 1 else "s")
        draw_fit(ax, e, *window_range(cell, e["window"]), color, ls, lw=lw, band_alpha=0.15)
        legend.append(Line2D([0], [0], color=color, ls=ls, lw=lw, marker="o" if L == 1 else "s", ms=2.6,
                             label=rf"$L{{=}}{L}$, $\gamma{{=}}{100 * g:g}\%$, {alpha_text(e)}"))
    if not legend:
        plt.close(fig)
        print("  skip: no combined cells")
        return
    setup_loglog_axes(ax)
    ax.axvline(fit.EPS_ASYMPTOTIC, color=REF_COLOR, ls=":", lw=0.6, zorder=1)
    ax.text(fit.EPS_ASYMPTOTIC * 1.1, 0.03, "fit window", fontsize=6, color=REF_COLOR, ha="left", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.legend(handles=legend, loc="upper right", fontsize=6.5, handlelength=2.0, handletextpad=0.5, labelspacing=0.4)
    save(fig, out, "fig3_combined_protocol")


# --------------------------------------------------------------------------- figure 4
def figure4(df, nums, out):
    """Heisenberg window: T(N) mean (solid) and median (dashed) at sigma in {0.05, 0.10}; N* and N*/2 marked."""
    print("fig4_heisenberg_window")
    hw = df[df["protocol"] == "hwindow"]
    if hw.empty:
        print("  skip: no hwindow rows")
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    sigmas = sorted(hw["sigma_epsilon"].unique())
    colors = dict(zip(sigmas, [GAMMA_COLORS[0.05], GAMMA_COLORS[0.10], ED_COLOR][: len(sigmas)]))
    legend = []
    for s in sigmas:
        p = f"fig4.window.{sk(s)}"
        Ns = sorted(int(n) for n in rows(hw, sigma_epsilon=s)["N"].unique())
        mean = [entry(nums, f"{p}.T_mean_N{N}") for N in Ns]
        median = [entry(nums, f"{p}.T_median_N{N}") for N in Ns]
        keep = [i for i, (m, d) in enumerate(zip(mean, median)) if m and d]
        if not keep:
            continue
        N_arr = np.array([Ns[i] for i in keep], float)
        m_arr = np.array([mean[i]["value"] for i in keep])
        m_sem = np.array([mean[i]["sem"] or 0.0 for i in keep])
        d_arr = np.array([median[i]["value"] for i in keep])
        ax.errorbar(N_arr, m_arr, yerr=m_sem, color=colors[s], marker="o", ms=3, lw=1.1, capsize=1.5, elinewidth=0.6, zorder=3)
        ax.plot(N_arr, d_arr, color=colors[s], ls="--", marker="s", ms=2.4, lw=0.9, alpha=0.8, zorder=3)
        for key, ls, lab in ((f"{p}.N_star", ":", r"$N_*$"), (f"{p}.N_star_half", "-.", r"$N_*/2$")):
            e = entry(nums, key)
            if e:
                ax.axvline(e["value"], color=colors[s], ls=ls, lw=0.7, alpha=0.9, zorder=1)
                ax.text(e["value"], 0.985, lab, transform=ax.get_xaxis_transform(), fontsize=6, color=colors[s],
                        ha="center", va="top", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        e = entry(nums, f"{p}.N_argmin")
        if e:
            i = keep[[Ns[j] for j in keep].index(int(e["value"]))] if int(e["value"]) in [Ns[j] for j in keep] else None
            if i is not None:
                ax.plot([Ns[i]], [mean[i]["value"]], marker="*", ms=8, color=colors[s], markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        legend.append(Line2D([0], [0], color=colors[s], marker="o", ms=3, label=rf"$\sigma_\epsilon = {s:g}$"))
    legend += [Line2D([0], [0], color=REF_COLOR, ls="-", marker="o", ms=3, label="mean over seeds"),
               Line2D([0], [0], color=REF_COLOR, ls="--", marker="s", ms=2.4, label="median over seeds"),
               Line2D([0], [0], color=REF_COLOR, ls="none", marker="*", ms=7, label=r"$\arg\min_N \bar T$")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Probe size $N$")
    ax.set_ylabel(r"Total resources $T$ at $\varepsilon_{\mathrm{tar}} = 2 \times 10^{-2}/N_*$")
    ax.legend(handles=legend, loc="upper left", fontsize=6.3, handlelength=1.8, labelspacing=0.3)
    save(fig, out, "fig4_heisenberg_window")


# --------------------------------------------------------------------------- figure 5
def figure5(df, nums, out):
    """(a) readout errors: T vs eps for bare (adaptive N) and ED-FL L=1 at each q, mismatched solid / q-aware dashed,
    eps = 2q marked (bare probe size reaches n = 1/(2q)); (b) prep infidelity: asymptotic alpha vs k per protocol."""
    print("fig5_realistic")
    real = df[(df["q"] > 0) | (df["k"] < 1) | (df["F"] < 1)]
    if real.empty:
        print("  skip: no readout/prep rows")
        return
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.9, 2.8))
    fig.subplots_adjust(wspace=0.32)

    # (a) readout
    qs = sorted(real.loc[real["q"] > 0, "q"].unique())
    qcolors = dict(zip(qs, THREE_COLORS[: len(qs)]))
    legend = []
    for q in qs:
        for proto, mode, marker, L in (("bare_adaptive", "bare_hamiltonian_adaptive", "o", None), ("ed", "ed_full_likelihood", "s", 1)):
            for est, ls in (("mismatched", "-"), ("aware", "--")):
                sub = df[(df["protocol"] == proto) & (df["mode"] == mode) & (df["estimator"] == est)]
                cell = rows(sub, q=q) if L is None else rows(sub, q=q, L=L)      # adaptive bare has no L
                if cell.empty:
                    continue
                d = gmean_T_per_eps(cell)
                ax_a.plot(d["epsilon"], d["T"], ls=ls, marker=marker, ms=2.2, lw=0.9, color=qcolors[q],
                          alpha=1.0 if proto == "ed" else 0.75, markeredgewidth=0.3, markeredgecolor="white", zorder=3)
        legend.append(Line2D([0], [0], color=qcolors[q], lw=1.2, label=rf"$q = {q:g}$"))
        eb = entry(nums, f"text.readout.n_breakdown_{qk(q)}")
        if eb:
            ax_a.axvline(2 * q, color=qcolors[q], ls=":", lw=0.7, zorder=1)
            ax_a.text(2 * q, 0.02, rf"$n{{=}}{int(eb['value'])}$", transform=ax_a.get_xaxis_transform(), fontsize=5.8,
                      color=qcolors[q], ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    legend += [Line2D([0], [0], color=REF_COLOR, marker="o", ms=2.4, lw=0.9, label="bare GHZ (adaptive $N$)"),
               Line2D([0], [0], color=REF_COLOR, marker="s", ms=2.4, lw=0.9, label=r"ED-FL $L{=}1$"),
               Line2D([0], [0], color=REF_COLOR, ls="-", lw=0.9, label="mismatched estimator"),
               Line2D([0], [0], color=REF_COLOR, ls="--", lw=0.9, label=r"$q$-aware estimator")]
    setup_loglog_axes(ax_a)
    ax_a.text(-0.16, 1.02, "(a)", transform=ax_a.transAxes, fontweight="bold", fontsize=9, va="bottom")
    if qs:
        ax_a.legend(handles=legend, loc="upper right", fontsize=5.6, handlelength=1.8, labelspacing=0.25, handletextpad=0.4)

    # (b) prep infidelity
    ks = sorted(real.loc[real["k"] < 1, "k"].unique())
    protos = [("bare", "bare_adaptive", None, "o"), ("ed", "ed", 1, "s"), ("ed", "ed", 2, "D"), ("ed", "ed", 3, "^"),
              ("combined", "combined", 1, "v"), ("combined", "combined", 2, "P"), ("sequential", "sequential", 1, "x")]
    pcolors = dict(zip([p[0] + str(p[2]) for p in protos], plt.cm.viridis(np.linspace(0.05, 0.9, len(protos)))))
    legend_b = []
    for name, proto, L, marker in protos:
        if name == "sequential":
            # product-state probe: F only, no k dependence and no estimator (key tab_realistic.prep.sequential_L{L}_F{F})
            e = next((nums[k] for k in nums if k.startswith(f"tab_realistic.prep.sequential_L{L}_F") and k.endswith(".alpha")), None)
            if e and e.get("value") is not None:
                c = pcolors[name + str(L)]
                ax_b.axhline(e["value"], color=c, ls="-", lw=0.8, zorder=2)
                legend_b.append(Line2D([0], [0], color=c, lw=0.8, label=f"sequential L={L} (F only)"))
            continue
        for est, fill in (("mismatched", "full"), ("aware", "none")):
            xs, ys, es = [], [], []
            for k in ks:
                key = f"tab_realistic.prep.{name}_L{L}_{kk(k)}_{est}.alpha" if L is not None else f"tab_realistic.prep.{name}_{kk(k)}_{est}.alpha"
                e = nums.get(key)
                if e and e.get("value") is not None:
                    xs.append(k)
                    ys.append(e["value"])
                    es.append(e["sem"] or 0.0)
            if not xs:
                continue
            c = pcolors[name + str(L)]
            ax_b.errorbar(xs, ys, yerr=es, color=c, marker=marker, ms=4, lw=0.8, ls="-" if fill == "full" else "--",
                          fillstyle=fill, capsize=1.5, elinewidth=0.6, zorder=3)
            legend_b.append(Line2D([0], [0], color=c, marker=marker, ms=4, lw=0.8, fillstyle=fill, ls="-" if fill == "full" else "--",
                                   label=f"{name}{'' if L is None else f' L={L}'}, {est}"))
    ax_b.axhline(1.0, color=REF_COLOR, ls="--", lw=0.7, zorder=1)
    ax_b.axhline(2.0, color=REF_COLOR, ls=":", lw=0.7, zorder=1)
    ax_b.set_xlabel(r"Entangling-gate fidelity $k$")
    ax_b.set_ylabel(r"Asymptotic exponent $\alpha$")
    ax_b.text(-0.16, 1.02, "(b)", transform=ax_b.transAxes, fontweight="bold", fontsize=9, va="bottom")
    if ks:
        ax_b.set_xticks(ks)
        ax_b.set_xticklabels([f"{k:g}" for k in ks])
    if legend_b:
        ax_b.legend(handles=legend_b, loc="best", fontsize=4.8, ncol=2, handlelength=1.6, labelspacing=0.2, columnspacing=0.8)
    save(fig, out, "fig5_realistic")


# --------------------------------------------------------------------------- main
FIGURES = {"fig1": figure1, "fig2": figure2, "fig3": figure3, "fig4": figure4, "fig5": figure5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=FIGURES_DIR, help="output directory (default ../figures)")
    ap.add_argument("--figs", nargs="+", choices=sorted(FIGURES), default=sorted(FIGURES))
    args = ap.parse_args()
    results_dir = args.results_dir or V2_DIR
    df = load_results(results_dir)
    nums = load_numbers(results_dir)
    for name in args.figs:
        FIGURES[name](df, nums, args.out)


if __name__ == "__main__":
    main()
