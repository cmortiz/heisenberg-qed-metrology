#!/usr/bin/env python3
"""
Publication figures for the Numerical Simulations section (Figures 3-5 in the paper).

Usage: uv run python plot_redesigned.py [--outdir PATH]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from scipy import stats as sp_stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Single-column academic theme (3.4in target width)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#3c3836",
    "axes.labelcolor": "#3c3836",
    "xtick.color": "#504945",
    "ytick.color": "#504945",
    "font.family": "serif",
    "font.size": 8,
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": "#504945",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "legend.labelcolor": "#3c3836",
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "figure.dpi": 300,
    "hatch.linewidth": 0.8,
    "hatch.color": "#504945",
})

# Gruvbox palette: color = noise level (consistent across all figures)
COLORS = {
    "gamma_1":   "#458588",   # gruvbox blue
    "gamma_5":   "#b57614",   # gruvbox dark yellow
    "gamma_10":  "#cc241d",   # gruvbox red
    "ref_line":  "#7c6f64",   # gruvbox fg4 (neutral reference lines)
}

GAMMA_COLORS = {0.01: "#458588", 0.05: "#b57614", 0.10: "#cc241d"}

# Bounding box for reference line labels
REF_BBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def fit_alpha_per_seed(df_config, min_points=10):
    """OLS alpha fit per seed on log-log converged data."""
    fits = []
    for seed in sorted(df_config["seed"].unique()):
        sd = df_config[(df_config["seed"] == seed) & (df_config["converged"] == True)]
        if len(sd) < min_points:
            continue
        log_eps = np.log(sd["epsilon"].values)
        log_T = np.log(sd["total_resources"].values)
        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        coeffs = np.linalg.lstsq(A, log_T, rcond=None)[0]
        fits.append({"alpha": -coeffs[0], "intercept": coeffs[1], "seed": seed})
    return fits


def compute_alpha_summary(df_config, min_points=10):
    """Alpha mean, SEM, convergence rate, and per-seed fits."""
    fits = fit_alpha_per_seed(df_config, min_points=min_points)
    alphas = [f["alpha"] for f in fits]
    n_seeds = df_config["seed"].nunique()
    conv_rate = df_config["converged"].mean() * 100

    if len(alphas) >= 2:
        alpha_mean = np.mean(alphas)
        alpha_sem = np.std(alphas, ddof=1) / np.sqrt(len(alphas))
    elif len(alphas) == 1:
        alpha_mean = alphas[0]
        alpha_sem = 0.0
    else:
        alpha_mean = np.nan
        alpha_sem = np.nan

    return {
        "alpha_mean": alpha_mean,
        "alpha_sem": alpha_sem,
        "conv_rate": conv_rate,
        "n_seeds": n_seeds,
        "n_alpha_seeds": len(alphas),
        "fits": fits,
    }


def plot_fit_line(ax, fits, df_conv, color, linestyle, linewidth=1.2,
                  ci_alpha=0.22):
    """Mean fit line with 95% CI band."""
    if not fits:
        return None

    alphas = [f["alpha"] for f in fits]
    intercepts = [f["intercept"] for f in fits]
    mean_alpha = np.mean(alphas)
    sem = np.std(alphas, ddof=1) / np.sqrt(len(alphas)) if len(alphas) > 1 else 0
    mean_intercept = np.mean(intercepts)

    if len(df_conv) == 0:
        return None

    eps_range = np.logspace(np.log10(df_conv["epsilon"].min()),
                            np.log10(df_conv["epsilon"].max()), 200)
    T_fit = np.exp(mean_intercept - mean_alpha * np.log(eps_range))

    ax.plot(eps_range, T_fit, color=color, linewidth=linewidth,
            linestyle=linestyle, zorder=4)

    # 95% CI band
    if sem > 0 and len(fits) > 1:
        n_seeds = len(fits)
        t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)
        ci95 = t_crit * sem
        T_lo = np.exp(mean_intercept - (mean_alpha + ci95) * np.log(eps_range))
        T_hi = np.exp(mean_intercept - (mean_alpha - ci95) * np.log(eps_range))
        ax.fill_between(eps_range, T_lo, T_hi, alpha=ci_alpha, color=color, zorder=2)

    return {"alpha": mean_alpha, "sem": sem, "eps_range": eps_range, "T_fit": T_fit}


REF_COLOR = "#a89984"  # gruvbox fg3 (lighter, reads as background guide)


def draw_slope_triangle(ax, x0, y0, slope, width_decades=0.4, label="",
                        fontsize=7, facecolor="#e8e3dc", edgecolor="#7c6f64"):
    """Right-angle triangle showing reference slope on log-log axes."""
    x1 = x0 * 10**width_decades
    y1 = y0 * (x1 / x0)**slope  # = y0 * 10^(slope * width_decades)

    # Right angle vertex: same x as endpoint, same y as start
    x_right, y_right = x1, y0

    ax.fill([x0, x_right, x1, x0], [y0, y_right, y1, y0],
            facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5,
            zorder=3, clip_on=True)

    # Label at log-space centroid
    log_xc = (np.log10(x0) + np.log10(x_right) + np.log10(x1)) / 3
    log_yc = (np.log10(y0) + np.log10(y_right) + np.log10(y1)) / 3
    ax.text(10**log_xc, 10**log_yc, label, fontsize=fontsize,
            ha="center", va="center", color=edgecolor, fontweight="bold",
            zorder=4)


def add_alpha_annotation(ax, result, text, color, x_frac=0.5, offset_pts=6):
    """Annotate fit line with alpha, rotated to match slope. Call after axis limits are set."""
    eps = result["eps_range"]
    T = result["T_fit"]
    idx = int(len(eps) * x_frac)
    idx = max(1, min(idx, len(eps) - 2))

    # Compute rotation in display (pixel) space
    pts = ax.transData.transform(
        np.column_stack((eps[[idx - 1, idx + 1]], T[[idx - 1, idx + 1]])))
    angle = np.degrees(np.arctan2(pts[1, 1] - pts[0, 1],
                                  pts[1, 0] - pts[0, 0]))

    offset = mtransforms.offset_copy(
        ax.transData, fig=ax.figure, y=offset_pts, units="points")
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=0.85)
    ax.text(eps[idx], T[idx], text, color=color, fontsize=7,
            ha="center", va="bottom", rotation=angle,
            rotation_mode="anchor", transform=offset, zorder=5, bbox=bbox)


def setup_loglog_axes(ax, xlabel=r"Target precision $\varepsilon$",
                      ylabel=r"Total resources $T$"):
    """Standard log-log axes for T vs epsilon plots."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(8e-5, 1.5e-1)
    ax.set_ylim(1e2, 2e10)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# =========================================================================
# FIGURE 1: Error Detection Payoff
# =========================================================================

def figure1_error_detection(ghz_df, ed_df, outdir):
    """Figure 1: bare GHZ at gamma=1%,10% + ED full-likelihood L=1 at gamma=10%.

    Caller passes ed_full_likelihood_df as ed_df; Algorithm~ref{alg:bitflip} is
    the full-likelihood estimator, not post-selection (see sensing.tex §4).
    """
    print("\n--- Figure 1: Error-detection payoff ---")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    gamma_configs = [
        (0.01, COLORS["gamma_1"],  "-",  r"$\gamma{=}1\%$"),
        (0.10, COLORS["gamma_10"], "-.", r"$\gamma{=}10\%$"),
    ]

    # Bare GHZ at bookend noise levels
    ghz_results = {}
    for gamma, color, ls, label in gamma_configs:
        if ghz_df is not None:
            bare = ghz_df[(np.isclose(ghz_df["gamma"], gamma)) &
                          (np.isclose(ghz_df["h"], 0.0))]
            if len(bare) > 0:
                fits = fit_alpha_per_seed(bare)
                conv = bare[bare["converged"] == True]
                result = plot_fit_line(ax, fits, conv, color, ls)
                if result:
                    print(f"  Bare GHZ {label}: alpha = {result['alpha']:.3f} "
                          f"+/- {result['sem']:.3f}")
                    ghz_results[gamma] = (result, ls)

    # Single ED full-likelihood L=1 representative (gamma=10%, dashed, distinct color)
    ed_color = "#504945"  # dark gray, distinct from data colors
    ed_result = None
    if ed_df is not None:
        ed_l1 = ed_df[(ed_df["L"] == 1) & (np.isclose(ed_df["gamma"], 0.10))]
        if len(ed_l1) > 0:
            fits = fit_alpha_per_seed(ed_l1)
            conv = ed_l1[ed_l1["converged"] == True]
            result = plot_fit_line(ax, fits, conv, ed_color,
                                   (0, (4, 2)), linewidth=1.4)
            if result:
                print(f"  ED-FL L=1 gamma=10%: alpha = {result['alpha']:.3f} "
                      f"+/- {result['sem']:.3f}")
                ed_result = result

    setup_loglog_axes(ax)
    ax.set_ylim(1e2, 1e9)

    # Alpha annotations (after axes finalized for correct rotation)
    if 0.01 in ghz_results:
        res_1, _ = ghz_results[0.01]
        add_alpha_annotation(ax, res_1,
                             rf"$\alpha = {res_1['alpha']:.2f}$",
                             COLORS["gamma_1"], x_frac=0.40)
    if 0.10 in ghz_results:
        res_10, _ = ghz_results[0.10]
        add_alpha_annotation(ax, res_10,
                             rf"$\alpha = {res_10['alpha']:.2f}$",
                             COLORS["gamma_10"], x_frac=0.3)
    if ed_result:
        add_alpha_annotation(ax, ed_result,
                             rf"$\alpha = {ed_result['alpha']:.2f}$",
                             ed_color, x_frac=0.80)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS["gamma_1"], ls="-", lw=1.2,
               label=r"Bare GHZ, $\gamma{=}1\%$"),
        Line2D([0], [0], color=COLORS["gamma_10"], ls="-.", lw=1.2,
               label=r"Bare GHZ, $\gamma{=}10\%$"),
        Line2D([0], [0], color=ed_color, ls=(0, (4, 2)), lw=1.4,
               label=r"ED-FL, $L{=}1$, $\gamma{=}10\%$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              handlelength=2.0, handletextpad=0.5, labelspacing=0.3)

    fig.savefig(outdir / "fig1_error_detection.pdf", dpi=300,
                bbox_inches="tight")
    fig.savefig(outdir / "fig1_error_detection.png", dpi=300,
                bbox_inches="tight")
    print(f"  Saved: {outdir}/fig1_error_detection.pdf")
    plt.close()


# =========================================================================
# FIGURE 2: Inference Mode Comparison (grouped bar chart)
# =========================================================================

def figure2_inference_comparison(ghz_df, ed_ps_df, ed_fl_df, outdir):
    """Figure 4: grouped bar chart of alpha and convergence by noise and inference mode."""
    print("\n--- Figure 2: Inference mode comparison ---")

    fig, (ax_alpha, ax_conv) = plt.subplots(
        2, 1, figsize=(3.6, 4.2), height_ratios=[2, 1], sharex=True)
    fig.subplots_adjust(hspace=0.25)

    gammas = [0.01, 0.05, 0.10]
    gamma_labels = ["1%", "5%", "10%"]

    # Build groups
    groups = []

    # Bare GHZ
    bare_items = []
    if ghz_df is not None:
        for g, gl in zip(gammas, gamma_labels):
            bare = ghz_df[(np.isclose(ghz_df["gamma"], g)) &
                          (np.isclose(ghz_df["h"], 0.0))]
            if len(bare) > 0:
                summary = compute_alpha_summary(bare)
                bare_items.append((g, gl, summary, None))
                print(f"  Bare GHZ gamma={gl}: alpha={summary['alpha_mean']:.2f}, "
                      f"conv={summary['conv_rate']:.0f}%")
    if bare_items:
        groups.append(("Bare GHZ", bare_items))

    # ED L=1, L=2, L=3
    for L in [1, 2, 3]:
        ed_items = []
        for g, gl in zip(gammas, gamma_labels):
            ps_sub = ed_ps_df[(ed_ps_df["L"] == L) & (np.isclose(ed_ps_df["gamma"], g))]
            fl_sub = ed_fl_df[(ed_fl_df["L"] == L) & (np.isclose(ed_fl_df["gamma"], g))]
            ps_sum = compute_alpha_summary(ps_sub) if len(ps_sub) > 0 else None
            fl_sum = compute_alpha_summary(fl_sub) if len(fl_sub) > 0 else None
            if ps_sum:
                ed_items.append((g, gl, ps_sum, fl_sum))
                ps_a = ps_sum['alpha_mean']
                fl_a = fl_sum['alpha_mean'] if fl_sum else float('nan')
                fl_conv = fl_sum['conv_rate'] if fl_sum else 0
                print(f"  ED L={L} gamma={gl}: PS alpha={ps_a:.2f}, "
                      f"FL alpha={fl_a:.2f}, "
                      f"PS conv={ps_sum['conv_rate']:.0f}%, "
                      f"FL conv={fl_conv:.0f}%")
        if ed_items:
            groups.append((f"ED $L\\!={L}$", ed_items))

    # Panel labels
    ax_alpha.text(-0.14, 1.02, "(a)", transform=ax_alpha.transAxes,
                  fontweight="bold", fontsize=9, va="bottom")
    ax_conv.text(-0.14, 1.02, "(b)", transform=ax_conv.transAxes,
                 fontweight="bold", fontsize=9, va="bottom")

    # Bar layout
    bar_width = 0.35
    gap_within_pair = 0.05
    gap_between_gamma = 0.55
    gap_between_groups = 0.85

    positions = []
    group_ranges = []
    x = 0

    for gi, (group_label, items) in enumerate(groups):
        group_start = x
        for g_val, g_label, ps_sum, fl_sum in items:
            color = GAMMA_COLORS[g_val]
            if fl_sum is not None:
                x_ps = x
                x_fl = x + bar_width + gap_within_pair
                positions.append((x_ps, x_fl, gi, g_label, color, ps_sum, fl_sum))
                x = x_fl + bar_width + gap_between_gamma
            else:
                # Bare GHZ: single bar, wider gap so tick labels don't collide
                positions.append((x, None, gi, g_label, color, ps_sum, None))
                x += bar_width + gap_between_gamma + 0.35
        group_end = x - gap_between_gamma
        group_ranges.append((group_start, group_end, group_label))
        x += gap_between_groups

    # Draw bars
    edge_c = "#504945"
    for entry in positions:
        x_ps, x_fl, gi, gl, color, ps_sum, fl_sum = entry
        if fl_sum is None:
            # Bare GHZ: single bar, same width as individual bars
            ax_alpha.bar(x_ps, ps_sum["alpha_mean"], bar_width,
                         yerr=ps_sum["alpha_sem"], capsize=2,
                         color=color, edgecolor=edge_c, linewidth=0.4,
                         error_kw={"linewidth": 0.8, "color": edge_c},
                         zorder=3)
            ax_conv.bar(x_ps, ps_sum["conv_rate"], bar_width,
                        color=color, edgecolor=edge_c, linewidth=0.4,
                        zorder=3)
        else:
            # PS bar (solid fill)
            ax_alpha.bar(x_ps, ps_sum["alpha_mean"], bar_width,
                         yerr=ps_sum["alpha_sem"], capsize=2,
                         color=color, edgecolor=edge_c, linewidth=0.4,
                         error_kw={"linewidth": 0.8, "color": edge_c},
                         zorder=3)
            # FL bar (hatched)
            ax_alpha.bar(x_fl, fl_sum["alpha_mean"], bar_width,
                         yerr=fl_sum["alpha_sem"], capsize=2,
                         color=color, edgecolor=edge_c, linewidth=0.4,
                         hatch="//////",
                         error_kw={"linewidth": 0.8, "color": edge_c},
                         zorder=3)
            # Convergence
            ax_conv.bar(x_ps, ps_sum["conv_rate"], bar_width,
                        color=color, edgecolor=edge_c, linewidth=0.4,
                        zorder=3)
            ax_conv.bar(x_fl, fl_sum["conv_rate"], bar_width,
                        color=color, edgecolor=edge_c, linewidth=0.4,
                        hatch="//////", zorder=3)

    # Reference lines (labels as right-side annotations to avoid bar overlap)
    ref_c = COLORS["ref_line"]
    ax_alpha.axhline(y=1.0, color=ref_c, linestyle="--",
                     linewidth=0.7, zorder=1)
    ax_alpha.axhline(y=2.0, color=ref_c, linestyle=":",
                     linewidth=0.7, zorder=1)
    # Annotate on right edge, outside the bar area
    ax_alpha.text(1.02, 1.0, r"HL", fontsize=7, color=ref_c,
                  va="center", ha="left",
                  transform=ax_alpha.get_yaxis_transform(), clip_on=False)
    ax_alpha.text(1.02, 2.0, r"SQL", fontsize=7, color=ref_c,
                  va="center", ha="left",
                  transform=ax_alpha.get_yaxis_transform(), clip_on=False)

    # X-axis: gamma labels per cluster
    gamma_tick_pos = []
    gamma_tick_labels = []
    for entry in positions:
        x_ps, x_fl = entry[0], entry[1]
        gl = entry[3]
        if x_fl is not None:
            center = (x_ps + x_fl + bar_width) / 2
        else:
            center = x_ps + bar_width / 2
        gamma_tick_pos.append(center)
        gamma_tick_labels.append(gl)

    ax_conv.set_xticks(gamma_tick_pos)
    ax_conv.set_xticklabels(gamma_tick_labels, fontsize=6.5)

    # Group labels below x-axis (tied to axis coordinates)
    for gstart, gend, glabel in group_ranges:
        mid = (gstart + gend) / 2
        ax_conv.text(mid, -0.35, glabel, ha="center", va="top",
                     fontsize=7.5, fontweight="bold",
                     transform=ax_conv.get_xaxis_transform(),
                     clip_on=False)

    # Group separators
    for i in range(1, len(group_ranges)):
        gstart = group_ranges[i][0]
        sep_x = gstart - gap_between_groups / 2
        for a in (ax_alpha, ax_conv):
            a.axvline(x=sep_x, color="#d5c4a1", linewidth=0.5,
                      linestyle=":", zorder=0)

    # Shared x-label
    ax_conv.set_xlabel(r"Noise rate $\gamma$", fontsize=9, labelpad=22)

    # Y-axis formatting
    ax_alpha.set_ylabel(r"Scaling exponent $\alpha$")
    ax_alpha.set_ylim(0.8, 2.2)
    ax_alpha.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_conv.set_ylabel("Converged (%)")
    ax_conv.set_ylim(50, 105)

    # Legend: noise colors + fill style (interleaved for column-first layout)
    legend_elements = [
        Patch(facecolor=COLORS["gamma_1"], edgecolor=edge_c,
              linewidth=0.4, label=r"$\gamma{=}1\%$"),
        Patch(facecolor="white", edgecolor=edge_c,
              linewidth=0.4, label="Post-selection"),
        Patch(facecolor=COLORS["gamma_5"], edgecolor=edge_c,
              linewidth=0.4, label=r"$\gamma{=}5\%$"),
        Patch(facecolor="white", edgecolor=edge_c,
              linewidth=0.4, hatch="//////", label="Full-likelihood"),
        Patch(facecolor=COLORS["gamma_10"], edgecolor=edge_c,
              linewidth=0.4, label=r"$\gamma{=}10\%$"),
    ]
    ax_alpha.legend(handles=legend_elements, loc="lower left",
                    bbox_to_anchor=(0.0, 1.02), ncol=3,
                    fontsize=6.5, columnspacing=0.6, handletextpad=0.3,
                    handlelength=1.2, labelspacing=0.3,
                    borderaxespad=0, frameon=True, fancybox=False,
                    edgecolor="none", facecolor="white")

    fig.savefig(outdir / "fig2_inference_comparison.pdf", dpi=300,
                bbox_inches="tight")
    fig.savefig(outdir / "fig2_inference_comparison.png", dpi=300,
                bbox_inches="tight")
    print(f"  Saved: {outdir}/fig2_inference_comparison.pdf")
    plt.close()


# =========================================================================
# FIGURE 3: Combined Protocol
# =========================================================================

def figure3_combined_protocol(combined_df, outdir):
    """Figure 5: combined protocol T vs epsilon under joint noise."""
    print("\n--- Figure 3: Combined protocol ---")

    if combined_df is None or len(combined_df) == 0:
        print("  SKIP: no combined protocol data")
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Simplified: L=1 at two noise levels + L=2 in distinct color
    l2_color = "#504945"  # dark gray, distinct from data colors
    combined_configs = [
        (1, 0.01, 0.01, COLORS["gamma_1"],  "-",         1.2,
         r"$L{=}1$, $\gamma{=}1\%$"),
        (1, 0.10, 0.01, COLORS["gamma_10"], (0, (3, 1, 1, 1)), 1.2,
         r"$L{=}1$, $\gamma{=}10\%$"),
        (2, 0.10, 0.01, l2_color,           (0, (4, 2)), 1.4,
         r"$L{=}2$, $\gamma{=}10\%$"),
    ]

    comb_results = []
    for L, gamma, sigma, color, ls, lw, label in combined_configs:
        mask = ((combined_df["L"] == L) &
                (np.isclose(combined_df["gamma"], gamma)) &
                (np.isclose(combined_df["sigma_epsilon"], sigma)))
        sub = combined_df[mask]
        if len(sub) == 0:
            print(f"  SKIP: {label} (no data)")
            continue

        fits = fit_alpha_per_seed(sub)
        conv = sub[sub["converged"] == True]
        result = plot_fit_line(ax, fits, conv, color, ls, linewidth=lw,
                              ci_alpha=0.15)
        if result:
            avg_accept = sub["acceptance_rate"].mean()
            print(f"  {label}: alpha = {result['alpha']:.3f} +/- {result['sem']:.3f}, "
                  f"accept = {avg_accept:.1%}")
            comb_results.append((result, color, label, ls, lw))

    setup_loglog_axes(ax)
    ax.set_ylim(1e2, 1e9)

    # Alpha values in legend labels (annotations would overlap since all ~1.1)
    legend_elements = []
    for result, color, label, ls, lw in comb_results:
        legend_elements.append(
            Line2D([0], [0], color=color, ls=ls, lw=lw,
                   label=rf"{label}, $\alpha = {result['alpha']:.2f}$"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              handlelength=2.0, handletextpad=0.5, labelspacing=0.4)

    fig.savefig(outdir / "fig3_combined_protocol.pdf", dpi=300,
                bbox_inches="tight")
    fig.savefig(outdir / "fig3_combined_protocol.png", dpi=300,
                bbox_inches="tight")
    print(f"  Saved: {outdir}/fig3_combined_protocol.pdf")
    plt.close()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Redesigned numerics plots")
    parser.add_argument("--outdir", default="../plots", help="Output directory")
    parser.add_argument("--ghz-csv", default=None)
    parser.add_argument("--ed-csv", default=None, help="ED post-selection CSV")
    parser.add_argument("--ed-fl-csv", default=None, help="ED full-likelihood CSV")
    parser.add_argument("--combined-csv", default=None, help="Combined post-selection CSV")
    parser.add_argument("--combined-fl-csv", default=None, help="Combined full-likelihood CSV")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def load_csv(arg_path, default_name):
        path = Path(arg_path) if arg_path else script_dir / "results" / default_name
        if path.exists():
            df = pd.read_csv(path)
            print(f"Loaded {default_name}: {len(df):,} rows")
            return df
        print(f"WARNING: {path} not found")
        return None

    ghz_df = load_csv(args.ghz_csv, "bare_ghz.csv")
    if ghz_df is not None and "mode" in ghz_df.columns:
        ghz_df = ghz_df[ghz_df["mode"] == "ghz"]

    ed_ps_df = load_csv(args.ed_csv, "ed_postselect.csv")
    ed_fl_df = load_csv(args.ed_fl_csv, "ed_full_likelihood.csv")
    combined_ps_df = load_csv(args.combined_csv, "combined_postselect.csv")

    if ghz_df is not None or ed_fl_df is not None:
        figure1_error_detection(ghz_df, ed_fl_df, outdir)
    else:
        print("\nFigure 1: SKIP (no data)")

    if ed_ps_df is not None and ed_fl_df is not None:
        figure2_inference_comparison(ghz_df, ed_ps_df, ed_fl_df, outdir)
    else:
        print("\nFigure 2: SKIP (need both ED post-selection and full-likelihood)")

    if combined_ps_df is not None:
        figure3_combined_protocol(combined_ps_df, outdir)
    else:
        print("\nFigure 3: SKIP (no combined data)")

    print("\nDone.")


if __name__ == "__main__":
    main()
