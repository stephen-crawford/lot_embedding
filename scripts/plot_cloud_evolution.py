#!/usr/bin/env python3
"""
Publication-quality graphics showing real SST cloud data evolution with
velocity and positions predictions overlaid at key forecast timesteps.

Produces:
  1. Evolution strip: ground truth cloud morphing over time
  2. Three-row comparison at snapshots: truth / velocity / positions
  3. Overlay panels: truth + velocity + positions on same axes
  4. Hero figure: error curve + overlay snapshots for the best config
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "cloud_evolution"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Sinkhorn per-step ─────────────────────────────────────────

import ot as pot

def sinkhorn_div(Xa, Xb, w, eps):
    Xa, Xb = Xa.astype(np.float64), Xb.astype(np.float64)
    M_ab = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)
    M_aa = np.sum((Xa[:, None, :] - Xa[None, :, :]) ** 2, axis=-1)
    M_bb = np.sum((Xb[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)
    W_ab = float(pot.sinkhorn2(w, w, M_ab, reg=eps, numItermax=5000))
    W_aa = float(pot.sinkhorn2(w, w, M_aa, reg=eps, numItermax=5000))
    W_bb = float(pot.sinkhorn2(w, w, M_bb, reg=eps, numItermax=5000))
    return max(W_ab - 0.5 * (W_aa + W_bb), 0.0)

def sinkhorn_per_step(pred, true, eps):
    T, N, d = pred.shape
    w = np.full(N, 1.0 / N, dtype=np.float64)
    return np.array([sinkhorn_div(pred[t], true[t], w, eps) for t in range(T)])


# ── Load experiment data ──────────────────────────────────────

def load_experiment(tag):
    """Load vel/pos predicted and true measures for a given experiment tag."""
    fout = REPO_ROOT / "forecast_output"
    vel_dir = pos_dir = None
    for d in sorted(fout.glob(f"{tag}*/**/velocity_*")):
        if (d / "predicted_measures_X.npy").exists():
            vel_dir = d; break
    for d in sorted(fout.glob(f"{tag}*/**/positions_*")):
        if (d / "predicted_measures_X.npy").exists():
            pos_dir = d; break
    if vel_dir is None or pos_dir is None:
        raise FileNotFoundError(f"Cannot find output for {tag}")

    vp = np.load(vel_dir / "predicted_measures_X.npy")
    vt = np.load(vel_dir / "true_measures_X.npy")
    pp = np.load(pos_dir / "predicted_measures_X.npy")
    pt = np.load(pos_dir / "true_measures_X.npy")
    Tf = min(vp.shape[0], pp.shape[0])
    return vp[:Tf], vt[:Tf], pp[:Tf], pt[:Tf]


# ── Color helpers ─────────────────────────────────────────────

C_TRUTH = "#333333"
C_VEL = "#2166ac"
C_POS = "#b2182b"
C_FILL = "#4daf4a"


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Ground truth cloud evolution strip
# ═══════════════════════════════════════════════════════════════

def plot_truth_evolution(true_X, n_cols=8, save_path=None):
    T = true_X.shape[0]
    snap_idx = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.8))
    for col, t in enumerate(snap_idx):
        ax = axes[col]
        ax.scatter(true_X[t, :, 0], true_X[t, :, 1],
                   s=4, alpha=0.5, c=C_TRUTH, edgecolors="none")
        ax.set_title(f"Day {t}", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("SST particles", fontsize=9)

    fig.suptitle("Real SST Cloud Evolution (Gulf of Mexico, NOAA OISST — 165 days)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Three-row comparison (truth / velocity / positions)
# ═══════════════════════════════════════════════════════════════

def plot_three_row(vp, vt, pp, pt, snap_frac, label, eps, save_path=None):
    Tf = vp.shape[0]
    snap_idx = [min(int(f * Tf), Tf - 1) for f in snap_frac]
    n = len(snap_idx)
    N = vt.shape[1]
    w = np.full(N, 1.0 / N, dtype=np.float64)

    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 9.5))
    s, alpha = 5, 0.5

    for col, t in enumerate(snap_idx):
        # Row 0: Truth
        ax = axes[0, col]
        ax.scatter(vt[t, :, 0], vt[t, :, 1], s=s, alpha=alpha,
                   c=C_TRUTH, edgecolors="none")
        ax.set_title(f"Day {t}", fontsize=10)
        if col == 0:
            ax.set_ylabel("Ground truth", fontsize=11, fontweight="bold")

        # Row 1: Velocity prediction
        ax = axes[1, col]
        ax.scatter(vt[t, :, 0], vt[t, :, 1], s=s + 2, alpha=0.15,
                   c="#aaaaaa", edgecolors="none")
        ax.scatter(vp[t, :, 0], vp[t, :, 1], s=s, alpha=alpha,
                   c=C_VEL, edgecolors="none")
        sd = sinkhorn_div(vp[t], vt[t], w, eps)
        ax.set_title(f"$S_\\varepsilon$ = {sd:.4f}", fontsize=8.5, color=C_VEL)
        if col == 0:
            ax.set_ylabel("Velocity pred.", fontsize=11,
                          fontweight="bold", color=C_VEL)

        # Row 2: Positions prediction
        ax = axes[2, col]
        ax.scatter(pt[t, :, 0], pt[t, :, 1], s=s + 2, alpha=0.15,
                   c="#aaaaaa", edgecolors="none")
        ax.scatter(pp[t, :, 0], pp[t, :, 1], s=s, alpha=alpha,
                   c=C_POS, edgecolors="none")
        sd = sinkhorn_div(pp[t], pt[t], w, eps)
        ax.set_title(f"$S_\\varepsilon$ = {sd:.4f}", fontsize=8.5, color=C_POS)
        if col == 0:
            ax.set_ylabel("Positions pred.", fontsize=11,
                          fontweight="bold", color=C_POS)

        for row in range(3):
            axes[row, col].set_xlim(-0.02, 1.02)
            axes[row, col].set_ylim(-0.02, 1.02)
            axes[row, col].set_aspect("equal")
            axes[row, col].grid(alpha=0.1)
            axes[row, col].tick_params(labelsize=6)

    fig.suptitle(f"Forecast comparison — {label}", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Overlay panels (truth + vel + pos on same axes)
# ═══════════════════════════════════════════════════════════════

def plot_overlay(vp, vt, pp, pt, snap_frac, label, eps, save_path=None):
    Tf = vp.shape[0]
    snap_idx = [min(int(f * Tf), Tf - 1) for f in snap_frac]
    n = len(snap_idx)
    N = vt.shape[1]
    w = np.full(N, 1.0 / N, dtype=np.float64)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.8))
    if n == 1:
        axes = [axes]

    for col, t in enumerate(snap_idx):
        ax = axes[col]
        # Truth as larger, semi-transparent circles
        ax.scatter(vt[t, :, 0], vt[t, :, 1], s=18, alpha=0.2,
                   c=C_TRUTH, edgecolors="none", zorder=1)
        # Velocity on top
        ax.scatter(vp[t, :, 0], vp[t, :, 1], s=6, alpha=0.6,
                   c=C_VEL, edgecolors="none", zorder=3)
        # Positions
        ax.scatter(pp[t, :, 0], pp[t, :, 1], s=6, alpha=0.6,
                   c=C_POS, edgecolors="none", marker="^", zorder=2)

        sd_v = sinkhorn_div(vp[t], vt[t], w, eps)
        sd_p = sinkhorn_div(pp[t], pt[t], w, eps)
        ax.set_title(f"Day {t}\nvel $S_\\varepsilon$={sd_v:.4f}  "
                     f"pos $S_\\varepsilon$={sd_p:.4f}", fontsize=8)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); ax.grid(alpha=0.1)
        ax.tick_params(labelsize=6)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_TRUTH,
               markersize=8, alpha=0.5, label='Truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VEL,
               markersize=6, label='Velocity'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=C_POS,
               markersize=6, label='Positions'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.suptitle(f"Overlay: Truth + Velocity + Positions — {label}", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Hero figure — error curves + overlay snapshots
# ═══════════════════════════════════════════════════════════════

def plot_hero(vp, vt, pp, pt, label, eps, save_path=None):
    Tf = vp.shape[0]
    N = vt.shape[1]
    w = np.full(N, 1.0 / N, dtype=np.float64)

    print(f"    Computing per-step Sinkhorn for hero figure ({Tf} steps)...")
    vel_sink = sinkhorn_per_step(vp, vt, eps)
    pos_sink = sinkhorn_per_step(pp, pt, eps)

    n_snaps = 5
    snap_frac = [0.02, 0.15, 0.35, 0.6, 0.9]
    snap_idx = [min(int(f * Tf), Tf - 1) for f in snap_frac]

    fig = plt.figure(figsize=(3.5 * n_snaps, 9))
    gs = GridSpec(3, n_snaps, figure=fig, hspace=0.30, wspace=0.25,
                  height_ratios=[1.0, 1.1, 1.1])

    # ── Top row: Sinkhorn error curves ──
    ax_err = fig.add_subplot(gs[0, :])
    steps = np.arange(Tf)
    ax_err.plot(steps, vel_sink, color=C_VEL, lw=1.5, alpha=0.9,
                label=f"Velocity (mean $S_\\varepsilon$ = {np.mean(vel_sink):.4f})")
    ax_err.plot(steps, pos_sink, color=C_POS, lw=1.5, alpha=0.9,
                label=f"Positions (mean $S_\\varepsilon$ = {np.mean(pos_sink):.4f})")
    ax_err.fill_between(steps, vel_sink, pos_sink,
                        where=pos_sink > vel_sink,
                        alpha=0.12, color=C_FILL, label="Velocity advantage")
    for t in snap_idx:
        ax_err.axvline(t, color="gray", ls="--", lw=0.6, alpha=0.4)
    ax_err.set_xlabel("Forecast step (days)", fontsize=11)
    ax_err.set_ylabel("Sinkhorn divergence $S_\\varepsilon$", fontsize=11)
    ax_err.set_title(f"Real SST — {label}", fontsize=13)
    ax_err.legend(fontsize=9, loc="upper left"); ax_err.grid(alpha=0.2)
    ax_err.set_xlim(0, Tf - 1)

    s_pt, alpha_pt = 5, 0.5

    # ── Middle row: velocity snapshots ──
    for col, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[1, col])
        ax.scatter(vt[t, :, 0], vt[t, :, 1], s=s_pt + 4, alpha=0.18,
                   c="#999999", edgecolors="none", zorder=1)
        ax.scatter(vp[t, :, 0], vp[t, :, 1], s=s_pt, alpha=alpha_pt,
                   c=C_VEL, edgecolors="none", zorder=2)
        sd = sinkhorn_div(vp[t], vt[t], w, eps)
        ax.set_title(f"day {t}   $S_\\varepsilon$={sd:.4f}", fontsize=8, color=C_VEL)
        if col == 0:
            ax.set_ylabel("Velocity", fontsize=10, fontweight="bold", color=C_VEL)
            legend_el = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#999999",
                       markersize=6, alpha=0.4, label='Truth'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VEL,
                       markersize=5, label='Predicted'),
            ]
            ax.legend(handles=legend_el, fontsize=6, loc="upper right")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); ax.grid(alpha=0.08); ax.tick_params(labelsize=5)

    # ── Bottom row: positions snapshots ──
    for col, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[2, col])
        ax.scatter(pt[t, :, 0], pt[t, :, 1], s=s_pt + 4, alpha=0.18,
                   c="#999999", edgecolors="none", zorder=1)
        ax.scatter(pp[t, :, 0], pp[t, :, 1], s=s_pt, alpha=alpha_pt,
                   c=C_POS, edgecolors="none", zorder=2)
        sd = sinkhorn_div(pp[t], pt[t], w, eps)
        ax.set_title(f"day {t}   $S_\\varepsilon$={sd:.4f}", fontsize=8, color=C_POS)
        if col == 0:
            ax.set_ylabel("Positions", fontsize=10, fontweight="bold", color=C_POS)
            legend_el = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#999999",
                       markersize=6, alpha=0.4, label='Truth'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_POS,
                       markersize=5, label='Predicted'),
            ]
            ax.legend(handles=legend_el, fontsize=6, loc="upper right")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); ax.grid(alpha=0.08); ax.tick_params(labelsize=5)

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight"); plt.close(fig)
        print(f"  Saved {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # Sinkhorn epsilon (same as evaluation script)
    traj_raw = np.load(REPO_ROOT / "sst_data_180d" / "particles.npy")
    sample = traj_raw[0].astype(np.float64)
    M = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    eps = float(np.median(M)) * 0.05
    print(f"Sinkhorn eps = {eps:.6f}")

    # ── Figure 1: Ground truth evolution ──
    print("\n[1] Ground truth evolution strip")
    # Use truth from the raw baseline experiment
    vp, vt, pp, pt = load_experiment("sink_raw_180d")
    plot_truth_evolution(vt, n_cols=8,
                         save_path=PLOTS_DIR / "01_truth_evolution.png")

    # ── Experiments to plot ──
    experiments = [
        ("sink_raw_180d",        "Raw SST 180d (baseline, +12.6% Sinkhorn)"),
        ("sink_ot2_adapt_175",   "OT-2x + adaptive tau T=175 (+86.3% Sinkhorn)"),
        ("sink_adapt_tau_330",   "Adaptive tau T=330 (+84.4% Sinkhorn)"),
        ("sink_hp_tuned",        "Tuned HP on OT-2x+adapt (+72.8% Sinkhorn)"),
        ("sink_365d_ot2_adapt500", "365d OT-2x + adapt T=500 (+62.5% Sinkhorn)"),
        ("sink_raw_180d_uniform_pf", "Uniform per_frame (+55.7% Sinkhorn)"),
    ]

    snap_fracs = [0.02, 0.2, 0.45, 0.7, 0.95]

    for tag, label in experiments:
        print(f"\n[{tag}] {label}")
        try:
            vp, vt, pp, pt = load_experiment(tag)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        short = tag.replace("sink_", "")

        # Figure 2: Three-row comparison
        print("  Three-row comparison...")
        plot_three_row(vp, vt, pp, pt, snap_fracs, label, eps,
                       save_path=PLOTS_DIR / f"02_threerow_{short}.png")

        # Figure 3: Overlay
        print("  Overlay panels...")
        plot_overlay(vp, vt, pp, pt, snap_fracs, label, eps,
                     save_path=PLOTS_DIR / f"03_overlay_{short}.png")

    # ── Figure 4: Hero figures for the two best configs ──
    for tag, label in experiments[:3]:
        print(f"\n[HERO] {label}")
        try:
            vp, vt, pp, pt = load_experiment(tag)
        except FileNotFoundError:
            continue
        short = tag.replace("sink_", "")
        plot_hero(vp, vt, pp, pt, label, eps,
                  save_path=PLOTS_DIR / f"04_hero_{short}.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
