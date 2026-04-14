#!/usr/bin/env python3
"""
Generate three-row and overlay cloud evolution plots for real GOES data.

Row 1: Ground truth LOT maps (gray)
Row 2: Velocity prediction (blue) with Sinkhorn error
Row 3: Positions prediction (red) with Sinkhorn error

Also produces:
  - Overlay plot: truth + velocity + positions on same axes
  - Original GOES particles evolution for context
  - Multiple HP configurations for comparison
"""

import json
import sys
from pathlib import Path

import numpy as np
import ot as pot

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_DIR = REPO_ROOT / "plots" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
})

# Colors
C_TRUTH = "#555555"
C_VEL = "#2166ac"
C_POS = "#b2182b"
C_VEL_LIGHT = "#92c5de"
C_POS_LIGHT = "#f4a582"


def sinkhorn_div(mu, nu, reg=0.05):
    a = np.ones(len(mu)) / len(mu)
    b = np.ones(len(nu)) / len(nu)
    M = pot.dist(mu, nu, metric="sqeuclidean")
    W = pot.sinkhorn2(a, b, M, reg, numItermax=200, warn=False)
    Ma = pot.dist(mu, mu, metric="sqeuclidean")
    Wa = pot.sinkhorn2(a, a, Ma, reg, numItermax=200, warn=False)
    Mb = pot.dist(nu, nu, metric="sqeuclidean")
    Wb = pot.sinkhorn2(b, b, Mb, reg, numItermax=200, warn=False)
    return max(0.0, W - 0.5 * Wa - 0.5 * Wb)


def l2_error(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=-1))))


def load_forecast(tag, vel_tag, pos_tag):
    """Load velocity and positions forecast data for a given experiment tag."""
    base = REPO_ROOT / "forecast_output" / f"{tag}_N_200" / "gaussian_iso" / "frac100"

    # Find directories
    vel_dir = None
    pos_dir = None
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and vel_tag in d.name:
                vel_dir = d
            elif d.is_dir() and pos_tag in d.name:
                pos_dir = d

    if vel_dir is None or pos_dir is None:
        return None

    data = {}
    data["vel_pred"] = np.load(vel_dir / "predicted_maps.npy")
    data["vel_true"] = np.load(vel_dir / "true_future_maps.npy")
    data["pos_pred"] = np.load(pos_dir / "predicted_maps.npy")
    data["pos_true"] = np.load(pos_dir / "true_future_maps.npy")
    data["reference"] = np.load(vel_dir / "reference_sigma_X.npy")

    with open(vel_dir / "metadata.json") as f:
        data["vel_meta"] = json.load(f)
    with open(pos_dir / "metadata.json") as f:
        data["pos_meta"] = json.load(f)

    # Align shapes
    if data["vel_pred"].shape[0] == data["vel_true"].shape[0] + 1:
        data["vel_pred"] = data["vel_pred"][1:]

    Tf = min(data["vel_pred"].shape[0], data["pos_pred"].shape[0])
    data["vel_pred"] = data["vel_pred"][:Tf]
    data["vel_true"] = data["vel_true"][:Tf]
    data["pos_pred"] = data["pos_pred"][:Tf]
    data["pos_true"] = data["pos_true"][:Tf]
    data["forecast_steps"] = Tf
    data["warm_steps"] = int(data["vel_meta"].get("warm_steps", 0))

    return data


def load_original_goes():
    p = REPO_ROOT / "goes_data_gp_summer_72h" / "particles.npy"
    if not p.exists():
        return None
    traj = np.load(p)
    # Normalize to [0,1]
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    return np.clip((traj - lo) / span * 0.96 + 0.02, 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# THREE-ROW PLOT
# ══════════════════════════════════════════════════════════════

def plot_three_row(data, n_cols=6, title="", filename="threerow.png",
                   goes_particles=None, cadence_hours=0.5):
    """
    Three-row figure: truth / velocity / positions at selected forecast times.
    """
    Tf = data["forecast_steps"]
    warm = data["warm_steps"]
    vel_pred = data["vel_pred"]
    vel_true = data["vel_true"]
    pos_pred = data["pos_pred"]
    pos_true = data["pos_true"]

    # Select evenly-spaced forecast steps
    indices = np.linspace(0, Tf - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(3, n_cols, figsize=(2.3 * n_cols, 6.6))

    for col, idx in enumerate(indices):
        forecast_hour = (idx + 1) * cadence_hours
        t_abs = warm + idx + 1  # absolute timestep in original trajectory

        # Compute errors at this step
        vel_sink = sinkhorn_div(vel_pred[idx], vel_true[idx])
        pos_sink = sinkhorn_div(pos_pred[idx], pos_true[idx])
        vel_l2 = l2_error(vel_pred[idx], vel_true[idx])
        pos_l2 = l2_error(pos_pred[idx], pos_true[idx])

        # Row 0: Ground truth
        ax = axes[0, col]
        if goes_particles is not None and t_abs < goes_particles.shape[0]:
            ax.scatter(goes_particles[t_abs, :, 0], goes_particles[t_abs, :, 1],
                       s=4, alpha=0.15, c="#cccccc", zorder=1, rasterized=True)
        ax.scatter(vel_true[idx, :, 0], vel_true[idx, :, 1],
                   s=6, alpha=0.7, c=C_TRUTH, zorder=2, rasterized=True)
        ax.set_title(f"+{forecast_hour:.0f}h (t={t_abs})", fontsize=9)
        if col == 0:
            ax.set_ylabel("Ground truth", fontsize=10, color=C_TRUTH, fontweight="bold")

        # Row 1: Velocity prediction
        ax = axes[1, col]
        if goes_particles is not None and t_abs < goes_particles.shape[0]:
            ax.scatter(goes_particles[t_abs, :, 0], goes_particles[t_abs, :, 1],
                       s=4, alpha=0.08, c="#cccccc", zorder=1, rasterized=True)
        ax.scatter(vel_pred[idx, :, 0], vel_pred[idx, :, 1],
                   s=6, alpha=0.7, c=C_VEL, zorder=2, rasterized=True)
        ax.text(0.03, 0.95, f"$S_\\varepsilon$={vel_sink:.4f}",
                transform=ax.transAxes, fontsize=7, va="top",
                color=C_VEL, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_VEL_LIGHT, alpha=0.9))
        if col == 0:
            ax.set_ylabel("Velocity pred.", fontsize=10, color=C_VEL, fontweight="bold")

        # Row 2: Positions prediction
        ax = axes[2, col]
        if goes_particles is not None and t_abs < goes_particles.shape[0]:
            ax.scatter(goes_particles[t_abs, :, 0], goes_particles[t_abs, :, 1],
                       s=4, alpha=0.08, c="#cccccc", zorder=1, rasterized=True)
        ax.scatter(pos_pred[idx, :, 0], pos_pred[idx, :, 1],
                   s=6, alpha=0.7, c=C_POS, zorder=2, rasterized=True)
        ax.text(0.03, 0.95, f"$S_\\varepsilon$={pos_sink:.4f}",
                transform=ax.transAxes, fontsize=7, va="top",
                color=C_POS, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_POS_LIGHT, alpha=0.9))
        if col == 0:
            ax.set_ylabel("Positions pred.", fontsize=10, color=C_POS, fontweight="bold")

        # Common formatting
        for row in range(3):
            ax = axes[row, col]
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", which="both", length=2)
            if row < 2:
                ax.set_xticklabels([])

    # Overall metrics
    vel_rmse = float(data["vel_meta"]["forecast_map_rmse"])
    pos_rmse = float(data["pos_meta"]["forecast_map_rmse"])
    imp = (pos_rmse - vel_rmse) / pos_rmse * 100

    fig.suptitle(f"{title}\n"
                 f"Velocity map RMSE = {vel_rmse:.4f},  "
                 f"Positions map RMSE = {pos_rmse:.4f}  "
                 f"(velocity {imp:+.1f}%)",
                 fontsize=11, y=1.02)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"))
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
    print(f"  {filename}: vel={vel_rmse:.4f} pos={pos_rmse:.4f} ({imp:+.1f}%)")


# ══════════════════════════════════════════════════════════════
# OVERLAY PLOT
# ══════════════════════════════════════════════════════════════

def plot_overlay(data, n_cols=6, title="", filename="overlay.png",
                 goes_particles=None, cadence_hours=0.5):
    """
    Single-row overlay: truth (gray) + velocity (blue) + positions (red).
    """
    Tf = data["forecast_steps"]
    warm = data["warm_steps"]
    vel_pred = data["vel_pred"]
    vel_true = data["vel_true"]
    pos_pred = data["pos_pred"]

    indices = np.linspace(0, Tf - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.8))
    if n_cols == 1:
        axes = [axes]

    for col, idx in enumerate(indices):
        ax = axes[col]
        forecast_hour = (idx + 1) * cadence_hours
        t_abs = warm + idx + 1

        vel_sink = sinkhorn_div(vel_pred[idx], vel_true[idx])
        pos_sink = sinkhorn_div(pos_pred[idx], vel_true[idx])

        # Original particles (faint background)
        if goes_particles is not None and t_abs < goes_particles.shape[0]:
            ax.scatter(goes_particles[t_abs, :, 0], goes_particles[t_abs, :, 1],
                       s=3, alpha=0.08, c="#cccccc", zorder=0, rasterized=True)

        # Truth
        ax.scatter(vel_true[idx, :, 0], vel_true[idx, :, 1],
                   s=10, alpha=0.4, c=C_TRUTH, zorder=1, rasterized=True,
                   label="Truth" if col == 0 else None)
        # Velocity
        ax.scatter(vel_pred[idx, :, 0], vel_pred[idx, :, 1],
                   s=8, alpha=0.6, c=C_VEL, marker="o", zorder=2, rasterized=True,
                   label="Velocity" if col == 0 else None)
        # Positions
        ax.scatter(pos_pred[idx, :, 0], pos_pred[idx, :, 1],
                   s=8, alpha=0.5, c=C_POS, marker="s", zorder=2, rasterized=True,
                   label="Positions" if col == 0 else None)

        ax.set_title(f"+{forecast_hour:.0f}h\n"
                     f"vel $S_\\varepsilon$={vel_sink:.3f}  pos $S_\\varepsilon$={pos_sink:.3f}",
                     fontsize=7.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", which="both", length=2)
        if col == 0:
            ax.legend(fontsize=7, loc="lower left", markerscale=1.5,
                      framealpha=0.9, edgecolor="#cccccc")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"))
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
    print(f"  {filename}")


# ══════════════════════════════════════════════════════════════
# ORIGINAL GOES PARTICLE EVOLUTION
# ══════════════════════════════════════════════════════════════

def plot_goes_evolution(goes_particles, n_cols=8, filename="goes_particles_evolution.png"):
    """Show the raw GOES cloud field evolving over time."""
    T = goes_particles.shape[0]
    indices = np.linspace(0, T - 1, n_cols, dtype=int)

    fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.5))

    for col, idx in enumerate(indices):
        ax = axes[col]
        hour = idx * 0.5
        ax.scatter(goes_particles[idx, :, 0], goes_particles[idx, :, 1],
                   s=6, alpha=0.6, c="#333333", rasterized=True)
        ax.set_title(f"t={idx} ({hour:.0f}h)", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", which="both", length=2)

    fig.suptitle("GOES-16 Great Plains 72h — cloud particle evolution (IR band C13)",
                 fontsize=10, y=1.02)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"))
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
    print(f"  {filename}")


# ══════════════════════════════════════════════════════════════
# ERROR ACCUMULATION COMPARISON
# ══════════════════════════════════════════════════════════════

def plot_error_accumulation(data, title="", filename="error_accumulation.png",
                            cadence_hours=0.5):
    """Side-by-side: per-step L2 + per-step Sinkhorn + cumulative."""
    Tf = data["forecast_steps"]
    vel_pred = data["vel_pred"]
    vel_true = data["vel_true"]
    pos_pred = data["pos_pred"]
    pos_true = data["pos_true"]

    hours = np.arange(Tf) * cadence_hours

    # Per-step L2
    vel_l2 = np.array([l2_error(vel_pred[t], vel_true[t]) for t in range(Tf)])
    pos_l2 = np.array([l2_error(pos_pred[t], pos_true[t]) for t in range(Tf)])

    # Cumulative RMSE
    vel_cum = np.sqrt(np.cumsum(vel_l2 ** 2) / np.arange(1, Tf + 1))
    pos_cum = np.sqrt(np.cumsum(pos_l2 ** 2) / np.arange(1, Tf + 1))

    # Sinkhorn at sampled points
    n_sink = min(20, Tf)
    sink_idx = np.linspace(0, Tf - 1, n_sink, dtype=int)
    vel_sink = np.array([sinkhorn_div(vel_pred[i], vel_true[i]) for i in sink_idx])
    pos_sink = np.array([sinkhorn_div(pos_pred[i], pos_true[i]) for i in sink_idx])
    sink_hours = hours[sink_idx]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    # Panel A: Per-step L2
    ax = axes[0]
    ax.plot(hours, vel_l2, color=C_VEL, linewidth=1.3, label="Velocity")
    ax.plot(hours, pos_l2, color=C_POS, linewidth=1.3, alpha=0.8, label="Positions")
    vel_better = vel_l2 < pos_l2
    ax.fill_between(hours, 0, max(vel_l2.max(), pos_l2.max()) * 1.05,
                     where=vel_better, alpha=0.06, color=C_VEL)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(r"$L^2(\sigma)$ error")
    ax.set_title(r"(a) Per-step $L^2(\sigma)$ error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel B: Sinkhorn
    ax = axes[1]
    ax.plot(sink_hours, vel_sink, "o-", color=C_VEL, linewidth=1.3, markersize=3,
            label="Velocity")
    ax.plot(sink_hours, pos_sink, "s-", color=C_POS, linewidth=1.3, markersize=3,
            alpha=0.8, label="Positions")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(r"Sinkhorn divergence $S_\varepsilon$")
    ax.set_title(r"(b) Sinkhorn divergence $S_\varepsilon$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel C: Cumulative RMSE
    ax = axes[2]
    ax.plot(hours, vel_cum, color=C_VEL, linewidth=1.3, label="Velocity")
    ax.plot(hours, pos_cum, color=C_POS, linewidth=1.3, alpha=0.8, label="Positions")
    vel_total = vel_cum[-1]
    pos_total = pos_cum[-1]
    imp = (pos_total - vel_total) / pos_total * 100
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(r"Cumulative $L^2(\sigma)$ RMSE")
    ax.set_title(f"(c) Cumulative RMSE (velocity {imp:+.0f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=10, y=1.03)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT_DIR / filename.replace(".png", ".pdf"))
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)
    print(f"  {filename}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating cloud evolution plots for paper...")

    goes = load_original_goes()

    # ── Config 1: shared HPs, sr=0.9, lk=0.9, ridge=0.01 ──
    d1 = load_forecast(
        "vdom_shared_sr0.9_lk0.9_rd0.01",
        "vdom_vel_sr0.9_lk0.9_rd0.01",
        "vdom_pos_sr0.9_lk0.9_rd0.01",
    )
    if d1:
        plot_three_row(d1, n_cols=6,
                       title="GOES-16 Great Plains 72h — shared HPs "
                             r"($\rho$=0.9, $\alpha$=0.9, $\lambda$=0.01)",
                       filename="fig_threerow_shared_best.png",
                       goes_particles=goes)
        plot_overlay(d1, n_cols=6,
                     title="GOES-16 GP 72h — truth + velocity + positions overlay",
                     filename="fig_overlay_shared_best.png",
                     goes_particles=goes)
        plot_error_accumulation(d1,
                                title="GOES-16 GP 72h — error accumulation "
                                      r"($\rho$=0.9, $\alpha$=0.9, $\lambda$=0.01)",
                                filename="fig_error_accum_shared_best.png")

    # ── Config 2: HP-swept (best per method) ──
    d2 = load_forecast(
        "vdom_best_gauss_pf_cl48",
        "vdom_vel_gauss_pf_cl48",
        "vdom_pos_gauss_pf_cl48",
    )
    if d2:
        plot_three_row(d2, n_cols=6,
                       title="GOES-16 GP 72h — HP-swept "
                             "(each method selects own best HPs)",
                       filename="fig_threerow_swept.png",
                       goes_particles=goes)
        plot_overlay(d2, n_cols=6,
                     title="GOES-16 GP 72h — HP-swept overlay",
                     filename="fig_overlay_swept.png",
                     goes_particles=goes)
        plot_error_accumulation(d2,
                                title="GOES-16 GP 72h — HP-swept error accumulation",
                                filename="fig_error_accum_swept.png")

    # ── Config 3: shared HPs, different setting for comparison ──
    d3 = load_forecast(
        "vdom_shared_sr0.7_lk0.7_rd0.01",
        "vdom_vel_sr0.7_lk0.7_rd0.01",
        "vdom_pos_sr0.7_lk0.7_rd0.01",
    )
    if d3:
        plot_three_row(d3, n_cols=6,
                       title="GOES-16 GP 72h — shared HPs "
                             r"($\rho$=0.7, $\alpha$=0.7, $\lambda$=0.01)",
                       filename="fig_threerow_shared_low_sr.png",
                       goes_particles=goes)

    # ── GOES particle evolution ──
    if goes is not None:
        plot_goes_evolution(goes, n_cols=8, filename="fig_goes_particles_evolution.png")

    print(f"\nAll plots saved to {OUT_DIR}/")
