#!/usr/bin/env python3
"""
Three-row visualization of GOES-16 cloud particle forecasting.

Row 1: Ground-truth cloud density at selected time steps
Row 2: Velocity forecast density (blue) vs ground truth contour (grey)
Row 3: Positions forecast density (red) vs ground truth contour (grey)

Particles are converted to smooth density fields via KDE so the
output actually looks like clouds rather than scattered dots.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[1]

# ── data paths (500-particle GOES) ──
VEL_DIR  = REPO / "forecast_output/goes_cloud_500_N_500/snapshot_begin/frac100/velocity_resFromOrigN_100pct_pipeline_velocity"
POS_DIR  = REPO / "forecast_output/goes_cloud_500_N_500/snapshot_begin/frac100/positions_resFromOrigN_100pct_pipeline_positions"
TRAJ     = REPO / "results/goes_cloud_500_N_500/trajectory.npy"
SUMMARY  = REPO / "pipeline_run_summary.json"

# ── load ──
traj     = np.load(TRAJ)
vel_pred = np.load(VEL_DIR / "predicted_measures_X.npy")
vel_true = np.load(VEL_DIR / "true_measures_X.npy")
pos_pred = np.load(POS_DIR / "predicted_measures_X.npy")
pos_true = np.load(POS_DIR / "true_measures_X.npy")
vel_warm = np.load(VEL_DIR / "warm_start_info.npy")

with open(SUMMARY) as fh:
    summary = json.load(fh)
vel_sink = np.array(summary.get("sinkhorn_per_step_velocity", []))
pos_sink = np.array(summary.get("sinkhorn_per_step_positions", []))

vel_ws   = int(vel_warm[0])
F_vel    = vel_pred.shape[0]
F_pos    = pos_pred.shape[0]
fc_start = vel_ws + 1

# ── column selection: 1 training + 6 forecast ──
n_cols = 7
F_show = min(F_vel, F_pos)
if F_show <= n_cols - 1:
    fidxs = list(range(F_show))
else:
    fidxs = sorted(set(
        int(round(i * (F_show - 1) / (n_cols - 2))) for i in range(n_cols - 1)
    ))
col_abs = [vel_ws] + [fc_start + i for i in fidxs]
col_abs = col_abs[:n_cols]
n_cols  = len(col_abs)

# ── KDE grid ──
GRID_RES = 120
xg = np.linspace(0, 1, GRID_RES)
yg = np.linspace(0, 1, GRID_RES)
XX, YY = np.meshgrid(xg, yg)
grid_pts = np.vstack([XX.ravel(), YY.ravel()])   # (2, GRID_RES^2)

# Shared bandwidth for consistent appearance across panels
KDE_BW = 0.04


def particles_to_density(pts, bw=KDE_BW):
    """KDE of (N,2) particles onto the grid. Returns (GRID_RES, GRID_RES)."""
    try:
        kde = gaussian_kde(pts.T, bw_method=bw)
        Z = kde(grid_pts).reshape(GRID_RES, GRID_RES)
    except np.linalg.LinAlgError:
        Z = np.zeros((GRID_RES, GRID_RES))
    return Z


def w2_at(arr, idx):
    if 0 <= idx < len(arr):
        return np.sqrt(max(arr[idx], 0.0))
    return None


# ── pre-compute all densities ──
print("Computing KDE densities...", flush=True)
gt_densities  = {}
vel_densities = {}
pos_densities = {}
gt_truth_densities = {}   # truth at forecast steps (for contour overlay)

for ci, t_abs in enumerate(col_abs):
    gt_densities[ci] = particles_to_density(traj[t_abs])
    fi = t_abs - fc_start
    if 0 <= fi < F_vel:
        vel_densities[ci] = particles_to_density(vel_pred[fi])
        gt_truth_densities[ci] = particles_to_density(vel_true[fi])
    if 0 <= fi < F_pos:
        pos_densities[ci] = particles_to_density(pos_pred[fi])

# Global density range for consistent colormap
all_gt = [gt_densities[ci] for ci in range(n_cols)]
vmax_gt = np.percentile(np.concatenate([z.ravel() for z in all_gt]), 99)

all_pred = ([vel_densities[ci] for ci in vel_densities] +
            [pos_densities[ci] for ci in pos_densities])
if all_pred:
    vmax_pred = np.percentile(np.concatenate([z.ravel() for z in all_pred]), 99)
else:
    vmax_pred = vmax_gt

vmax = max(vmax_gt, vmax_pred)

# ── colormaps ──
from matplotlib.colors import LinearSegmentedColormap

# Ground truth: white → dark grey (cloud-like)
gt_cmap = LinearSegmentedColormap.from_list("gt_cloud",
    ["#ffffff", "#e0e0e0", "#a0a0a0", "#505050", "#1a1a1a"])

# Velocity: white → blue
vel_cmap = LinearSegmentedColormap.from_list("vel_cloud",
    ["#ffffff", "#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"])

# Positions: white → red
pos_cmap = LinearSegmentedColormap.from_list("pos_cloud",
    ["#ffffff", "#FFCDD2", "#EF5350", "#C62828", "#7f0000"])

# Truth contour colour
CONTOUR_C = "#888888"

# ── figure ──
print("Rendering figure...", flush=True)
fig = plt.figure(figsize=(3.0 * n_cols, 9.4), facecolor="white")
gs = gridspec.GridSpec(3, n_cols, hspace=0.24, wspace=0.06,
                       left=0.07, right=0.98, top=0.87, bottom=0.06)

for ci, t_abs in enumerate(col_abs):
    is_train = (t_abs <= vel_ws)
    fi = t_abs - fc_start

    for ri in range(3):
        ax = fig.add_subplot(gs[ri, ci])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5, length=2)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if ci > 0:
            ax.set_yticklabels([])
        if ri < 2:
            ax.set_xticklabels([])

        # ── Row 0: ground truth density ──
        if ri == 0:
            Z = gt_densities[ci]
            ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower",
                      cmap=gt_cmap, vmin=0, vmax=vmax, aspect="equal",
                      interpolation="bilinear")
            tag = f"t = {t_abs}"
            if is_train:
                tag += "  (train)"
            ax.set_title(tag, fontsize=9, fontweight="bold", pad=5)

        # ── Row 1: velocity forecast density + truth contour ──
        elif ri == 1:
            if ci in vel_densities:
                Z_pred = vel_densities[ci]
                Z_true = gt_truth_densities[ci]
                ax.imshow(Z_pred, extent=[0, 1, 0, 1], origin="lower",
                          cmap=vel_cmap, vmin=0, vmax=vmax, aspect="equal",
                          interpolation="bilinear")
                # truth contour overlay
                ax.contour(XX, YY, Z_true, levels=4, colors=CONTOUR_C,
                           linewidths=0.6, alpha=0.55)
                w2 = w2_at(vel_sink, fi)
                if w2 is not None:
                    ax.text(0.97, 0.97, f"W$_2$={w2:.4f}",
                            transform=ax.transAxes, fontsize=6.5, va="top",
                            ha="right", color="#0D47A1", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec="#1565C0", alpha=0.9, lw=0.5))
            else:
                # training frame: show ground truth in blue tint
                Z = gt_densities[ci]
                ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower",
                          cmap=vel_cmap, vmin=0, vmax=vmax, aspect="equal",
                          interpolation="bilinear", alpha=0.5)
                ax.text(0.5, 0.50, "teacher-forced",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#444", style="italic",
                        bbox=dict(fc="white", alpha=0.7, ec="none"))

        # ── Row 2: positions forecast density + truth contour ──
        elif ri == 2:
            if ci in pos_densities:
                Z_pred = pos_densities[ci]
                Z_true = gt_truth_densities.get(ci, gt_densities[ci])
                ax.imshow(Z_pred, extent=[0, 1, 0, 1], origin="lower",
                          cmap=pos_cmap, vmin=0, vmax=vmax, aspect="equal",
                          interpolation="bilinear")
                ax.contour(XX, YY, Z_true, levels=4, colors=CONTOUR_C,
                           linewidths=0.6, alpha=0.55)
                w2 = w2_at(pos_sink, fi)
                if w2 is not None:
                    ax.text(0.97, 0.97, f"W$_2$={w2:.4f}",
                            transform=ax.transAxes, fontsize=6.5, va="top",
                            ha="right", color="#7f0000", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec="#C62828", alpha=0.9, lw=0.5))
            else:
                Z = gt_densities[ci]
                ax.imshow(Z, extent=[0, 1, 0, 1], origin="lower",
                          cmap=pos_cmap, vmin=0, vmax=vmax, aspect="equal",
                          interpolation="bilinear", alpha=0.5)
                ax.text(0.5, 0.50, "teacher-forced",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#444", style="italic",
                        bbox=dict(fc="white", alpha=0.7, ec="none"))

# ── row labels ──
fig.text(0.022, 0.77, "Ground Truth\n(GOES-16 IR)", fontsize=9.5,
         fontweight="bold", va="center", ha="center", rotation=90, color="#1a1a1a")
fig.text(0.022, 0.49, "LOT Velocity\nForecast", fontsize=9.5,
         fontweight="bold", va="center", ha="center", rotation=90, color="#1565C0")
fig.text(0.022, 0.21, "LOT Position\nForecast", fontsize=9.5,
         fontweight="bold", va="center", ha="center", rotation=90, color="#C62828")

# ── legend ──
legend_els = [
    Line2D([], [], marker="s", ls="none", color="#505050", markersize=7,
           label="Ground-truth cloud density"),
    Line2D([], [], ls="-", color=CONTOUR_C, lw=1, alpha=0.6,
           label="Truth contour (overlay)"),
    Line2D([], [], marker="s", ls="none", color="#1E88E5", markersize=7,
           label="Velocity forecast density"),
    Line2D([], [], marker="s", ls="none", color="#EF5350", markersize=7,
           label="Position forecast density"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=7.5,
           frameon=True, fancybox=True, edgecolor="#ccc",
           bbox_to_anchor=(0.53, -0.005))

# ── title ──
vel_w2 = summary.get("sinkhorn_w2_velocity")
pos_w2 = summary.get("sinkhorn_w2_positions")
fig.text(0.53, 0.95,
         "GOES-16 Cloud Forecasting:  LOT Velocity  vs  LOT Position",
         ha="center", fontsize=12, fontweight="bold")
if vel_w2 and pos_w2:
    winner = "Velocity" if vel_w2 < pos_w2 else "Position"
    ratio = pos_w2 / max(vel_w2, 1e-12)
    fig.text(0.53, 0.915,
             f"Debiased Sinkhorn W$_2$:   "
             f"Velocity = {vel_w2:.4f}     Position = {pos_w2:.4f}     "
             f"({ratio:.2f}x, {winner} wins)",
             ha="center", fontsize=8.5, color="#444")

# ── save ──
out = REPO / "plots" / "goes_cloud_forecast_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved: {out.with_suffix('.pdf')}")
