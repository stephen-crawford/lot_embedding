#!/usr/bin/env python3
"""
Generate publication-quality plots comparing VELOCITY vs POSITIONS forecasts
on real-world-style data, using the paper's empirical L^2(sigma) RMSE metric.

Produces:
  plots/error_over_time_{tag}.pdf     — per-timestep L^2(sigma) error curves
  plots/particle_snapshots_{tag}.pdf  — particle scatter at key forecast steps
  plots/summary_bar.pdf               — bar chart across all winning configs
  plots/cumulative_error_{tag}.pdf    — cumulative error growth

Usage:
  python scripts/plot_velocity_wins.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FORECAST_ROOT = REPO_ROOT / "forecast_output"


# ── Paper L^2(sigma) metric ────────────────────────────────────

def l2_sigma_per_step(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-timestep L^2(sigma) error: sqrt( (1/R) sum_i ||pred-true||^2 )."""
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)       # (T, R)
    return np.sqrt(np.mean(sq_norms, axis=-1))   # (T,)


def l2_sigma_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


# ── Discover experiments ───────────────────────────────────────

def find_velocity_run(exp_dir: Path) -> Path | None:
    for d in exp_dir.rglob("velocity_*"):
        if d.is_dir() and (d / "predicted_maps.npy").exists():
            return d
    return None


def find_positions_run(exp_dir: Path) -> Path | None:
    for d in exp_dir.rglob("positions_*"):
        if d.is_dir() and (d / "predicted_maps.npy").exists():
            return d
    return None


def load_run_data(run_dir: Path) -> dict:
    pred_maps = np.load(run_dir / "predicted_maps.npy")
    true_maps = np.load(run_dir / "true_future_maps.npy")
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    sigma = None
    sigma_path = run_dir / "reference_sigma_X.npy"
    if sigma_path.exists():
        sigma = np.load(sigma_path)
    return dict(pred_maps=pred_maps, true_maps=true_maps, meta=meta, sigma=sigma)


# ── Winning experiments from test results ──────────────────────

WINNING_EXPERIMENTS = [
    "test_syn_120p_500s_4c_gaussian_iso_per_frame_sc1.0_N_120",
    "test_syn_120p_500s_4c_gaussian_iso_fixed_sc1.0_N_120",
    "test_syn_120p_400s_3c_gaussian_iso_fixed_sc1.0_N_120",
    "test_syn_120p_500s_4c_gaussian_iso_fixed_sc1.0_sr0.85_lk0.85_rd0.02_N_120",
    "test_syn_120p_400s_3c_gaussian_iso_fixed_sc1.0_sr0.85_lk0.85_rd0.02_N_120",
    "test_syn_120p_400s_3c_gaussian_iso_fixed_sc0.75_N_120",
    "test_syn_120p_400s_3c_gaussian_iso_fixed_sc0.5_sr0.9_lk0.7_rd0.005_N_120",
]

SHORT_LABELS = [
    "500s/4c per_frame",
    "500s/4c fixed",
    "400s/3c fixed",
    "500s/4c sr=.85",
    "400s/3c sr=.85",
    "400s/3c sc=.75",
    "400s/3c sc=.5 sr=.9",
]


def make_short_label(tag: str) -> str:
    """Extract a readable label from the experiment tag."""
    parts = tag.replace("test_syn_120p_", "").replace("_N_120", "")
    parts = parts.replace("gaussian_iso_", "").replace("_", " ")
    return parts


# ── Plot 1: Per-timestep error curves ──────────────────────────

def plot_error_over_time(vel_data, pos_data, tag, label, save_path):
    vel_pred = vel_data["pred_maps"]
    vel_true = vel_data["true_maps"]
    pos_pred = pos_data["pred_maps"]
    pos_true = pos_data["true_maps"]

    # Align shapes: velocity stores seed + forecast, skip maps[0]
    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    vel_err = l2_sigma_per_step(vel_pred, vel_true)
    pos_err = l2_sigma_per_step(pos_pred, pos_true)

    T = min(len(vel_err), len(pos_err))
    vel_err, pos_err = vel_err[:T], pos_err[:T]
    steps = np.arange(T)

    vel_rmse = l2_sigma_rmse(vel_pred[:T], vel_true[:T])
    pos_rmse = l2_sigma_rmse(pos_pred[:T], pos_true[:T])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, vel_err, color="#2166ac", lw=1.6, alpha=0.9,
            label=f"Velocity (RMSE={vel_rmse:.4f})")
    ax.plot(steps, pos_err, color="#b2182b", lw=1.6, alpha=0.9,
            label=f"Positions (RMSE={pos_rmse:.4f})")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Per-step $L^2(\\sigma)$ error — {label}", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 2: Cumulative error ──────────────────────────────────

def plot_cumulative_error(vel_data, pos_data, tag, label, save_path):
    vel_pred = vel_data["pred_maps"]
    vel_true = vel_data["true_maps"]
    pos_pred = pos_data["pred_maps"]
    pos_true = pos_data["true_maps"]

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    vel_err = l2_sigma_per_step(vel_pred, vel_true)
    pos_err = l2_sigma_per_step(pos_pred, pos_true)

    T = min(len(vel_err), len(pos_err))
    vel_cum = np.cumsum(vel_err[:T] ** 2)
    pos_cum = np.cumsum(pos_err[:T] ** 2)
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, vel_cum, color="#2166ac", lw=1.8,
            label="Velocity (cumulative)")
    ax.plot(steps, pos_cum, color="#b2182b", lw=1.8,
            label="Positions (cumulative)")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"Cumulative $\||\cdot\||^2_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Cumulative squared $L^2(\\sigma)$ error — {label}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 3: Particle snapshots ─────────────────────────────────

def plot_particle_snapshots(vel_data, pos_data, tag, label, save_path):
    vel_pred = vel_data["pred_maps"]
    vel_true = vel_data["true_maps"]
    pos_pred = pos_data["pred_maps"]
    pos_true = pos_data["true_maps"]

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    T_vel = vel_pred.shape[0]
    T_pos = pos_pred.shape[0]
    T = min(T_vel, T_pos)

    # Pick 4 snapshot times spread across the forecast
    snap_frac = [0.0, 0.25, 0.5, 0.85]
    snap_idx = [min(int(f * T), T - 1) for f in snap_frac]

    fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 10.5))

    for col, t in enumerate(snap_idx):
        s = 8
        alpha = 0.55

        # Row 0: Ground truth
        ax = axes[0, col]
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s, alpha=alpha, c="#333333", edgecolors="none")
        ax.set_title(f"$t = {t}$", fontsize=11)
        if col == 0:
            ax.set_ylabel("Ground truth", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)

        # Row 1: Velocity prediction
        ax = axes[1, col]
        ax.scatter(vel_pred[t, :, 0], vel_pred[t, :, 1],
                   s=s, alpha=alpha, c="#2166ac", edgecolors="none")
        vel_t_err = np.sqrt(np.mean(np.sum((vel_pred[t] - vel_true[t]) ** 2, axis=-1)))
        ax.set_title(f"err = {vel_t_err:.4f}", fontsize=9, color="#2166ac")
        if col == 0:
            ax.set_ylabel("Velocity pred.", fontsize=11,
                          fontweight="bold", color="#2166ac")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)

        # Row 2: Positions prediction
        ax = axes[2, col]
        ax.scatter(pos_pred[min(t, T_pos - 1), :, 0],
                   pos_pred[min(t, T_pos - 1), :, 1],
                   s=s, alpha=alpha, c="#b2182b", edgecolors="none")
        pos_t_err = np.sqrt(np.mean(np.sum(
            (pos_pred[min(t, T_pos - 1)] - pos_true[min(t, T_pos - 1)]) ** 2, axis=-1)))
        ax.set_title(f"err = {pos_t_err:.4f}", fontsize=9, color="#b2182b")
        if col == 0:
            ax.set_ylabel("Positions pred.", fontsize=11,
                          fontweight="bold", color="#b2182b")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Particle snapshots — {label}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 4: Summary bar chart ──────────────────────────────────

def plot_summary_bar(results: list[dict], save_path: Path):
    labels = [r["label"] for r in results]
    vel_vals = [r["vel_rmse"] for r in results]
    pos_vals = [r["pos_rmse"] for r in results]
    improv = [r["improvement"] for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, n * 1.6), 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    bars_v = ax1.bar(x - width / 2, vel_vals, width, color="#2166ac",
                     alpha=0.85, label="Velocity", edgecolor="white", linewidth=0.5)
    bars_p = ax1.bar(x + width / 2, pos_vals, width, color="#b2182b",
                     alpha=0.85, label="Positions", edgecolor="white", linewidth=0.5)

    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE", fontsize=12)
    ax1.set_title("Velocity vs Positions — all winning configurations", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    # Add value labels on bars
    for bar in bars_v:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#2166ac")
    for bar in bars_p:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#b2182b")

    # Improvement subplot
    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    for i, v in enumerate(improv):
        ax2.text(i, v + (1.5 if v > 0 else -3), f"{v:+.1f}%",
                 ha="center", va="bottom" if v > 0 else "top", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 5: Combined best-case figure ──────────────────────────

def plot_best_case_combined(vel_data, pos_data, tag, label, save_path):
    """Single figure: error curve on top, then velocity and positions snapshot rows."""
    vel_pred = vel_data["pred_maps"]
    vel_true = vel_data["true_maps"]
    pos_pred = pos_data["pred_maps"]
    pos_true = pos_data["true_maps"]

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    T_vel = vel_pred.shape[0]
    T_pos = pos_pred.shape[0]
    T = min(T_vel, T_pos)

    vel_err = l2_sigma_per_step(vel_pred[:T], vel_true[:T])
    pos_err = l2_sigma_per_step(pos_pred[:T], pos_true[:T])
    vel_rmse = l2_sigma_rmse(vel_pred[:T], vel_true[:T])
    pos_rmse = l2_sigma_rmse(pos_pred[:T], pos_true[:T])

    n_snaps = 4
    snap_frac = [0.05, 0.25, 0.5, 0.85]
    snap_idx = [min(int(f * T), T - 1) for f in snap_frac]

    # Layout: 3 rows (error curve, velocity snapshots, positions snapshots)
    # x n_snaps columns
    fig = plt.figure(figsize=(3.6 * n_snaps, 11))
    gs = GridSpec(3, n_snaps, figure=fig, hspace=0.32, wspace=0.30,
                  height_ratios=[1, 1.1, 1.1])

    # ── Row 0: error curve spanning full width ──
    ax_err = fig.add_subplot(gs[0, :])
    steps = np.arange(T)
    ax_err.plot(steps, vel_err, color="#2166ac", lw=1.8, alpha=0.9,
                label=f"Velocity (RMSE = {vel_rmse:.4f})")
    ax_err.plot(steps, pos_err, color="#b2182b", lw=1.8, alpha=0.9,
                label=f"Positions (RMSE = {pos_rmse:.4f})")
    ax_err.fill_between(steps, vel_err, pos_err,
                        where=pos_err > vel_err,
                        alpha=0.12, color="#4daf4a",
                        label="Velocity advantage")
    ax_err.set_xlabel("Forecast step $t$", fontsize=12)
    ax_err.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax_err.set_title(f"Per-step $L^2(\\sigma)$ error and particle snapshots — {label}",
                     fontsize=14)
    ax_err.legend(fontsize=10, loc="upper left")
    ax_err.grid(alpha=0.25)
    ax_err.set_xlim(0, T - 1)

    for t in snap_idx:
        ax_err.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.5)

    # ── Row 1: Velocity prediction snapshots ──
    s = 8
    alpha_pt = 0.6
    for col_i, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[1, col_i])

        # Truth as gray background
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s + 4, alpha=0.25, c="#888888", edgecolors="none",
                   label="Truth" if col_i == 0 else None, zorder=1)
        # Velocity prediction
        ax.scatter(vel_pred[t, :, 0], vel_pred[t, :, 1],
                   s=s, alpha=alpha_pt, c="#2166ac", edgecolors="none",
                   label="Velocity" if col_i == 0 else None, zorder=2)

        vel_t_err = np.sqrt(np.mean(np.sum(
            (vel_pred[t] - vel_true[t]) ** 2, axis=-1)))
        ax.set_title(f"$t = {t}$    err = {vel_t_err:.3f}", fontsize=9,
                     color="#2166ac")
        if col_i == 0:
            ax.set_ylabel("Velocity pred.", fontsize=11,
                          fontweight="bold", color="#2166ac")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=7)
        if col_i == 0:
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)

    # ── Row 2: Positions prediction snapshots ──
    for col_i, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[2, col_i])
        t_pos = min(t, T_pos - 1)

        # Truth as gray background
        ax.scatter(pos_true[t_pos, :, 0], pos_true[t_pos, :, 1],
                   s=s + 4, alpha=0.25, c="#888888", edgecolors="none",
                   label="Truth" if col_i == 0 else None, zorder=1)
        # Positions prediction
        ax.scatter(pos_pred[t_pos, :, 0], pos_pred[t_pos, :, 1],
                   s=s, alpha=alpha_pt, c="#b2182b", edgecolors="none",
                   label="Positions" if col_i == 0 else None, zorder=2)

        pos_t_err = np.sqrt(np.mean(np.sum(
            (pos_pred[t_pos] - pos_true[t_pos]) ** 2, axis=-1)))
        ax.set_title(f"$t = {t}$    err = {pos_t_err:.3f}", fontsize=9,
                     color="#b2182b")
        if col_i == 0:
            ax.set_ylabel("Positions pred.", fontsize=11,
                          fontweight="bold", color="#b2182b")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=7)
        if col_i == 0:
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Main ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generating velocity-wins plots")
    print("=" * 60)

    summary_results = []

    for i, exp_name in enumerate(WINNING_EXPERIMENTS):
        exp_dir = FORECAST_ROOT / exp_name
        if not exp_dir.is_dir():
            print(f"[SKIP] {exp_name}: directory not found")
            continue

        vel_dir = find_velocity_run(exp_dir)
        pos_dir = find_positions_run(exp_dir)

        if vel_dir is None or pos_dir is None:
            print(f"[SKIP] {exp_name}: missing velocity or positions run")
            continue

        label = SHORT_LABELS[i] if i < len(SHORT_LABELS) else make_short_label(exp_name)
        print(f"\n[{i+1}] {exp_name}")
        print(f"    Label: {label}")

        vel_data = load_run_data(vel_dir)
        pos_data = load_run_data(pos_dir)

        # Compute RMSE
        vp = vel_data["pred_maps"]
        vt = vel_data["true_maps"]
        pp = pos_data["pred_maps"]
        pt = pos_data["true_maps"]
        if vp.shape[0] == vt.shape[0] + 1:
            vp = vp[1:]
        T = min(vp.shape[0], pp.shape[0])
        vel_rmse = l2_sigma_rmse(vp[:T], vt[:T])
        pos_rmse = l2_sigma_rmse(pp[:T], pt[:T])
        improvement = (pos_rmse - vel_rmse) / pos_rmse * 100

        print(f"    Vel RMSE: {vel_rmse:.4f}  Pos RMSE: {pos_rmse:.4f}"
              f"  Improvement: {improvement:+.1f}%")

        summary_results.append(dict(
            tag=exp_name, label=label,
            vel_rmse=vel_rmse, pos_rmse=pos_rmse,
            improvement=improvement,
        ))

        # Sanitized tag for filenames
        short_tag = exp_name.replace("test_syn_120p_", "").replace("_N_120", "")

        # Error over time
        plot_error_over_time(
            vel_data, pos_data, exp_name, label,
            PLOTS_DIR / f"error_over_time_{short_tag}.png",
        )

        # Cumulative error
        plot_cumulative_error(
            vel_data, pos_data, exp_name, label,
            PLOTS_DIR / f"cumulative_error_{short_tag}.png",
        )

        # Particle snapshots
        plot_particle_snapshots(
            vel_data, pos_data, exp_name, label,
            PLOTS_DIR / f"particle_snapshots_{short_tag}.png",
        )

        # Combined figure for the best case
        if i == 0:
            plot_best_case_combined(
                vel_data, pos_data, exp_name, label,
                PLOTS_DIR / "best_case_combined.png",
            )

    # Summary bar chart
    if summary_results:
        plot_summary_bar(summary_results, PLOTS_DIR / "summary_bar.png")

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print(f"Total experiments plotted: {len(summary_results)}")


if __name__ == "__main__":
    main()
