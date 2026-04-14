#!/usr/bin/env python3
"""
Generate plots for velocity-winning configs on real 180-day SST data.
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
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def l2_sigma_per_step(pred, true):
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred, true):
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


def load_run(run_dir):
    pred = np.load(Path(run_dir) / "predicted_maps.npy")
    true = np.load(Path(run_dir) / "true_future_maps.npy")
    return pred, true


def align_maps(vel_pred, vel_true, pos_pred, pos_true):
    """Trim velocity seed row and equalize horizon length."""
    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]
    T = min(vel_pred.shape[0], pos_pred.shape[0])
    return vel_pred[:T], vel_true[:T], pos_pred[:T], pos_true[:T], T


def plot_combined(vel_pred, vel_true, pos_pred, pos_true, T, label, save_path):
    """Error curve on top, velocity snapshots middle row, positions snapshots bottom row."""
    vel_err = l2_sigma_per_step(vel_pred, vel_true)
    pos_err = l2_sigma_per_step(pos_pred, pos_true)
    vel_rmse = l2_sigma_rmse(vel_pred, vel_true)
    pos_rmse = l2_sigma_rmse(pos_pred, pos_true)

    n_snaps = 4
    snap_frac = [0.05, 0.3, 0.6, 0.9]
    snap_idx = [min(int(f * T), T - 1) for f in snap_frac]

    fig = plt.figure(figsize=(3.8 * n_snaps, 11.5))
    gs = GridSpec(3, n_snaps, figure=fig, hspace=0.33, wspace=0.30,
                  height_ratios=[1, 1.1, 1.1])

    # Row 0: error curve
    ax_err = fig.add_subplot(gs[0, :])
    steps = np.arange(T)
    ax_err.plot(steps, vel_err, color="#2166ac", lw=1.8, alpha=0.9,
                label=f"Velocity (RMSE = {vel_rmse:.4f})")
    ax_err.plot(steps, pos_err, color="#b2182b", lw=1.8, alpha=0.9,
                label=f"Positions (RMSE = {pos_rmse:.4f})")
    ax_err.fill_between(steps, vel_err, pos_err,
                        where=pos_err > vel_err,
                        alpha=0.12, color="#4daf4a", label="Velocity advantage")
    ax_err.set_xlabel("Forecast step (days)", fontsize=12)
    ax_err.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax_err.set_title(f"Real SST data — {label}", fontsize=14)
    ax_err.legend(fontsize=10, loc="upper left")
    ax_err.grid(alpha=0.25)
    ax_err.set_xlim(0, T - 1)
    for t in snap_idx:
        ax_err.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.5)

    s = 6
    alpha_pt = 0.55

    # Row 1: velocity snapshots
    for col_i, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[1, col_i])
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s + 4, alpha=0.22, c="#888888", edgecolors="none",
                   label="Truth" if col_i == 0 else None, zorder=1)
        ax.scatter(vel_pred[t, :, 0], vel_pred[t, :, 1],
                   s=s, alpha=alpha_pt, c="#2166ac", edgecolors="none",
                   label="Velocity" if col_i == 0 else None, zorder=2)
        err_t = np.sqrt(np.mean(np.sum((vel_pred[t] - vel_true[t]) ** 2, axis=-1)))
        ax.set_title(f"day {t}    err = {err_t:.3f}", fontsize=9, color="#2166ac")
        if col_i == 0:
            ax.set_ylabel("Velocity pred.", fontsize=11,
                          fontweight="bold", color="#2166ac")
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=7)

    # Row 2: positions snapshots
    for col_i, t in enumerate(snap_idx):
        ax = fig.add_subplot(gs[2, col_i])
        ax.scatter(pos_true[t, :, 0], pos_true[t, :, 1],
                   s=s + 4, alpha=0.22, c="#888888", edgecolors="none",
                   label="Truth" if col_i == 0 else None, zorder=1)
        ax.scatter(pos_pred[t, :, 0], pos_pred[t, :, 1],
                   s=s, alpha=alpha_pt, c="#b2182b", edgecolors="none",
                   label="Positions" if col_i == 0 else None, zorder=2)
        err_t = np.sqrt(np.mean(np.sum((pos_pred[t] - pos_true[t]) ** 2, axis=-1)))
        ax.set_title(f"day {t}    err = {err_t:.3f}", fontsize=9, color="#b2182b")
        if col_i == 0:
            ax.set_ylabel("Positions pred.", fontsize=11,
                          fontweight="bold", color="#b2182b")
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
        ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_summary_bar(results, save_path):
    labels = [r["label"] for r in results]
    vel_vals = [r["vel_rmse"] for r in results]
    pos_vals = [r["pos_rmse"] for r in results]
    improv = [r["improvement"] for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, n * 1.8), 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    bars_v = ax1.bar(x - width / 2, vel_vals, width, color="#2166ac",
                     alpha=0.85, label="Velocity", edgecolor="white", linewidth=0.5)
    bars_p = ax1.bar(x + width / 2, pos_vals, width, color="#b2182b",
                     alpha=0.85, label="Positions", edgecolor="white", linewidth=0.5)

    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE", fontsize=12)
    ax1.set_title("Real SST Data (180 days, Gulf of Mexico) — Velocity vs Positions", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    for bar in bars_v:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#2166ac")
    for bar in bars_p:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#b2182b")

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(improv):
        ax2.text(i, v + (0.8 if v > 0 else -1.5), f"{v:+.1f}%",
                 ha="center", va="bottom" if v > 0 else "top", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def main():
    report_path = REPO_ROOT / "sst180_experiment_report.json"
    with open(report_path) as f:
        report = json.load(f)

    # Filter to wins, sorted by improvement
    wins = [r for r in report["all_results"] if r["velocity_wins"]]
    wins.sort(key=lambda r: -r["improvement_pct"])

    print(f"Real SST 180-day data: {len(wins)} velocity-winning configurations")

    SHORT_LABELS = {
        "sst180_uniform_square_per_frame_sc1.0_nc3": "uniform per_frame",
        "sst180_gaussian_iso_per_frame_sc1.0_nc3": "gauss per_frame",
        "sst180_gaussian_iso_per_frame_sc0.5_nc5": "gauss pf sc=.5 nc5",
        "sst180_gaussian_iso_fixed_sc0.5_nc3": "gauss fixed sc=.5",
        "sst180_gaussian_iso_fixed_sc0.5_nc5": "gauss fixed sc=.5 nc5",
        "sst180_gaussian_iso_fixed_sc0.5_nc5_sr0.9_lk0.7_rd0.005": "gauss sc=.5 sr=.9 nc5",
        "sst180_gaussian_iso_fixed_sc0.75_nc3": "gauss fixed sc=.75",
        "sst180_uniform_square_fixed_sc1.0_nc3": "uniform fixed",
    }

    bar_results = []

    for i, r in enumerate(wins):
        tag = r["tag"]
        label = SHORT_LABELS.get(tag, tag.replace("sst180_", ""))
        print(f"\n[{i+1}] {tag}")
        print(f"    vel={r['vel_l2_rmse']:.4f}  pos={r['pos_l2_rmse']:.4f}"
              f"  improvement={r['improvement_pct']:+.1f}%")

        vel_pred, vel_true = load_run(r["vel_dir"])
        pos_pred, pos_true = load_run(r["pos_dir"])
        vp, vt, pp, pt, Tf = align_maps(vel_pred, vel_true, pos_pred, pos_true)

        short_tag = tag.replace("sst180_", "")

        # Combined figure
        plot_combined(vp, vt, pp, pt, Tf, label,
                      PLOTS_DIR / f"sst_real_{short_tag}_combined.png")

        bar_results.append(dict(
            tag=tag, label=label,
            vel_rmse=r["vel_l2_rmse"], pos_rmse=r["pos_l2_rmse"],
            improvement=r["improvement_pct"],
        ))

    # Summary bar
    if bar_results:
        plot_summary_bar(bar_results, PLOTS_DIR / "sst_real_summary_bar.png")

    print(f"\nAll SST plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
