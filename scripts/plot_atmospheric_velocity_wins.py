#!/usr/bin/env python3
"""
Generate publication-quality plots for atmospheric cloud velocity-vs-positions
experiments.  Reads forecast_output/ for completed runs and produces:

  plots/atmospheric_cloud/error_over_time_*.png    -- per-step L^2(sigma) curves
  plots/atmospheric_cloud/cumulative_error_*.png   -- cumulative error growth
  plots/atmospheric_cloud/particle_snapshots_*.png -- scatter panels at key steps
  plots/atmospheric_cloud/atmospheric_summary_bar.png -- bar chart of all configs
  plots/atmospheric_cloud/error_growth_comparison.png -- overlaid growth rates
  plots/atmospheric_cloud/hero_atmospheric_*.png   -- combined error+snapshots

Usage:
  python scripts/plot_atmospheric_velocity_wins.py
  python scripts/plot_atmospheric_velocity_wins.py --report atmospheric_velocity_wins_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "atmospheric_cloud"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_ROOT = REPO_ROOT / "forecast_output"

C_VEL = "#2166ac"
C_POS = "#b2182b"
C_FILL = "#4daf4a"
C_TRUTH = "#333333"


# ── L^2(sigma) metrics ──────────────────────────────────────

def l2_sigma_per_step(pred, true):
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred, true):
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


# ── Discover runs ────────────────────────────────────────────

def find_paired_runs(exp_dir: Path):
    """Find velocity and positions run directories."""
    vel_dir = pos_dir = None
    for d in sorted(exp_dir.rglob("velocity_*")):
        if d.is_dir() and (d / "predicted_maps.npy").exists():
            vel_dir = d
            break
    for d in sorted(exp_dir.rglob("positions_*")):
        if d.is_dir() and (d / "predicted_maps.npy").exists():
            pos_dir = d
            break
    return vel_dir, pos_dir


def load_maps(run_dir):
    pred = np.load(run_dir / "predicted_maps.npy")
    true = np.load(run_dir / "true_future_maps.npy")
    return pred, true


def discover_atmospheric_experiments():
    """Find all atmospheric/weather experiments in forecast_output/."""
    patterns = [
        "atm_*", "goes_*", "sst180*", "sst365*",
        "goes_cloud_*", "sst180d_*", "sst365d_*",
    ]
    experiments = []
    for pattern in patterns:
        for d in sorted(FORECAST_ROOT.glob(f"{pattern}_N_*")):
            if not d.is_dir():
                continue
            vel_dir, pos_dir = find_paired_runs(d)
            if vel_dir and pos_dir:
                experiments.append((d.name, vel_dir, pos_dir))
    # Deduplicate
    seen = set()
    unique = []
    for name, vd, pd in experiments:
        if name not in seen:
            seen.add(name)
            unique.append((name, vd, pd))
    return unique


# ── Plot: Per-step error curve ───────────────────────────────

def plot_error_over_time(vel_pred, vel_true, pos_pred, pos_true, label, save_path):
    vel_err = l2_sigma_per_step(vel_pred, vel_true)
    pos_err = l2_sigma_per_step(pos_pred, pos_true)
    T = min(len(vel_err), len(pos_err))
    vel_err, pos_err = vel_err[:T], pos_err[:T]
    steps = np.arange(T)

    vel_rmse = l2_sigma_rmse(vel_pred[:T], vel_true[:T])
    pos_rmse = l2_sigma_rmse(pos_pred[:T], pos_true[:T])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, vel_err, color=C_VEL, lw=1.6, alpha=0.9,
            label=f"Velocity (RMSE={vel_rmse:.4f})")
    ax.plot(steps, pos_err, color=C_POS, lw=1.6, alpha=0.9,
            label=f"Positions (RMSE={pos_rmse:.4f})")
    ax.fill_between(steps, vel_err, pos_err,
                     where=pos_err > vel_err,
                     alpha=0.12, color=C_FILL, label="Velocity advantage")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Per-step error -- {label}", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot: Cumulative error ───────────────────────────────────

def plot_cumulative_error(vel_pred, vel_true, pos_pred, pos_true, label, save_path):
    vel_err = l2_sigma_per_step(vel_pred, vel_true)
    pos_err = l2_sigma_per_step(pos_pred, pos_true)
    T = min(len(vel_err), len(pos_err))
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, np.cumsum(vel_err[:T] ** 2), color=C_VEL, lw=1.8, label="Velocity")
    ax.plot(steps, np.cumsum(pos_err[:T] ** 2), color=C_POS, lw=1.8, label="Positions")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"Cumulative $\||\cdot\||^2_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Cumulative error -- {label}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot: Particle snapshots (3 rows x 4 cols) ──────────────

def plot_particle_snapshots(vel_pred, vel_true, pos_pred, pos_true, label, save_path):
    T_vel, T_pos = vel_pred.shape[0], pos_pred.shape[0]
    T = min(T_vel, T_pos)

    snap_frac = [0.0, 0.25, 0.5, 0.85]
    snap_idx = [min(int(f * T), T - 1) for f in snap_frac]

    fig, axes = plt.subplots(3, len(snap_idx), figsize=(4 * len(snap_idx), 10.5))
    s, alpha = 8, 0.55

    for col, t in enumerate(snap_idx):
        # Truth
        ax = axes[0, col]
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s, alpha=alpha, c=C_TRUTH, edgecolors="none")
        ax.set_title(f"$t = {t}$", fontsize=11)
        if col == 0:
            ax.set_ylabel("Ground truth", fontsize=11, fontweight="bold")

        # Velocity
        ax = axes[1, col]
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s + 3, alpha=0.18, c="#aaa", edgecolors="none")
        ax.scatter(vel_pred[t, :, 0], vel_pred[t, :, 1],
                   s=s, alpha=alpha, c=C_VEL, edgecolors="none")
        err = np.sqrt(np.mean(np.sum((vel_pred[t] - vel_true[t]) ** 2, axis=-1)))
        ax.set_title(f"err = {err:.4f}", fontsize=9, color=C_VEL)
        if col == 0:
            ax.set_ylabel("Velocity pred.", fontsize=11, fontweight="bold", color=C_VEL)

        # Positions
        ax = axes[2, col]
        t_p = min(t, T_pos - 1)
        ax.scatter(pos_true[t_p, :, 0], pos_true[t_p, :, 1],
                   s=s + 3, alpha=0.18, c="#aaa", edgecolors="none")
        ax.scatter(pos_pred[t_p, :, 0], pos_pred[t_p, :, 1],
                   s=s, alpha=alpha, c=C_POS, edgecolors="none")
        err = np.sqrt(np.mean(np.sum((pos_pred[t_p] - pos_true[t_p]) ** 2, axis=-1)))
        ax.set_title(f"err = {err:.4f}", fontsize=9, color=C_POS)
        if col == 0:
            ax.set_ylabel("Positions pred.", fontsize=11, fontweight="bold", color=C_POS)

        for row in range(3):
            axes[row, col].set_xlim(-0.05, 1.05)
            axes[row, col].set_ylim(-0.05, 1.05)
            axes[row, col].set_aspect("equal")
            axes[row, col].grid(alpha=0.15)
            axes[row, col].tick_params(labelsize=8)

    fig.suptitle(f"Particle snapshots -- {label}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot: Hero combined (error curve + snapshots) ────────────

def plot_hero_combined(vel_pred, vel_true, pos_pred, pos_true, label, save_path):
    T_vel, T_pos = vel_pred.shape[0], pos_pred.shape[0]
    T = min(T_vel, T_pos)

    vel_err = l2_sigma_per_step(vel_pred[:T], vel_true[:T])
    pos_err = l2_sigma_per_step(pos_pred[:T], pos_true[:T])
    vel_rmse = l2_sigma_rmse(vel_pred[:T], vel_true[:T])
    pos_rmse = l2_sigma_rmse(pos_pred[:T], pos_true[:T])

    n_snaps = 4
    snap_frac = [0.05, 0.25, 0.5, 0.85]
    snap_idx = [min(int(f * T), T - 1) for f in snap_frac]

    fig = plt.figure(figsize=(3.6 * n_snaps, 11))
    gs = GridSpec(3, n_snaps, figure=fig, hspace=0.32, wspace=0.30,
                  height_ratios=[1, 1.1, 1.1])

    # Error curve
    ax_err = fig.add_subplot(gs[0, :])
    steps = np.arange(T)
    ax_err.plot(steps, vel_err, color=C_VEL, lw=1.8, alpha=0.9,
                label=f"Velocity (RMSE = {vel_rmse:.4f})")
    ax_err.plot(steps, pos_err, color=C_POS, lw=1.8, alpha=0.9,
                label=f"Positions (RMSE = {pos_rmse:.4f})")
    ax_err.fill_between(steps, vel_err, pos_err,
                        where=pos_err > vel_err,
                        alpha=0.12, color=C_FILL, label="Velocity advantage")
    ax_err.set_xlabel("Forecast step $t$", fontsize=12)
    ax_err.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax_err.set_title(f"Atmospheric cloud forecast -- {label}", fontsize=14)
    ax_err.legend(fontsize=10, loc="upper left")
    ax_err.grid(alpha=0.25)
    ax_err.set_xlim(0, T - 1)
    for t in snap_idx:
        ax_err.axvline(t, color="gray", ls="--", lw=0.7, alpha=0.5)

    s, alpha_pt = 8, 0.6
    for col_i, t in enumerate(snap_idx):
        # Velocity snapshots
        ax = fig.add_subplot(gs[1, col_i])
        ax.scatter(vel_true[t, :, 0], vel_true[t, :, 1],
                   s=s + 4, alpha=0.25, c="#888", edgecolors="none", zorder=1)
        ax.scatter(vel_pred[t, :, 0], vel_pred[t, :, 1],
                   s=s, alpha=alpha_pt, c=C_VEL, edgecolors="none", zorder=2)
        err = np.sqrt(np.mean(np.sum((vel_pred[t] - vel_true[t]) ** 2, axis=-1)))
        ax.set_title(f"$t={t}$  err={err:.3f}", fontsize=9, color=C_VEL)
        if col_i == 0:
            ax.set_ylabel("Velocity", fontsize=11, fontweight="bold", color=C_VEL)
            ax.legend(handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#888",
                       markersize=6, alpha=0.5, label='Truth'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_VEL,
                       markersize=5, label='Predicted'),
            ], fontsize=7, loc="upper right")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.grid(alpha=0.12); ax.tick_params(labelsize=7)

        # Positions snapshots
        ax = fig.add_subplot(gs[2, col_i])
        t_p = min(t, T_pos - 1)
        ax.scatter(pos_true[t_p, :, 0], pos_true[t_p, :, 1],
                   s=s + 4, alpha=0.25, c="#888", edgecolors="none", zorder=1)
        ax.scatter(pos_pred[t_p, :, 0], pos_pred[t_p, :, 1],
                   s=s, alpha=alpha_pt, c=C_POS, edgecolors="none", zorder=2)
        err = np.sqrt(np.mean(np.sum((pos_pred[t_p] - pos_true[t_p]) ** 2, axis=-1)))
        ax.set_title(f"$t={t}$  err={err:.3f}", fontsize=9, color=C_POS)
        if col_i == 0:
            ax.set_ylabel("Positions", fontsize=11, fontweight="bold", color=C_POS)
            ax.legend(handles=[
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#888",
                       markersize=6, alpha=0.5, label='Truth'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=C_POS,
                       markersize=5, label='Predicted'),
            ], fontsize=7, loc="upper right")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.grid(alpha=0.12); ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot: Summary bar chart from report JSON ────────────────

def plot_summary_bar_from_report(report, save_path):
    results = report["all_results"]
    labels = [r["tag"].replace("atm_sweep_", "").replace("_N_120", "") for r in results]
    vel_vals = [r["velocity_l2_rmse"] for r in results]
    pos_vals = [r["positions_l2_rmse"] for r in results]
    improv = [r["improvement_pct"] for r in results]
    is_win = [r["velocity_wins"] for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n * 1.2), 10),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - width / 2, vel_vals, width, color=C_VEL,
            alpha=0.85, label="Velocity", edgecolor="white", linewidth=0.5)
    ax1.bar(x + width / 2, pos_vals, width, color=C_POS,
            alpha=0.85, label="Positions", edgecolor="white", linewidth=0.5)

    # Highlight winners
    for i, w in enumerate(is_win):
        if w:
            ax1.plot(i - width / 2, vel_vals[i], 'v', color="gold",
                     markersize=8, zorder=5)

    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE", fontsize=12)
    ax1.set_title(f"Atmospheric Cloud Forecasting: Velocity vs Positions "
                  f"({report['n_velocity_wins']}/{report['n_configs_tried']} wins)",
                  fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    colors = [C_FILL if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Velocity improvement (%)", fontsize=12)
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot: Category breakdown ─────────────────────────────────

def plot_category_breakdown(report, save_path):
    """Group results by scenario type and show win rates."""
    results = report["all_results"]

    categories = {}
    for r in results:
        tag = r["tag"]
        if "convective" in tag:
            cat = "Convective cells"
        elif "advection" in tag:
            cat = "Advection (jet)"
        elif "frontal" in tag:
            cat = "Frontal boundary"
        elif "diurnal" in tag:
            cat = "Diurnal cycle"
        elif "goes" in tag:
            cat = "GOES satellite"
        elif "sst180" in tag:
            cat = "SST 180-day"
        elif "sst365" in tag:
            cat = "SST 365-day"
        else:
            cat = "Other"

        if cat not in categories:
            categories[cat] = {"wins": 0, "total": 0, "improvements": []}
        categories[cat]["total"] += 1
        if r["velocity_wins"]:
            categories[cat]["wins"] += 1
        categories[cat]["improvements"].append(r["improvement_pct"])

    cats = list(categories.keys())
    win_rates = [categories[c]["wins"] / max(1, categories[c]["total"]) * 100 for c in cats]
    mean_improv = [np.mean(categories[c]["improvements"]) for c in cats]
    totals = [categories[c]["total"] for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate
    colors = [C_FILL if wr > 50 else ("#ffcc00" if wr > 0 else "#e41a1c") for wr in win_rates]
    bars = ax1.barh(cats, win_rates, color=colors, alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Velocity win rate (%)", fontsize=12)
    ax1.set_title("Win rate by scenario type", fontsize=13)
    ax1.set_xlim(0, 105)
    ax1.axvline(50, color="gray", ls="--", lw=0.8, alpha=0.5)
    for i, (bar, t) in enumerate(zip(bars, totals)):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"n={t}", va="center", fontsize=9)
    ax1.grid(axis="x", alpha=0.2)

    # Mean improvement
    colors2 = [C_FILL if mi > 0 else "#e41a1c" for mi in mean_improv]
    ax2.barh(cats, mean_improv, color=colors2, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Mean velocity improvement (%)", fontsize=12)
    ax2.set_title("Mean improvement by scenario type", fontsize=13)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=None,
                    help="Path to JSON report (atmospheric_velocity_wins_report.json)")
    args = ap.parse_args()

    print("=" * 60)
    print("Atmospheric Cloud Velocity-Wins Plots")
    print("=" * 60)

    # ── Plot from discovered experiments ──
    experiments = discover_atmospheric_experiments()
    print(f"\nFound {len(experiments)} atmospheric experiments in forecast_output/")

    summary_results = []

    for exp_name, vel_dir, pos_dir in experiments:
        print(f"\n[{exp_name}]")

        vel_pred, vel_true = load_maps(vel_dir)
        pos_pred, pos_true = load_maps(pos_dir)

        if vel_pred.shape[0] == vel_true.shape[0] + 1:
            vel_pred = vel_pred[1:]

        T = min(vel_pred.shape[0], pos_pred.shape[0])
        vel_pred, vel_true = vel_pred[:T], vel_true[:T]
        pos_pred, pos_true = pos_pred[:T], pos_true[:T]

        vel_rmse = l2_sigma_rmse(vel_pred, vel_true)
        pos_rmse = l2_sigma_rmse(pos_pred, pos_true)
        improvement = (pos_rmse - vel_rmse) / pos_rmse * 100 if pos_rmse > 0 else 0.0
        wins = vel_rmse < pos_rmse

        status = "WIN" if wins else "loss"
        print(f"  [{status}] vel={vel_rmse:.4f} pos={pos_rmse:.4f} "
              f"improvement={improvement:+.1f}%")

        summary_results.append({
            "tag": exp_name, "vel_rmse": vel_rmse, "pos_rmse": pos_rmse,
            "improvement": improvement, "wins": wins,
        })

        short_tag = exp_name.replace("_N_120", "").replace("_N_200", "").replace("_N_80", "")
        label = short_tag.replace("_", " ")

        plot_error_over_time(vel_pred, vel_true, pos_pred, pos_true,
                             label, PLOTS_DIR / f"error_over_time_{short_tag}.png")
        plot_cumulative_error(vel_pred, vel_true, pos_pred, pos_true,
                              label, PLOTS_DIR / f"cumulative_error_{short_tag}.png")

        if wins:
            plot_particle_snapshots(vel_pred, vel_true, pos_pred, pos_true,
                                    label, PLOTS_DIR / f"particle_snapshots_{short_tag}.png")
            plot_hero_combined(vel_pred, vel_true, pos_pred, pos_true,
                               label, PLOTS_DIR / f"hero_atmospheric_{short_tag}.png")

    # ── Summary from report JSON ──
    report_path = args.report or (REPO_ROOT / "atmospheric_velocity_wins_report.json")
    if report_path.is_file():
        print(f"\nLoading report from {report_path}")
        with open(report_path) as f:
            report = json.load(f)

        plot_summary_bar_from_report(report, PLOTS_DIR / "atmospheric_summary_bar.png")
        plot_category_breakdown(report, PLOTS_DIR / "category_breakdown.png")

    n_wins = sum(1 for r in summary_results if r["wins"])
    print(f"\n{'='*60}")
    print(f"Total: {n_wins}/{len(summary_results)} velocity wins")
    print(f"All plots saved to {PLOTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
