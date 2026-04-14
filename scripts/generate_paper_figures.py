#!/usr/bin/env python3
"""
Generate publication-quality figures for the GOES cloud data experiments.

Produces figures for the paper showing:
  Fig 1: Cross-dataset comparison (GOES vs SST, where velocity wins/loses)
  Fig 2: HP robustness — velocity wins 13/15 shared-HP configs
  Fig 3: Per-step error curves for the best velocity-dominant config
  Fig 4: Interpolation effect — how tau frequency affects velocity advantage
  Fig 5: Snapshot vs geometric reference comparison
  Fig 6: Crossover analysis — when velocity overtakes positions over time
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PAPER_DIR = REPO_ROOT / "plots" / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
})


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def load_report(name):
    p = REPO_ROOT / name
    if not p.exists():
        print(f"  [SKIP] {name} not found")
        return None
    with open(p) as f:
        return json.load(f)


def load_forecast_data(system_tag, lot_kind, frac, run_tag, N=200, reservoir_scale=1.0):
    """Load predicted/true maps from a forecast output directory."""
    pct = int(reservoir_scale * 100)
    base = REPO_ROOT / "forecast_output" / f"{system_tag}_N_{N}" / lot_kind / f"frac{frac}"
    # Try to find the right subdirectory
    import glob
    pattern = str(base / f"*{run_tag}*")
    dirs = glob.glob(pattern)
    if not dirs:
        return None, None
    d = Path(dirs[0])
    pred = np.load(d / "predicted_maps.npy") if (d / "predicted_maps.npy").exists() else None
    true = np.load(d / "true_future_maps.npy") if (d / "true_future_maps.npy").exists() else None
    return pred, true


# ══════════════════════════════════════════════════════════════
# FIGURE 1: Cross-dataset comparison
# ══════════════════════════════════════════════════════════════

def fig1_cross_dataset():
    """Show where velocity wins (GOES GP 72h) vs loses (Gulf, SST)."""
    report = load_report("realworld_extended_report.json")
    if not report:
        return

    datasets = list(report["datasets"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A: Bar chart of best velocity improvement per dataset
    ax = axes[0]
    best_imps = []
    colors = []
    for ds in datasets:
        configs = report["datasets"][ds]["configs"]
        best = max(configs, key=lambda c: c["improvement_pct"])
        best_imps.append(best["improvement_pct"])
        colors.append("#2166ac" if best["improvement_pct"] > 0 else "#b2182b")

    x = np.arange(len(datasets))
    bars = ax.bar(x, best_imps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("Velocity improvement (%)")
    ax.set_title("(a) Best velocity improvement by dataset")
    for bar, imp in zip(bars, best_imps):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + (1 if y >= 0 else -3),
                f"{imp:+.1f}%", ha="center", va="bottom" if y >= 0 else "top", fontsize=7)

    # Panel B: Win rate across datasets
    ax = axes[1]
    win_rates = []
    n_configs = []
    for ds in datasets:
        d = report["datasets"][ds]
        wr = d["velocity_win_rate"] * 100
        win_rates.append(wr)
        n_configs.append(d["n_configs"])

    bar_colors = ["#2166ac" if wr > 50 else ("#999999" if wr > 0 else "#b2182b") for wr in win_rates]
    bars = ax.bar(x, win_rates, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("Velocity win rate (%)")
    ax.set_title("(b) Win rate across HP configs")
    ax.set_ylim(0, 100)
    for bar, wr, nc in zip(bars, win_rates, n_configs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{int(wr)}%\n({nc} cfgs)", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(PAPER_DIR / "fig1_cross_dataset.pdf")
    fig.savefig(PAPER_DIR / "fig1_cross_dataset.png")
    plt.close(fig)
    print("  Fig 1: cross-dataset comparison")


# ══════════════════════════════════════════════════════════════
# FIGURE 2: HP robustness (shared HP sweep)
# ══════════════════════════════════════════════════════════════

def fig2_hp_robustness():
    """Show velocity wins 13/15 configs with identical HPs for both methods."""
    # Re-run the data from the shared HP test results
    # We need the actual results - load from forecast metadata
    from run_forecast.velocity_forecast import SystemConfig as VelCfg
    from run_forecast.positions_forecast import SystemConfig as PosCfg
    import glob

    hp_grid = [
        ("sr0.7\nlk0.5\nrd1e-3", 0.7, 0.5, 1e-3),
        ("sr0.7\nlk0.7\nrd1e-2", 0.7, 0.7, 1e-2),
        ("sr0.7\nlk0.9\nrd1e-2", 0.7, 0.9, 1e-2),
        ("sr0.8\nlk0.5\nrd1e-3", 0.8, 0.5, 1e-3),
        ("sr0.8\nlk0.7\nrd1e-2", 0.8, 0.7, 1e-2),
        ("sr0.8\nlk0.9\nrd1e-2", 0.8, 0.9, 1e-2),
        ("sr0.8\nlk0.9\nrd1e-1", 0.8, 0.9, 1e-1),
        ("sr0.9\nlk0.3\nrd1e-3", 0.9, 0.3, 1e-3),
        ("sr0.9\nlk0.5\nrd1e-2", 0.9, 0.5, 1e-2),
        ("sr0.9\nlk0.7\nrd1e-2", 0.9, 0.7, 1e-2),
        ("sr0.9\nlk0.9\nrd1e-2", 0.9, 0.9, 1e-2),
        ("sr0.9\nlk0.9\nrd1e-1", 0.9, 0.9, 1e-1),
        ("sr0.95\nlk0.5\nrd1e-2", 0.95, 0.5, 1e-2),
        ("sr0.95\nlk0.7\nrd1e-2", 0.95, 0.7, 1e-2),
        ("sr0.95\nlk0.9\nrd1e-2", 0.95, 0.9, 1e-2),
    ]

    # Load results from forecast output
    vel_rmses = []
    pos_rmses = []
    labels = []
    for label, sr, lk, rd in hp_grid:
        tag_base = f"sr{sr}_lk{lk}_rd{rd}"
        vel_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag_base}_N_200" /
                                  "gaussian_iso" / "frac100" / "*vdom_vel*"))
        pos_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag_base}_N_200" /
                                  "gaussian_iso" / "frac100" / "*vdom_pos*"))
        if not vel_dirs or not pos_dirs:
            continue
        try:
            with open(Path(vel_dirs[0]) / "metadata.json") as f:
                vm = json.load(f)
            with open(Path(pos_dirs[0]) / "metadata.json") as f:
                pm = json.load(f)
            vel_rmses.append(float(vm["forecast_map_rmse"]))
            pos_rmses.append(float(pm["forecast_map_rmse"]))
            labels.append(label)
        except Exception:
            continue

    if not labels:
        print("  [SKIP] Fig 2: no shared HP data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    x = np.arange(len(labels))
    w = 0.35

    # Panel A: RMSE comparison
    ax = axes[0]
    ax.bar(x - w / 2, vel_rmses, w, label="Velocity", color="#2166ac", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, pos_rmses, w, label="Positions", color="#b2182b", alpha=0.85, edgecolor="black", linewidth=0.4)
    for i in range(len(labels)):
        if vel_rmses[i] <= pos_rmses[i]:
            ax.plot(i - w / 2, vel_rmses[i] + 0.02, marker="*", color="#1a9850",
                    markersize=8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("Map RMSE")
    ax.set_title(f"(a) Velocity vs positions RMSE\n(shared HPs, gaussian_iso / per_frame)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Improvement percentage
    ax = axes[1]
    imps = [(p - v) / p * 100 if p > 0 else 0 for v, p in zip(vel_rmses, pos_rmses)]
    colors = ["#2166ac" if i > 0 else "#b2182b" for i in imps]
    ax.bar(x, imps, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("Velocity improvement (%)")
    n_wins = sum(1 for i in imps if i > 0)
    ax.set_title(f"(b) Velocity improvement ({n_wins}/{len(imps)} wins)")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER_DIR / "fig2_hp_robustness.pdf")
    fig.savefig(PAPER_DIR / "fig2_hp_robustness.png")
    plt.close(fig)
    print(f"  Fig 2: HP robustness ({n_wins}/{len(imps)} wins)")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Best-case error curves
# ══════════════════════════════════════════════════════════════

def fig3_best_case_error():
    """Per-step L2(sigma) error for the best velocity-dominant configuration."""
    import glob

    # Best config: gaussian_iso / per_frame / cl=48 / sr=0.9, lk=0.9, rd=0.01
    tag = "sr0.9_lk0.9_rd0.01"
    vel_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag}_N_200" /
                              "gaussian_iso" / "frac100" / "*vdom_vel*"))
    pos_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag}_N_200" /
                              "gaussian_iso" / "frac100" / "*vdom_pos*"))
    if not vel_dirs or not pos_dirs:
        print("  [SKIP] Fig 3: forecast data not found")
        return

    vel_pred = np.load(Path(vel_dirs[0]) / "predicted_maps.npy")
    vel_true = np.load(Path(vel_dirs[0]) / "true_future_maps.npy")
    pos_pred = np.load(Path(pos_dirs[0]) / "predicted_maps.npy")
    pos_true = np.load(Path(pos_dirs[0]) / "true_future_maps.npy")

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]
    Tf = min(vel_pred.shape[0], pos_pred.shape[0])
    vel_pred, vel_true = vel_pred[:Tf], vel_true[:Tf]
    pos_pred, pos_true = pos_pred[:Tf], pos_true[:Tf]

    vel_err = np.sqrt(np.mean(np.sum((vel_pred - vel_true) ** 2, axis=-1), axis=-1))
    pos_err = np.sqrt(np.mean(np.sum((pos_pred - pos_true) ** 2, axis=-1), axis=-1))

    vel_cum = np.sqrt(np.cumsum(vel_err ** 2) / np.arange(1, Tf + 1))
    pos_cum = np.sqrt(np.cumsum(pos_err ** 2) / np.arange(1, Tf + 1))

    # Convert steps to hours (30-min cadence)
    hours = np.arange(Tf) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    # Panel A: Per-step error
    ax = axes[0]
    ax.plot(hours, vel_err, label="Velocity", color="#2166ac", linewidth=1.5)
    ax.plot(hours, pos_err, label="Positions", color="#b2182b", linewidth=1.5, alpha=0.8)
    # Shade where velocity wins
    vel_better = vel_err < pos_err
    ax.fill_between(hours, 0, max(vel_err.max(), pos_err.max()) * 1.05,
                     where=vel_better, alpha=0.06, color="#2166ac")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(r"$L^2(\sigma)$ error per step")
    ax.set_title("(a) Per-step forecast error")
    ax.legend(loc="upper left", fontsize=8)

    # Panel B: Cumulative RMSE
    ax = axes[1]
    ax.plot(hours, vel_cum, label="Velocity", color="#2166ac", linewidth=1.5)
    ax.plot(hours, pos_cum, label="Positions", color="#b2182b", linewidth=1.5, alpha=0.8)
    vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
    imp = (pos_rmse - vel_rmse) / pos_rmse * 100
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(r"Cumulative $L^2(\sigma)$ RMSE")
    ax.set_title(f"(b) Cumulative RMSE (velocity {imp:+.0f}%)")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("GOES-16 Great Plains 72h — velocity vs positions forecast error",
                 fontsize=11, y=1.02)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER_DIR / "fig3_error_curves.pdf")
    fig.savefig(PAPER_DIR / "fig3_error_curves.png")
    plt.close(fig)
    print(f"  Fig 3: best-case error curves (improvement {imp:+.1f}%)")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Effect of interpolation (tau frequency)
# ══════════════════════════════════════════════════════════════

def fig4_interpolation_effect():
    """Show how Wasserstein interpolation affects velocity advantage."""
    report = load_report("interpolated_cloud_comprehensive_report.json")
    if not report:
        return

    # Extract raw vs interp results for gaussian_iso/fixed (clearest signal)
    gauss_fixed = [r for r in report["results"]
                   if "gaussian_iso/fixed" in r["preprocessing"]]

    if len(gauss_fixed) < 2:
        print("  [SKIP] Fig 4: not enough interpolation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    labels = [r["preprocessing"].split("/")[0] for r in gauss_fixed]
    t_vals = [r["data_shape"][0] for r in gauss_fixed]
    vel_l2 = [r["vel_l2_rmse"] for r in gauss_fixed]
    pos_l2 = [r["pos_l2_rmse"] for r in gauss_fixed]
    imps_l2 = [r["improvement_l2_pct"] for r in gauss_fixed]
    imps_sk = [r["improvement_sinkhorn_pct"] for r in gauss_fixed]

    # Panel A: RMSE vs preprocessing
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, vel_l2, w, label="Velocity", color="#2166ac", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, pos_l2, w, label="Positions", color="#b2182b", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    for i in range(len(labels)):
        if vel_l2[i] < pos_l2[i]:
            ax.plot(i - w / 2, vel_l2[i] + 0.015, marker="*", color="#1a9850",
                    markersize=8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(r"$L^2(\sigma)$ RMSE")
    ax.set_title("(a) Effect of preprocessing\n(gaussian_iso / fixed)")
    ax.legend(fontsize=7)

    # Panel B: Improvement % (both metrics)
    ax = axes[1]
    ax.bar(x - w / 2, imps_l2, w, label=r"$L^2(\sigma)$", color="#2166ac", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, imps_sk, w, label="Sinkhorn", color="#1a9850", alpha=0.85,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Velocity improvement (%)")
    ax.set_title("(b) Velocity improvement by preprocessing")
    ax.legend(fontsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER_DIR / "fig4_preprocessing_effect.pdf")
    fig.savefig(PAPER_DIR / "fig4_preprocessing_effect.png")
    plt.close(fig)
    print("  Fig 4: preprocessing/interpolation effect")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: Snapshot vs geometric reference
# ══════════════════════════════════════════════════════════════

def fig5_snapshot_vs_geometric():
    """Compare data-driven vs geometric LOT references."""
    report = load_report("interpolated_cloud_snapshot_vs_geometric_report.json")
    if not report:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    results = report["results"]
    # Split into raw and const-vel
    raw = [r for r in results if "(raw)" in r["preprocessing"]]
    cv = [r for r in results if "(const-vel)" in r["preprocessing"]]

    for ax, group, panel, preproc in [(axes[0], raw, "(a)", "raw data"),
                                       (axes[1], cv, "(b)", "constant-velocity")]:
        if not group:
            continue
        labels = [r["preprocessing"].replace(" (raw)", "").replace(" (const-vel)", "")
                  for r in group]
        vel_l2 = [r["vel_l2_rmse"] for r in group]
        pos_l2 = [r["pos_l2_rmse"] for r in group]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, vel_l2, w, label="Velocity", color="#2166ac", alpha=0.85,
               edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, pos_l2, w, label="Positions", color="#b2182b", alpha=0.85,
               edgecolor="black", linewidth=0.4)
        for i in range(len(labels)):
            if vel_l2[i] < pos_l2[i]:
                ax.plot(i - w / 2, vel_l2[i] + 0.015, marker="*", color="#1a9850",
                        markersize=7, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=40, ha="right")
        ax.set_ylabel(r"$L^2(\sigma)$ RMSE")
        ax.set_title(f"{panel} Reference comparison ({preproc})")
        ax.legend(fontsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER_DIR / "fig5_snapshot_vs_geometric.pdf")
    fig.savefig(PAPER_DIR / "fig5_snapshot_vs_geometric.png")
    plt.close(fig)
    print("  Fig 5: snapshot vs geometric")


# ══════════════════════════════════════════════════════════════
# FIGURE 6: Crossover analysis
# ══════════════════════════════════════════════════════════════

def fig6_crossover():
    """Show the crossover point where velocity becomes better than positions over time."""
    import glob

    # Load best per_frame config
    tag = "sr0.9_lk0.9_rd0.01"
    vel_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag}_N_200" /
                              "gaussian_iso" / "frac100" / "*vdom_vel*"))
    pos_dirs = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag}_N_200" /
                              "gaussian_iso" / "frac100" / "*vdom_pos*"))

    # Also load default (fixed) config for comparison
    tag2 = "sr0.9_lk0.7_rd0.01"
    vel_dirs2 = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag2}_N_200" /
                               "gaussian_iso" / "frac100" / "*vdom_vel*"))
    pos_dirs2 = glob.glob(str(REPO_ROOT / "forecast_output" / f"vdom_shared_{tag2}_N_200" /
                               "gaussian_iso" / "frac100" / "*vdom_pos*"))

    if not vel_dirs or not pos_dirs:
        print("  [SKIP] Fig 6: forecast data not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    for ax, vd, pd, panel, desc in [
        (axes[0], vel_dirs, pos_dirs, "(a)", r"$\rho$=0.9, $\alpha$=0.9, $\lambda$=0.01"),
        (axes[1], vel_dirs2, pos_dirs2, "(b)", r"$\rho$=0.9, $\alpha$=0.7, $\lambda$=0.01"),
    ]:
        if not vd or not pd:
            continue
        vp = np.load(Path(vd[0]) / "predicted_maps.npy")
        vt = np.load(Path(vd[0]) / "true_future_maps.npy")
        pp = np.load(Path(pd[0]) / "predicted_maps.npy")
        pt = np.load(Path(pd[0]) / "true_future_maps.npy")
        if vp.shape[0] == vt.shape[0] + 1:
            vp = vp[1:]
        Tf = min(vp.shape[0], pp.shape[0])
        ve = np.sqrt(np.mean(np.sum((vp[:Tf] - vt[:Tf]) ** 2, axis=-1), axis=-1))
        pe = np.sqrt(np.mean(np.sum((pp[:Tf] - pt[:Tf]) ** 2, axis=-1), axis=-1))
        hours = np.arange(Tf) * 0.5

        ax.plot(hours, ve, label="Velocity", color="#2166ac", linewidth=1.5)
        ax.plot(hours, pe, label="Positions", color="#b2182b", linewidth=1.5, alpha=0.8)
        vel_better = ve < pe
        ymax = max(ve.max(), pe.max()) * 1.05
        ax.fill_between(hours, 0, ymax, where=vel_better, alpha=0.06, color="#2166ac",
                         label="Velocity better")
        crossovers = np.where(np.diff(vel_better.astype(int)) != 0)[0]
        for cx in crossovers[:3]:
            ax.axvline(hours[cx], color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel(r"$L^2(\sigma)$ error")
        ax.set_title(f"{panel} {desc}")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("GOES-16 GP 72h — velocity advantage over forecast horizon",
                 fontsize=10, y=1.02)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(PAPER_DIR / "fig6_crossover.pdf")
    fig.savefig(PAPER_DIR / "fig6_crossover.png")
    plt.close(fig)
    print("  Fig 6: crossover analysis")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_cross_dataset()
    fig2_hp_robustness()
    fig3_best_case_error()
    fig4_interpolation_effect()
    fig5_snapshot_vs_geometric()
    fig6_crossover()
    print(f"\nAll figures saved to {PAPER_DIR}/")
    print("  PDF format for LaTeX inclusion")
    print("  PNG format for preview")
