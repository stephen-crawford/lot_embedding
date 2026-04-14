#!/usr/bin/env python3
"""
Evaluate all velocity vs positions forecasts using the *debiased Sinkhorn
divergence* as the error metric (instead of L^2(sigma) RMSE on LOT maps).

The Sinkhorn divergence S_eps(mu, nu) is a proper metric on measures that
approximates W_2^2 and is computed directly on the reconstructed particle
clouds — no dependence on LOT embedding geometry.

  S_eps(alpha, beta) = W_eps(alpha, beta) - 0.5*W_eps(alpha, alpha) - 0.5*W_eps(beta, beta)

We compute the per-timestep Sinkhorn divergence between predicted and true
measures, then report mean and time-averaged sqrt (analogous to RMSE).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "sinkhorn_eval"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SST_180 = REPO_ROOT / "sst_data_180d"
SST_365 = REPO_ROOT / "sst_data_365d"

# ── Sinkhorn divergence ───────────────────────────────────────

import ot as pot


def sinkhorn_divergence_per_step(pred_X, true_X, epsilon=0.01):
    """
    Compute debiased Sinkhorn divergence at each forecast timestep.

    Parameters
    ----------
    pred_X, true_X : (T, N, d) particle clouds
    epsilon : entropic regularization

    Returns
    -------
    (T,) array of S_eps(pred_t, true_t)
    """
    T, N, d = pred_X.shape
    w = np.full(N, 1.0 / N, dtype=np.float64)
    divs = np.empty(T, dtype=np.float64)

    for t in range(T):
        Xa = pred_X[t].astype(np.float64)
        Xb = true_X[t].astype(np.float64)

        M_ab = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)
        M_aa = np.sum((Xa[:, None, :] - Xa[None, :, :]) ** 2, axis=-1)
        M_bb = np.sum((Xb[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)

        W_ab = float(pot.sinkhorn2(w, w, M_ab, reg=epsilon, numItermax=5000))
        W_aa = float(pot.sinkhorn2(w, w, M_aa, reg=epsilon, numItermax=5000))
        W_bb = float(pot.sinkhorn2(w, w, M_bb, reg=epsilon, numItermax=5000))

        divs[t] = max(W_ab - 0.5 * (W_aa + W_bb), 0.0)

    return divs


def sinkhorn_mean(pred_X, true_X, epsilon=0.01):
    """Mean Sinkhorn divergence over forecast horizon."""
    return float(np.mean(sinkhorn_divergence_per_step(pred_X, true_X, epsilon)))


def sinkhorn_rms(pred_X, true_X, epsilon=0.01):
    """Root-mean-square Sinkhorn divergence (analogous to RMSE)."""
    divs = sinkhorn_divergence_per_step(pred_X, true_X, epsilon)
    return float(np.sqrt(np.mean(divs ** 2)))


# Also keep L2 RMSE for comparison
def l2_rmse(pred, true):
    diff = pred - true
    sq = np.sum(diff ** 2, axis=-1)
    return float(np.sqrt(np.mean(sq)))


# ── Pipeline helpers ──────────────────────────────────────────

def _normalize01(t, margin=0.02):
    lo = t.min(axis=(0, 1), keepdims=True)
    hi = t.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (t - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)

def interpolate_ot(traj, factor):
    T, N, d = traj.shape
    T_new = (T - 1) * factor + 1
    out = np.empty((T_new, N, d), dtype=np.float64)
    w = np.full(N, 1.0 / N)
    for i in range(T - 1):
        src, tgt = traj[i].astype(np.float64), traj[i + 1].astype(np.float64)
        M = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
        G = pot.emd(w, w, M)
        T_map = (G @ tgt) / (G.sum(1, keepdims=True) + 1e-18)
        for k in range(factor):
            s = k / factor
            out[i * factor + k] = (1 - s) * src + s * T_map
    out[-1] = traj[-1].astype(np.float64)
    return out.astype(traj.dtype)

def interpolate_linear(traj, factor):
    T, N, d = traj.shape
    T_new = (T - 1) * factor + 1
    out = np.empty((T_new, N, d), dtype=traj.dtype)
    for i in range(T - 1):
        for k in range(factor):
            s = k / factor
            out[i * factor + k] = (1 - s) * traj[i] + s * traj[i + 1]
    out[-1] = traj[-1]
    return out

def resample_adaptive(traj, target_T):
    diffs = traj[1:] - traj[:-1]
    v = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1), axis=-1))
    arc = np.concatenate([[0], np.cumsum(v)])
    t_idx = np.interp(np.linspace(0, arc[-1], target_T), arc, np.arange(traj.shape[0]))
    out = np.empty((target_T, traj.shape[1], traj.shape[2]), dtype=traj.dtype)
    for i, ti in enumerate(t_idx):
        lo = int(np.floor(ti)); hi = min(lo + 1, traj.shape[0] - 1)
        out[i] = (1 - (ti - lo)) * traj[lo] + (ti - lo) * traj[hi]
    return out

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("rwp", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def run_and_eval(traj, tag, n_cycles, kind="gaussian_iso", assign="per_frame",
                 scale=1.0, sr=None, lk=None, rd=None, epsilon=0.01):
    """Run pipeline, then evaluate with both Sinkhorn and L2."""
    tmp = REPO_ROOT / f"_tmp_{tag}"; tmp.mkdir(exist_ok=True)
    np.save(tmp / "particles.npy", traj)
    res_dir = REPO_ROOT / "results" / f"{tag}_N_{traj.shape[1]}"
    res_dir.mkdir(parents=True, exist_ok=True)
    np.save(res_dir / "trajectory.npy", traj)

    mod = _load_pipeline()
    s = mod.run_pipeline(
        system=tag, n_particles=traj.shape[1], n_steps=traj.shape[0],
        n_cycles=n_cycles, goes_dir=tmp, quick=False, ot_method="emd",
        ot_device="cpu", rc_backend="numpy", lot_kind=kind, lot_frac=100,
        assignment=assign, reservoir_scale=scale,
        run_tag_velocity=f"v_{tag}", run_tag_positions=f"p_{tag}",
        spectral_radius_vel=sr, leak_rate_vel=lk, ridge_vel=rd,
    )

    vel_dir, pos_dir = Path(s["velocity_forecast_dir"]), Path(s["positions_forecast_dir"])

    # Load predicted measures (particle clouds, not LOT maps)
    vel_pred_X = np.load(vel_dir / "predicted_measures_X.npy")
    vel_true_X = np.load(vel_dir / "true_measures_X.npy")
    pos_pred_X = np.load(pos_dir / "predicted_measures_X.npy")
    pos_true_X = np.load(pos_dir / "true_measures_X.npy")

    Tf = min(vel_pred_X.shape[0], pos_pred_X.shape[0])
    vel_pred_X, vel_true_X = vel_pred_X[:Tf], vel_true_X[:Tf]
    pos_pred_X, pos_true_X = pos_pred_X[:Tf], pos_true_X[:Tf]

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"    Computing Sinkhorn divergences (eps={epsilon}, {Tf} steps)...")
    vel_sink_steps = sinkhorn_divergence_per_step(vel_pred_X, vel_true_X, epsilon)
    pos_sink_steps = sinkhorn_divergence_per_step(pos_pred_X, pos_true_X, epsilon)

    vel_sink_mean = float(np.mean(vel_sink_steps))
    pos_sink_mean = float(np.mean(pos_sink_steps))
    vel_sink_rms = float(np.sqrt(np.mean(vel_sink_steps ** 2)))
    pos_sink_rms = float(np.sqrt(np.mean(pos_sink_steps ** 2)))

    vel_l2 = l2_rmse(vel_pred_X, vel_true_X)
    pos_l2 = l2_rmse(pos_pred_X, pos_true_X)

    vel_wins_sink = vel_sink_mean < pos_sink_mean
    improve_sink = (pos_sink_mean - vel_sink_mean) / max(pos_sink_mean, 1e-12) * 100

    return {
        "tag": tag, "label": tag,
        "vel_sinkhorn_mean": vel_sink_mean,
        "pos_sinkhorn_mean": pos_sink_mean,
        "vel_sinkhorn_rms": vel_sink_rms,
        "pos_sinkhorn_rms": pos_sink_rms,
        "vel_l2_rmse": vel_l2,
        "pos_l2_rmse": pos_l2,
        "velocity_wins_sinkhorn": vel_wins_sink,
        "improvement_sinkhorn_pct": improve_sink,
        "velocity_wins_l2": vel_l2 < pos_l2,
        "improvement_l2_pct": (pos_l2 - vel_l2) / max(pos_l2, 1e-12) * 100,
        "T": traj.shape[0], "forecast_steps": Tf,
        "epsilon": epsilon,
        "vel_sink_per_step": vel_sink_steps.tolist(),
        "pos_sink_per_step": pos_sink_steps.tolist(),
    }


# ── Plotting ──────────────────────────────────────────────────

def plot_sinkhorn_bar(results, title, save_path):
    labels = [r["label"] for r in results]
    n = len(labels); x = np.arange(n); w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(max(12, n * 1.3), 10))

    # Top left: Sinkhorn mean
    ax = axes[0, 0]
    ax.bar(x - w/2, [r["vel_sinkhorn_mean"] for r in results], w,
           color="#2166ac", alpha=0.85, label="Velocity")
    ax.bar(x + w/2, [r["pos_sinkhorn_mean"] for r in results], w,
           color="#b2182b", alpha=0.85, label="Positions")
    ax.set_ylabel("Mean Sinkhorn div."); ax.set_title("Sinkhorn Divergence (lower = better)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # Top right: Sinkhorn improvement
    ax = axes[0, 1]
    imp = [r["improvement_sinkhorn_pct"] for r in results]
    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in imp]
    ax.bar(x, imp, w * 1.5, color=colors, alpha=0.8)
    ax.axhline(0, color="black", lw=0.5); ax.set_ylabel("Sinkhorn improvement (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(imp):
        ax.text(i, v + (0.5 if v > 0 else -1), f"{v:+.1f}%", ha="center",
                va="bottom" if v > 0 else "top", fontsize=7)

    # Bottom left: L2 RMSE for comparison
    ax = axes[1, 0]
    ax.bar(x - w/2, [r["vel_l2_rmse"] for r in results], w,
           color="#2166ac", alpha=0.85, label="Velocity")
    ax.bar(x + w/2, [r["pos_l2_rmse"] for r in results], w,
           color="#b2182b", alpha=0.85, label="Positions")
    ax.set_ylabel(r"$L^2$ RMSE"); ax.set_title("L² RMSE (for comparison)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # Bottom right: Metric agreement scatter
    ax = axes[1, 1]
    sink_imp = [r["improvement_sinkhorn_pct"] for r in results]
    l2_imp = [r["improvement_l2_pct"] for r in results]
    for i in range(n):
        c = "#4daf4a" if sink_imp[i] > 0 and l2_imp[i] > 0 else (
            "#e41a1c" if sink_imp[i] < 0 and l2_imp[i] < 0 else "#ff7f00")
        ax.scatter(l2_imp[i], sink_imp[i], c=c, s=50, zorder=3, edgecolors="white")
        ax.annotate(labels[i], (l2_imp[i], sink_imp[i]), fontsize=5,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.plot([-200, 200], [-200, 200], "k--", lw=0.5, alpha=0.3)
    ax.set_xlabel("L² improvement (%)"); ax.set_ylabel("Sinkhorn improvement (%)")
    ax.set_title("Metric agreement: Sinkhorn vs L²"); ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {save_path}")


def plot_sinkhorn_curves(results, title, save_path):
    """Per-timestep Sinkhorn divergence curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.cm.tab10
    for i, r in enumerate(results):
        c = cmap(i / max(len(results) - 1, 1))
        vs = np.array(r["vel_sink_per_step"])
        ps = np.array(r["pos_sink_per_step"])
        ax1.plot(vs, color=c, lw=1.0, alpha=0.8, label=f"{r['label']} (vel)")
        ax1.plot(ps, color=c, lw=0.6, alpha=0.4, ls="--")

    ax1.set_xlabel("Forecast step"); ax1.set_ylabel("Sinkhorn divergence")
    ax1.set_title("Per-step Sinkhorn: solid=velocity, dashed=positions")
    ax1.legend(fontsize=6, loc="upper left"); ax1.grid(alpha=0.25)

    # Cumulative Sinkhorn
    for i, r in enumerate(results):
        c = cmap(i / max(len(results) - 1, 1))
        vs = np.cumsum(r["vel_sink_per_step"])
        ps = np.cumsum(r["pos_sink_per_step"])
        ax2.plot(vs, color=c, lw=1.2, label=f"{r['label']} vel")
        ax2.plot(ps, color=c, lw=0.8, ls="--")

    ax2.set_xlabel("Forecast step"); ax2.set_ylabel("Cumulative Sinkhorn div.")
    ax2.set_title("Cumulative Sinkhorn divergence")
    ax2.legend(fontsize=6, loc="upper left"); ax2.grid(alpha=0.25)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    traj_180 = _normalize01(np.load(SST_180 / "particles.npy").astype(np.float32))
    traj_365 = _normalize01(np.load(SST_365 / "particles.npy").astype(np.float32))
    T180, N, d = traj_180.shape
    T365 = traj_365.shape[0]
    print(f"SST 180d: {traj_180.shape}   SST 365d: {traj_365.shape}")

    # Sinkhorn regularization: pick epsilon from median pairwise distance
    sample = traj_180[0].astype(np.float64)
    M = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    eps = float(np.median(M)) * 0.05
    print(f"Sinkhorn epsilon = {eps:.6f} (5% of median sq. distance)")

    all_results = []

    # ═══════════════════════════════════════════════════════════
    # GROUP A: 180-day SST — baseline configs
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP A: 180d SST baseline configurations")
    print("=" * 70)

    configs_180 = [
        ("raw_180d",           traj_180, 3, "gaussian_iso", "per_frame", 1.0, None, None, None),
        ("raw_180d_fixed",     traj_180, 3, "gaussian_iso", "fixed",     1.0, None, None, None),
        ("raw_180d_uniform_pf",traj_180, 3, "uniform_square","per_frame",1.0, None, None, None),
        ("raw_180d_snap_fix",  traj_180, 3, "snapshot_begin","fixed",    1.0, None, None, None),
    ]

    for tag, traj, nc, kind, assign, sc, sr, lk, rd in configs_180:
        tag_full = f"sink_{tag}"
        print(f"\n  --- {tag} ---")
        r = run_and_eval(traj, tag_full, nc, kind, assign, sc, sr, lk, rd, eps)
        r["label"] = tag
        all_results.append(r)
        ws = "WIN" if r["velocity_wins_sinkhorn"] else "loss"
        print(f"  [{ws}] Sinkhorn: vel={r['vel_sinkhorn_mean']:.6f} "
              f"pos={r['pos_sinkhorn_mean']:.6f} {r['improvement_sinkhorn_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # GROUP B: Interpolation + adaptive tau (best configs from earlier)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP B: Interpolation + adaptive tau on 180d SST")
    print("=" * 70)

    print("  OT interpolating 2x...")
    traj_ot2 = interpolate_ot(traj_180, 2)

    interp_configs = [
        ("ot2_adapt_175",  resample_adaptive(traj_ot2, 175),                 3),
        ("ot2_adapt_200",  resample_adaptive(traj_ot2, 200),                 3),
        ("adapt_tau_165",  resample_adaptive(traj_180, 165),                 3),
        ("adapt_tau_330",  resample_adaptive(traj_180, 330),                 6),
        ("linear_4x",     interpolate_linear(traj_180, 4),                  12),
    ]

    for tag, traj_i, nc in interp_configs:
        tag_full = f"sink_{tag}"
        print(f"\n  --- {tag} (T={traj_i.shape[0]}) ---")
        r = run_and_eval(traj_i, tag_full, nc, epsilon=eps)
        r["label"] = tag
        all_results.append(r)
        ws = "WIN" if r["velocity_wins_sinkhorn"] else "loss"
        print(f"  [{ws}] Sinkhorn: vel={r['vel_sinkhorn_mean']:.6f} "
              f"pos={r['pos_sinkhorn_mean']:.6f} {r['improvement_sinkhorn_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # GROUP C: 365-day SST
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP C: 365d SST")
    print("=" * 70)

    configs_365 = [
        ("365d_baseline", traj_365, 4),
    ]

    # Adaptive tau on 365d
    traj_365_adapt = resample_adaptive(traj_365, T365 * 2)
    configs_365.append(("365d_adapt", traj_365_adapt, 8))

    # OT-2x + adaptive on 365d
    print("  OT interpolating 365d 2x...")
    traj_365_ot2 = interpolate_ot(traj_365, 2)
    traj_365_combo = resample_adaptive(traj_365_ot2, 500)
    configs_365.append(("365d_ot2_adapt500", traj_365_combo, 5))

    for tag, traj_i, nc in configs_365:
        tag_full = f"sink_{tag}"
        print(f"\n  --- {tag} (T={traj_i.shape[0]}) ---")
        r = run_and_eval(traj_i, tag_full, nc, epsilon=eps)
        r["label"] = tag
        all_results.append(r)
        ws = "WIN" if r["velocity_wins_sinkhorn"] else "loss"
        print(f"  [{ws}] Sinkhorn: vel={r['vel_sinkhorn_mean']:.6f} "
              f"pos={r['pos_sinkhorn_mean']:.6f} {r['improvement_sinkhorn_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # GROUP D: HP tuning on best pipeline, evaluated by Sinkhorn
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP D: Hyperparameter tuning (OT2+adapt T=200, Sinkhorn eval)")
    print("=" * 70)

    traj_best = resample_adaptive(traj_ot2, 200)
    for sr, lk, rd, lbl in [
        (0.75, 0.9, 0.005, "hp_tuned"),
        (0.8,  0.8, 0.01,  "hp_default"),
        (0.85, 0.85, 0.02, "hp_mid"),
    ]:
        tag_full = f"sink_{lbl}"
        print(f"\n  --- {lbl} ---")
        r = run_and_eval(traj_best, tag_full, 3, sr=sr, lk=lk, rd=rd, epsilon=eps)
        r["label"] = lbl
        all_results.append(r)
        ws = "WIN" if r["velocity_wins_sinkhorn"] else "loss"
        print(f"  [{ws}] Sinkhorn: vel={r['vel_sinkhorn_mean']:.6f} "
              f"pos={r['pos_sinkhorn_mean']:.6f} {r['improvement_sinkhorn_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL RESULTS — Sinkhorn divergence metric")
    print("=" * 70)

    wins_sink = [r for r in all_results if r["velocity_wins_sinkhorn"]]
    wins_l2 = [r for r in all_results if r["velocity_wins_l2"]]

    for r in sorted(all_results, key=lambda x: -x["improvement_sinkhorn_pct"]):
        s_sink = "SINK-WIN" if r["velocity_wins_sinkhorn"] else "sink-loss"
        s_l2 = "L2-WIN" if r["velocity_wins_l2"] else "l2-loss"
        print(f"  [{s_sink:9s}|{s_l2:8s}] {r['label']:25s}  "
              f"sink: vel={r['vel_sinkhorn_mean']:.6f} pos={r['pos_sinkhorn_mean']:.6f} "
              f"{r['improvement_sinkhorn_pct']:+.1f}%")

    print(f"\nSinkhorn velocity wins: {len(wins_sink)}/{len(all_results)}")
    print(f"L² velocity wins:      {len(wins_l2)}/{len(all_results)}")

    agree = sum(1 for r in all_results
                if r["velocity_wins_sinkhorn"] == r["velocity_wins_l2"])
    print(f"Metric agreement:      {agree}/{len(all_results)}")

    # Plots
    plot_sinkhorn_bar(all_results,
                      "Velocity vs Positions — Sinkhorn Divergence on Real SST",
                      PLOTS_DIR / "sinkhorn_summary.png")

    plot_sinkhorn_curves(all_results,
                         "Per-step Sinkhorn divergence curves",
                         PLOTS_DIR / "sinkhorn_curves.png")

    # Save report
    report = {
        "metric": "debiased_sinkhorn_divergence",
        "epsilon": eps,
        "n_sinkhorn_wins": len(wins_sink),
        "n_l2_wins": len(wins_l2),
        "n_total": len(all_results),
        "metric_agreement": agree,
        "results": [{k: v for k, v in r.items()
                     if k not in ("vel_sink_per_step", "pos_sink_per_step")}
                    for r in all_results],
    }
    rpath = REPO_ROOT / "sinkhorn_evaluation_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {rpath}")
    print(f"Plots:  {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
