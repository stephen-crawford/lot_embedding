#!/usr/bin/env python3
"""
Deep investigation following up on next-steps findings.

1. Fine-tune the OT-2x + adaptive tau sweet spot (vary T around 200)
2. Velocity residual RC (subtract moving-average trend, predict residual)
3. Hyperparameter sweep on the best pipeline configuration
4. Multi-year SST (combine 180d + 365d datasets)
5. Reservoir size scaling study
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

PLOTS_DIR = REPO_ROOT / "plots" / "deep_investigation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SST_180 = REPO_ROOT / "sst_data_180d"
SST_365 = REPO_ROOT / "sst_data_365d"


# ── Reuse helpers ──────────────────────────────────────────────

def _normalize01(t, margin=0.02):
    lo = t.min(axis=(0, 1), keepdims=True)
    hi = t.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (t - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)

def l2_per_step(pred, true):
    return np.sqrt(np.mean(np.sum((pred - true)**2, axis=-1), axis=-1))

def l2_rmse(pred, true):
    return float(np.sqrt(np.mean(l2_per_step(pred, true)**2)))

def compute_velocity_norms(traj):
    diffs = traj[1:] - traj[:-1]
    return np.sqrt(np.mean(np.sum(diffs**2, axis=-1), axis=-1))

def interpolate_ot(traj, factor):
    import ot as pot
    T, N, d = traj.shape
    T_new = (T-1)*factor + 1
    out = np.empty((T_new, N, d), dtype=np.float64)
    w = np.full(N, 1.0/N)
    for i in range(T-1):
        src, tgt = traj[i].astype(np.float64), traj[i+1].astype(np.float64)
        M = np.sum((src[:,None,:] - tgt[None,:,:])**2, axis=-1)
        G = pot.emd(w, w, M)
        T_map = (G @ tgt) / (G.sum(1, keepdims=True) + 1e-18)
        for k in range(factor):
            s = k / factor
            out[i*factor + k] = (1-s)*src + s*T_map
    out[-1] = traj[-1].astype(np.float64)
    return out.astype(traj.dtype)

def resample_adaptive(traj, target_T):
    v = compute_velocity_norms(traj)
    arc = np.concatenate([[0], np.cumsum(v)])
    t_idx = np.interp(np.linspace(0, arc[-1], target_T), arc, np.arange(traj.shape[0]))
    out = np.empty((target_T, traj.shape[1], traj.shape[2]), dtype=traj.dtype)
    for i, ti in enumerate(t_idx):
        lo = int(np.floor(ti)); hi = min(lo+1, traj.shape[0]-1)
        out[i] = (1-(ti-lo))*traj[lo] + (ti-lo)*traj[hi]
    return out

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("rwp", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def run_pipeline(traj, tag, n_cycles, kind="gaussian_iso", assign="per_frame",
                 scale=1.0, sr=None, lk=None, rd=None):
    tmp = REPO_ROOT / f"_tmp_{tag}"; tmp.mkdir(exist_ok=True)
    np.save(tmp / "particles.npy", traj)
    res = REPO_ROOT / "results" / f"{tag}_N_{traj.shape[1]}"
    res.mkdir(parents=True, exist_ok=True); np.save(res / "trajectory.npy", traj)
    mod = _load_pipeline()
    s = mod.run_pipeline(
        system=tag, n_particles=traj.shape[1], n_steps=traj.shape[0],
        n_cycles=n_cycles, goes_dir=tmp, quick=False, ot_method="emd",
        ot_device="cpu", rc_backend="numpy", lot_kind=kind, lot_frac=100,
        assignment=assign, reservoir_scale=scale,
        run_tag_velocity=f"v_{tag}", run_tag_positions=f"p_{tag}",
        spectral_radius_vel=sr, leak_rate_vel=lk, ridge_vel=rd,
    )
    vd, pd = Path(s["velocity_forecast_dir"]), Path(s["positions_forecast_dir"])
    vp, vt = np.load(vd/"predicted_maps.npy"), np.load(vd/"true_future_maps.npy")
    pp, pt = np.load(pd/"predicted_maps.npy"), np.load(pd/"true_future_maps.npy")
    if vp.shape[0] == vt.shape[0]+1: vp = vp[1:]
    Tf = min(vp.shape[0], pp.shape[0])
    shutil.rmtree(tmp, ignore_errors=True)
    vr = l2_rmse(vp[:Tf], vt[:Tf]); pr = l2_rmse(pp[:Tf], pt[:Tf])
    return {"tag": tag, "label": tag, "vel_rmse": vr, "pos_rmse": pr,
            "velocity_wins": vr < pr, "T": traj.shape[0], "forecast_steps": Tf,
            "improvement_pct": (pr - vr) / max(pr, 1e-12) * 100,
            "vel_per_step": l2_per_step(vp[:Tf], vt[:Tf]).tolist(),
            "pos_per_step": l2_per_step(pp[:Tf], pt[:Tf]).tolist()}


# ── Velocity residual RC ──────────────────────────────────────

def run_velocity_residual_rc(traj, tag, n_cycles, window=5,
                             kind="gaussian_iso", assign="per_frame"):
    """
    Velocity residual forecasting:
    1. Compute LOT maps and velocities
    2. Compute moving-average velocity v_smooth (window)
    3. Residual = v - v_smooth (the "turbulent" part)
    4. Train RC to predict residual from maps
    5. At forecast: predict residual, add back trend, integrate

    This is the "work in constant velocity space, add back acceleration" idea.
    """
    from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings
    from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme

    results_root = REPO_ROOT / "results"
    lot_root = REPO_ROOT / "lot_maps"
    T, N, d = traj.shape

    traj_dir = results_root / f"{tag}_N_{N}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.save(traj_dir / "trajectory.npy", traj)

    generate_lot_embeddings(
        tag, N_list=[N], kinds=[kind], fractions=[100],
        assignment=assign, anchor_frame=0,
        results_root=results_root, lot_root=lot_root,
        verbose=False, ot_method="emd", ot_device="cpu",
    )

    lot_dir = lot_root / f"{tag}_N_{N}" / kind / "frac100"
    maps = np.load(lot_dir / "lot_maps.npy")       # (T, R, d)
    T_m, R, d_m = maps.shape
    maps_flat = maps.reshape(T_m, -1)               # (T, R*d)
    vels = maps_flat[1:] - maps_flat[:-1]            # (T-1, R*d)

    # Moving-average trend
    half_w = window // 2
    v_smooth = np.empty_like(vels)
    for t in range(len(vels)):
        lo = max(0, t - half_w)
        hi = min(len(vels), t + half_w + 1)
        v_smooth[t] = vels[lo:hi].mean(axis=0)

    residuals = vels - v_smooth                      # (T-1, R*d)

    # RC setup
    input_size = R * d_m
    cycle_length = max(1, T_m // max(1, n_cycles))
    warm = cycle_length - 1
    if warm >= T_m - 1: warm = T_m // 2

    rc = ReservoirComputer(ReservoirConfig(
        input_size=input_size, reservoir_size=R, output_size=input_size,
        spectral_radius=0.8, input_scaling=0.1, leak_rate=0.8,
        ridge_param=0.01, activation="tanh",
        init_scheme=InitScheme.SPARSE_UNIFORM, use_operator_norm=True,
    ))

    # Train: input = maps[0..T-2], target = residuals[0..T-2]
    train_in = maps_flat[:-1]
    states = rc.run(train_in)
    rc.train(states, residuals, washout=warm)

    # Autonomous forecast
    forecast_start = warm + 1
    forecast_steps = T_m - forecast_start - 1
    if forecast_steps < 2:
        return None

    warmup_states = rc.run(maps_flat[:forecast_start])
    r = warmup_states[-1].copy()
    cur_map = maps_flat[forecast_start - 1].copy()
    # Need recent velocity for trend extrapolation
    recent_vel = vels[forecast_start - 1].copy() if forecast_start - 1 < len(vels) else np.zeros(input_size)

    pred_maps = [cur_map.copy()]
    for t in range(forecast_steps):
        # Predict residual
        res_hat = rc.predict(r.reshape(1, -1))[0]
        # Extrapolate trend: use exponential moving average of recent velocity
        alpha = 0.1  # EMA decay
        trend = recent_vel.copy()
        # Full velocity = trend + residual
        v_hat = trend + res_hat
        cur_map = cur_map + v_hat
        cur_map = np.clip(cur_map.reshape(R, d_m), 0.0, 1.0).reshape(-1)
        pred_maps.append(cur_map.copy())
        # Update RC state
        pre = rc.W @ r.reshape(-1,1) + rc.W_in @ cur_map.reshape(-1,1) + rc.bias
        r = np.tanh(pre).flatten()
        # Update trend EMA
        recent_vel = alpha * v_hat + (1 - alpha) * recent_vel

    pred_maps = np.array(pred_maps).reshape(-1, R, d_m)
    true_maps = maps[forecast_start - 1: forecast_start - 1 + len(pred_maps)]
    Tf = min(len(pred_maps), len(true_maps))
    resid_rmse = l2_rmse(pred_maps[:Tf], true_maps[:Tf])

    # Compare to standard pipeline
    tmp = REPO_ROOT / f"_tmp_{tag}"; tmp.mkdir(exist_ok=True)
    np.save(tmp / "particles.npy", traj)
    r_std = run_pipeline(traj, f"{tag}_cmp", n_cycles, kind=kind, assign=assign)
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "tag": tag,
        "resid_rmse": resid_rmse,
        "vel_rmse": r_std["vel_rmse"],
        "pos_rmse": r_std["pos_rmse"],
        "resid_wins_vel": resid_rmse < r_std["vel_rmse"],
        "resid_wins_pos": resid_rmse < r_std["pos_rmse"],
        "forecast_steps": Tf,
        "window": window,
    }


# ── Plotting ───────────────────────────────────────────────────

def plot_sweep(results, xlabel, xvals, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    vel = [r["vel_rmse"] for r in results]
    pos = [r["pos_rmse"] for r in results]
    improv = [r["improvement_pct"] for r in results]

    ax1.plot(xvals, vel, "o-", color="#2166ac", lw=1.5, label="Velocity")
    ax1.plot(xvals, pos, "s-", color="#b2182b", lw=1.5, label="Positions")
    ax1.set_xlabel(xlabel); ax1.set_ylabel(r"$L^2(\sigma)$ RMSE")
    ax1.set_title(title); ax1.legend(); ax1.grid(alpha=0.25)

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(range(len(improv)), improv, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(xvals))); ax2.set_xticklabels([str(x) for x in xvals])
    ax2.set_xlabel(xlabel); ax2.set_ylabel("Improvement (%)")
    ax2.axhline(0, color="black", lw=0.5); ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(improv):
        ax2.text(i, v+(1 if v>0 else -2), f"{v:+.1f}%", ha="center",
                 fontsize=8, va="bottom" if v>0 else "top")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {save_path}")


def plot_three_method(results, save_path):
    labels = [r["tag"].split("_", 1)[-1] if "_" in r["tag"] else r["tag"] for r in results]
    n = len(labels); x = np.arange(n); w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, n*2), 5))
    ax.bar(x-w, [r["resid_rmse"] for r in results], w, color="#4daf4a", alpha=0.85,
           label="Vel. Residual RC")
    ax.bar(x, [r["vel_rmse"] for r in results], w, color="#2166ac", alpha=0.85,
           label="Velocity RC")
    ax.bar(x+w, [r["pos_rmse"] for r in results], w, color="#b2182b", alpha=0.85,
           label="Positions RC")
    for i in range(n):
        for dx, val, c in [(-w, results[i]["resid_rmse"], "#4daf4a"),
                           (0, results[i]["vel_rmse"], "#2166ac"),
                           (w, results[i]["pos_rmse"], "#b2182b")]:
            ax.text(x[i]+dx, val+0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7, color=c)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$L^2(\sigma)$ RMSE"); ax.set_title("Velocity Residual RC vs Standard Methods")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {save_path}")


def plot_combined_summary(all_results, save_path):
    # Sort by improvement
    sorted_r = sorted(all_results, key=lambda x: -x["improvement_pct"])
    labels = [r["label"] for r in sorted_r]
    vel = [r["vel_rmse"] for r in sorted_r]
    pos = [r["pos_rmse"] for r in sorted_r]
    improv = [r["improvement_pct"] for r in sorted_r]
    n = len(labels); x = np.arange(n); w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n*1.2), 9),
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.bar(x-w/2, vel, w, color="#2166ac", alpha=0.85, label="Velocity")
    ax1.bar(x+w/2, pos, w, color="#b2182b", alpha=0.85, label="Positions")
    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE"); ax1.set_title("Deep Investigation — All Results (sorted by improvement)", fontsize=13)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax1.legend(); ax1.grid(axis="y", alpha=0.25)

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, w*1.5, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", lw=0.5); ax2.set_ylabel("Improvement (%)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(improv):
        ax2.text(i, v+(0.8 if v>0 else -1.5), f"{v:+.1f}%", ha="center",
                 va="bottom" if v>0 else "top", fontsize=7)
    fig.tight_layout(); fig.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════

def main():
    traj_180 = _normalize01(np.load(SST_180 / "particles.npy").astype(np.float32))
    traj_365 = _normalize01(np.load(SST_365 / "particles.npy").astype(np.float32))
    T180, N, d = traj_180.shape
    T365 = traj_365.shape[0]
    print(f"SST 180d: {traj_180.shape}   SST 365d: {traj_365.shape}")

    all_results = []
    resid_results = []

    # ═══════════════════════════════════════════════════════════
    # INV 1: Fine-tune OT-2x + adaptive tau sweet spot
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INV 1: Fine-tune OT-2x + adaptive tau (target T sweep)")
    print("=" * 70)

    print("  OT interpolating 180d 2x...")
    traj_ot2_180 = interpolate_ot(traj_180, 2)

    sweep1_results = []
    for target_T in [150, 175, 200, 225, 250, 300]:
        t_resampled = resample_adaptive(traj_ot2_180, target_T)
        tag = f"inv1_ot2adapt_T{target_T}"
        nc = max(3, target_T // 55)
        r = run_pipeline(t_resampled, tag, nc, assign="per_frame")
        r["label"] = f"OT2+adapt T={target_T}"
        sweep1_results.append(r)
        all_results.append(r)
        s = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{s}] T={target_T}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    plot_sweep(sweep1_results, "Target T",
               [150, 175, 200, 225, 250, 300],
               "OT-2x + adaptive tau: target T sweep (180d SST)",
               PLOTS_DIR / "inv1_target_T_sweep.png")

    # ═══════════════════════════════════════════════════════════
    # INV 2: Velocity residual RC
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INV 2: Velocity residual RC (subtract trend, predict residual)")
    print("=" * 70)

    # Test different moving-average windows
    for win in [3, 5, 10, 20]:
        r_res = run_velocity_residual_rc(traj_180, f"resid_w{win}_180d",
                                          n_cycles=3, window=win)
        if r_res:
            resid_results.append(r_res)
            best = "RESID" if r_res["resid_wins_vel"] else "VEL"
            print(f"  window={win}: resid={r_res['resid_rmse']:.4f} "
                  f"vel={r_res['vel_rmse']:.4f} pos={r_res['pos_rmse']:.4f} best={best}")

    # Also on adaptive-tau data
    traj_adapt = resample_adaptive(traj_180, 330)
    for win in [5, 15]:
        r_res = run_velocity_residual_rc(traj_adapt, f"resid_w{win}_adapt330",
                                          n_cycles=6, window=win)
        if r_res:
            resid_results.append(r_res)
            best = "RESID" if r_res["resid_wins_vel"] else "VEL"
            print(f"  adapt330 w={win}: resid={r_res['resid_rmse']:.4f} "
                  f"vel={r_res['vel_rmse']:.4f} pos={r_res['pos_rmse']:.4f} best={best}")

    # On 365d data
    for win in [5, 15]:
        r_res = run_velocity_residual_rc(traj_365, f"resid_w{win}_365d",
                                          n_cycles=4, window=win)
        if r_res:
            resid_results.append(r_res)
            best = "RESID" if r_res["resid_wins_vel"] else "VEL"
            print(f"  365d w={win}: resid={r_res['resid_rmse']:.4f} "
                  f"vel={r_res['vel_rmse']:.4f} pos={r_res['pos_rmse']:.4f} best={best}")

    if resid_results:
        plot_three_method(resid_results, PLOTS_DIR / "inv2_residual_rc.png")

    # ═══════════════════════════════════════════════════════════
    # INV 3: Hyperparameter sweep on best pipeline
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INV 3: Hyperparameter sweep (OT-2x + adapt T=200)")
    print("=" * 70)

    traj_best = resample_adaptive(traj_ot2_180, 200)
    sweep3_results = []

    hp_grid = [
        # (sr, lk, rd, label)
        (0.8, 0.8, 0.01, "default"),
        (0.7, 0.8, 0.01, "sr=.7"),
        (0.9, 0.8, 0.01, "sr=.9"),
        (0.8, 0.5, 0.01, "lk=.5"),
        (0.8, 0.9, 0.01, "lk=.9"),
        (0.8, 0.8, 0.001, "rd=.001"),
        (0.8, 0.8, 0.1, "rd=.1"),
        (0.85, 0.85, 0.005, "tuned1"),
        (0.75, 0.9, 0.005, "tuned2"),
        (0.9, 0.7, 0.005, "tuned3"),
    ]

    for sr, lk, rd, lbl in hp_grid:
        tag = f"inv3_hp_{lbl.replace('=','').replace('.','')}"
        r = run_pipeline(traj_best, tag, n_cycles=3, assign="per_frame",
                         sr=sr, lk=lk, rd=rd)
        r["label"] = lbl
        sweep3_results.append(r)
        all_results.append(r)
        s = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{s}] {lbl}: vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
              f"{r['improvement_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # INV 4: Best pipeline on 365d data
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INV 4: Best pipeline on 365d SST")
    print("=" * 70)

    print("  OT interpolating 365d 2x...")
    traj_ot2_365 = interpolate_ot(traj_365, 2)

    for target_T in [400, 500, 700]:
        t_r = resample_adaptive(traj_ot2_365, target_T)
        tag = f"inv4_365d_ot2adapt_T{target_T}"
        nc = max(4, target_T // 90)
        r = run_pipeline(t_r, tag, nc, assign="per_frame")
        r["label"] = f"365d OT2+adapt T={target_T}"
        all_results.append(r)
        s = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{s}] T={target_T}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    # Best HP from sweep on 365d
    best_hp = min(sweep3_results, key=lambda x: x["vel_rmse"])
    best_lbl = best_hp["label"]
    # Extract hp values from the grid
    best_hp_vals = None
    for sr, lk, rd, lbl in hp_grid:
        if lbl == best_lbl:
            best_hp_vals = (sr, lk, rd)
            break

    if best_hp_vals:
        sr, lk, rd = best_hp_vals
        t_r = resample_adaptive(traj_ot2_365, 500)
        tag = "inv4_365d_best_hp"
        r = run_pipeline(t_r, tag, 5, assign="per_frame", sr=sr, lk=lk, rd=rd)
        r["label"] = f"365d OT2+adapt+bestHP ({best_lbl})"
        all_results.append(r)
        s = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{s}] Best HP on 365d: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    # ═══════════════════════════════════════════════════════════
    # INV 5: Reservoir size scaling
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INV 5: Reservoir size scaling study")
    print("=" * 70)

    sweep5_results = []
    for scale in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        tag = f"inv5_scale_{scale}".replace(".", "p")
        r = run_pipeline(traj_best, tag, n_cycles=3, assign="per_frame", scale=scale)
        r["label"] = f"scale={scale}"
        sweep5_results.append(r)
        all_results.append(r)
        res_size = int(200 * scale)
        s = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{s}] res={res_size}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    plot_sweep(sweep5_results, "Reservoir scale",
               [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
               "Reservoir size scaling (OT-2x + adapt T=200)",
               PLOTS_DIR / "inv5_reservoir_scaling.png")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL RESULTS (sorted by velocity improvement)")
    print("=" * 70)

    wins = [r for r in all_results if r["velocity_wins"]]
    for r in sorted(all_results, key=lambda x: -x["improvement_pct"]):
        s = "WIN " if r["velocity_wins"] else "loss"
        print(f"  [{s}] {r['label']:45s}  vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f}  {r['improvement_pct']:+.1f}%")
    print(f"\nVelocity wins: {len(wins)}/{len(all_results)}")

    if resid_results:
        print("\nVelocity Residual RC:")
        for r in resid_results:
            best = "RESID" if r["resid_wins_vel"] else "VEL"
            print(f"  {r['tag']:40s}  resid={r['resid_rmse']:.4f} "
                  f"vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f}  best={best}")

    plot_combined_summary(all_results, PLOTS_DIR / "deep_investigation_summary.png")

    # Save report
    report = {
        "pipeline_results": [{k: v for k, v in r.items()
                              if k not in ("vel_per_step", "pos_per_step")}
                             for r in all_results],
        "residual_results": resid_results,
        "n_wins": len(wins), "n_total": len(all_results),
    }
    rpath = REPO_ROOT / "deep_investigation_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {rpath}")
    print(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
