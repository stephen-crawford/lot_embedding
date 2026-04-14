#!/usr/bin/env python3
"""
Next-steps experiments on real SST data, building on findings from
interpolation_experiments.py.

Implements:
  1. Adaptive tau + OT interpolation combined
  2. Acceleration-based RC training (predict Δv, double-integrate)
  3. Optimal interpolation factor search (3x, 5x around the 4x sweet spot)
  4. Longer real data (download 365-day SST for full annual cycle)

All experiments use the paper's L^2(sigma) RMSE metric.
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
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "next_steps"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SST_DIR = REPO_ROOT / "sst_data_180d"
SST_365_DIR = REPO_ROOT / "sst_data_365d"


# ── Helpers from previous experiments ──────────────────────────

def _normalize01(t, margin=0.02):
    lo = t.min(axis=(0, 1), keepdims=True)
    hi = t.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (t - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def l2_sigma_per_step(pred, true):
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred, true):
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


def compute_velocity_norms(traj):
    diffs = traj[1:] - traj[:-1]
    sq_norms = np.sum(diffs ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def interpolate_measures_linear(traj, factor):
    T, N, d = traj.shape
    T_new = (T - 1) * factor + 1
    out = np.empty((T_new, N, d), dtype=traj.dtype)
    for i in range(T - 1):
        for k in range(factor):
            s = k / factor
            out[i * factor + k] = (1 - s) * traj[i] + s * traj[i + 1]
    out[-1] = traj[-1]
    return out


def interpolate_measures_ot(traj, factor):
    import ot as pot
    T, N, d = traj.shape
    T_new = (T - 1) * factor + 1
    out = np.empty((T_new, N, d), dtype=np.float64)
    w = np.full(N, 1.0 / N)
    for i in range(T - 1):
        src = traj[i].astype(np.float64)
        tgt = traj[i + 1].astype(np.float64)
        M = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
        G = pot.emd(w, w, M)
        row_sums = G.sum(axis=1, keepdims=True) + 1e-18
        T_map = (G @ tgt) / row_sums
        for k in range(factor):
            s = k / factor
            out[i * factor + k] = (1 - s) * src + s * T_map
    out[-1] = traj[-1].astype(np.float64)
    return out.astype(traj.dtype)


def resample_adaptive_tau(traj, target_T):
    v_norms = compute_velocity_norms(traj)
    T_orig = traj.shape[0]
    arc = np.concatenate([[0], np.cumsum(v_norms)])
    total_arc = arc[-1]
    target_arcs = np.linspace(0, total_arc, target_T)
    t_indices = np.interp(target_arcs, arc, np.arange(T_orig))
    out = np.empty((target_T, traj.shape[1], traj.shape[2]), dtype=traj.dtype)
    for i, ti in enumerate(t_indices):
        t_lo = int(np.floor(ti))
        t_hi = min(t_lo + 1, T_orig - 1)
        frac = ti - t_lo
        out[i] = (1 - frac) * traj[t_lo] + frac * traj[t_hi]
    return out, t_indices


# ── Pipeline runner ────────────────────────────────────────────

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_on_trajectory(traj, tag, n_cycles, lot_kind="gaussian_iso",
                      assignment="per_frame", scale=1.0, sr=None, lk=None, rd=None):
    tmp_dir = REPO_ROOT / f"_tmp_{tag}"
    tmp_dir.mkdir(exist_ok=True)
    np.save(tmp_dir / "particles.npy", traj)
    results_root = REPO_ROOT / "results"
    traj_dir = results_root / f"{tag}_N_{traj.shape[1]}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.save(traj_dir / "trajectory.npy", traj)

    mod = _load_pipeline()
    summary = mod.run_pipeline(
        system=tag,
        n_particles=traj.shape[1],
        n_steps=traj.shape[0],
        n_cycles=n_cycles,
        goes_dir=tmp_dir,
        quick=False,
        ot_method="emd",
        ot_device="cpu",
        rc_backend="numpy",
        lot_kind=lot_kind,
        lot_frac=100,
        assignment=assignment,
        reservoir_scale=scale,
        run_tag_velocity=f"v_{tag}",
        run_tag_positions=f"p_{tag}",
        spectral_radius_vel=sr,
        leak_rate_vel=lk,
        ridge_vel=rd,
    )

    vel_dir = Path(summary["velocity_forecast_dir"])
    pos_dir = Path(summary["positions_forecast_dir"])
    vp = np.load(vel_dir / "predicted_maps.npy")
    vt = np.load(vel_dir / "true_future_maps.npy")
    pp = np.load(pos_dir / "predicted_maps.npy")
    pt = np.load(pos_dir / "true_future_maps.npy")
    if vp.shape[0] == vt.shape[0] + 1:
        vp = vp[1:]
    Tf = min(vp.shape[0], pp.shape[0])

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "tag": tag, "label": tag,
        "vel_rmse": l2_sigma_rmse(vp[:Tf], vt[:Tf]),
        "pos_rmse": l2_sigma_rmse(pp[:Tf], pt[:Tf]),
        "velocity_wins": l2_sigma_rmse(vp[:Tf], vt[:Tf]) < l2_sigma_rmse(pp[:Tf], pt[:Tf]),
        "improvement_pct": (l2_sigma_rmse(pp[:Tf], pt[:Tf]) - l2_sigma_rmse(vp[:Tf], vt[:Tf]))
                           / max(l2_sigma_rmse(pp[:Tf], pt[:Tf]), 1e-12) * 100,
        "T": traj.shape[0], "forecast_steps": Tf,
        "vel_dir": str(vel_dir), "pos_dir": str(pos_dir),
        "vel_per_step": l2_sigma_per_step(vp[:Tf], vt[:Tf]).tolist(),
        "pos_per_step": l2_sigma_per_step(pp[:Tf], pt[:Tf]).tolist(),
    }


# ── Acceleration RC: predict Δv instead of v ──────────────────

def run_acceleration_rc(traj, tag, n_cycles, lot_kind="gaussian_iso",
                        assignment="per_frame"):
    """
    Custom acceleration-based forecasting:
    1. Compute LOT maps
    2. Compute velocities v_t = maps[t+1] - maps[t]
    3. Compute accelerations a_t = v_{t+1} - v_t
    4. Train RC: maps[t] -> a_t
    5. Predict: a_hat, then v_{t+1} = v_t + a_hat, then maps[t+1] = maps[t] + v_{t+1}

    Compared against standard velocity and positions forecasts.
    """
    from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings
    from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme

    # Save trajectory
    results_root = REPO_ROOT / "results"
    traj_dir = results_root / f"{tag}_N_{traj.shape[1]}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.save(traj_dir / "trajectory.npy", traj)

    tmp_dir = REPO_ROOT / f"_tmp_{tag}"
    tmp_dir.mkdir(exist_ok=True)
    np.save(tmp_dir / "particles.npy", traj)

    T, N, d = traj.shape
    lot_root = REPO_ROOT / "lot_maps"
    forecast_root = REPO_ROOT / "forecast_output"

    # Generate LOT embeddings
    generate_lot_embeddings(
        tag, N_list=[N], kinds=[lot_kind], fractions=[100],
        assignment=assignment, anchor_frame=0,
        results_root=results_root, lot_root=lot_root,
        verbose=False, ot_method="emd", ot_device="cpu",
    )

    lot_dir = lot_root / f"{tag}_N_{N}" / lot_kind / "frac100"
    maps = np.load(lot_dir / "lot_maps.npy")  # (T, R, d)
    T_map, R, d_map = maps.shape

    # Flatten maps for RC
    maps_flat = maps.reshape(T_map, -1)  # (T, R*d)

    # Velocities and accelerations
    velocities = maps_flat[1:] - maps_flat[:-1]   # (T-1, R*d)
    accelerations = velocities[1:] - velocities[:-1]  # (T-2, R*d)

    # Cycle and warmup
    cycle_length = max(1, T_map // max(1, n_cycles))
    warm_steps = cycle_length - 1

    if warm_steps + 2 >= T_map:
        warm_steps = T_map // 2

    # Training data: input = maps[t], target = accelerations[t]
    # We need maps[0..T-3] as inputs, accelerations[0..T-3] as targets
    train_input = maps_flat[:T_map - 2]  # (T-2, R*d)
    train_target_accel = accelerations     # (T-2, R*d)

    input_size = R * d_map
    reservoir_size = R  # Match LOT dimensionality

    # ── Acceleration RC ──
    rc_accel = ReservoirComputer(ReservoirConfig(
        input_size=input_size,
        reservoir_size=reservoir_size,
        output_size=input_size,
        spectral_radius=0.8,
        input_scaling=0.1,
        leak_rate=0.8,
        ridge_param=0.01,
        activation="tanh",
        init_scheme=InitScheme.SPARSE_UNIFORM,
        use_operator_norm=True,
    ))

    states_accel = rc_accel.run(train_input)
    rc_accel.train(states_accel, train_target_accel, washout=warm_steps)

    # Autonomous rollout for acceleration RC
    forecast_start = warm_steps + 2  # Need +2 because accel needs 2 prior maps
    forecast_steps = T_map - forecast_start - 1

    if forecast_steps < 2:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Warm up with teacher forcing
    warmup_input = maps_flat[:forecast_start]
    states_warm = rc_accel.run(warmup_input)
    last_state = states_warm[-1]

    # We need the last velocity to integrate
    last_vel = maps_flat[forecast_start - 1] - maps_flat[forecast_start - 2]
    current_map = maps_flat[forecast_start - 1].copy()
    current_vel = last_vel.copy()

    pred_maps_accel = [current_map.copy()]
    r = last_state.copy()

    for t in range(forecast_steps):
        # Predict acceleration
        a_hat = rc_accel.predict(r.reshape(1, -1))[0]
        # Integrate: v_{t+1} = v_t + a_hat
        current_vel = current_vel + a_hat
        # Integrate: map_{t+1} = map_t + v_{t+1}
        current_map = current_map + current_vel
        current_map = np.clip(current_map.reshape(R, d_map), 0.0, 1.0).reshape(-1)
        pred_maps_accel.append(current_map.copy())
        # Update reservoir state
        pre = rc_accel.W @ r.reshape(-1, 1) + rc_accel.W_in @ current_map.reshape(-1, 1) + rc_accel.bias
        r = np.tanh(pre).flatten()

    pred_maps_accel = np.array(pred_maps_accel).reshape(-1, R, d_map)
    true_maps = maps[forecast_start - 1: forecast_start - 1 + len(pred_maps_accel)]

    Tf = min(len(pred_maps_accel), len(true_maps))
    accel_rmse = l2_sigma_rmse(pred_maps_accel[:Tf], true_maps[:Tf])

    # ── Also run standard velocity and positions for comparison ──
    r_std = run_on_trajectory(traj, f"{tag}_std", n_cycles,
                              lot_kind=lot_kind, assignment=assignment)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "tag": tag,
        "accel_rmse": accel_rmse,
        "vel_rmse": r_std["vel_rmse"],
        "pos_rmse": r_std["pos_rmse"],
        "forecast_steps": Tf,
        "accel_wins_vs_vel": accel_rmse < r_std["vel_rmse"],
        "accel_wins_vs_pos": accel_rmse < r_std["pos_rmse"],
        "accel_per_step": l2_sigma_per_step(pred_maps_accel[:Tf], true_maps[:Tf]).tolist(),
        "vel_per_step": r_std.get("vel_per_step", []),
        "pos_per_step": r_std.get("pos_per_step", []),
    }


# ── Plotting ───────────────────────────────────────────────────

def plot_comparison_bar(results, title, save_path):
    labels = [r["label"] for r in results]
    vel_vals = [r["vel_rmse"] for r in results]
    pos_vals = [r["pos_rmse"] for r in results]
    improv = [r["improvement_pct"] for r in results]
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.5), 8.5),
                                    gridspec_kw={"height_ratios": [2, 1]})
    ax1.bar(x - width/2, vel_vals, width, color="#2166ac", alpha=0.85,
            label="Velocity", edgecolor="white")
    ax1.bar(x + width/2, pos_vals, width, color="#b2182b", alpha=0.85,
            label="Positions", edgecolor="white")
    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE"); ax1.set_title(title, fontsize=13)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.legend(); ax1.grid(axis="y", alpha=0.25)
    for i in range(n):
        ax1.text(x[i]-width/2, vel_vals[i]+0.003, f"{vel_vals[i]:.3f}",
                 ha="center", va="bottom", fontsize=6.5, color="#2166ac")
        ax1.text(x[i]+width/2, pos_vals[i]+0.003, f"{pos_vals[i]:.3f}",
                 ha="center", va="bottom", fontsize=6.5, color="#b2182b")

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width*1.5, color=colors, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Improvement (%)"); ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(improv):
        ax2.text(i, v+(0.8 if v>0 else -1.5), f"{v:+.1f}%",
                 ha="center", va="bottom" if v>0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_accel_comparison(accel_results, save_path):
    """Three-bar comparison: acceleration vs velocity vs positions."""
    results = [r for r in accel_results if r is not None]
    if not results:
        return
    labels = [r["tag"].replace("accel_", "") for r in results]
    n = len(labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 5))
    ax.bar(x - width, [r["accel_rmse"] for r in results], width,
           color="#4daf4a", alpha=0.85, label="Acceleration RC", edgecolor="white")
    ax.bar(x, [r["vel_rmse"] for r in results], width,
           color="#2166ac", alpha=0.85, label="Velocity RC", edgecolor="white")
    ax.bar(x + width, [r["pos_rmse"] for r in results], width,
           color="#b2182b", alpha=0.85, label="Positions RC", edgecolor="white")

    for i in range(n):
        for dx, val, c in [(-width, results[i]["accel_rmse"], "#4daf4a"),
                           (0, results[i]["vel_rmse"], "#2166ac"),
                           (width, results[i]["pos_rmse"], "#b2182b")]:
            ax.text(x[i]+dx, val+0.005, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, color=c)

    ax.set_ylabel(r"$L^2(\sigma)$ RMSE"); ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_title("Acceleration RC vs Velocity RC vs Positions RC — Real SST", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_error_curves(results_with_steps, title, save_path):
    """Per-timestep error curves for multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_with_steps)))
    for i, r in enumerate(results_with_steps):
        vel_steps = np.array(r.get("vel_per_step", []))
        pos_steps = np.array(r.get("pos_per_step", []))
        if len(vel_steps) > 0:
            ax.plot(vel_steps, color=colors[i], lw=1.2, alpha=0.8,
                    label=f"{r['label']} vel={r['vel_rmse']:.3f}")
        if len(pos_steps) > 0:
            ax.plot(pos_steps, color=colors[i], lw=0.8, alpha=0.4, ls="--")
    ax.set_xlabel("Forecast step"); ax.set_ylabel(r"$L^2(\sigma)$ error")
    ax.set_title(title); ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    traj_raw = np.load(SST_DIR / "particles.npy")
    traj = _normalize01(traj_raw.astype(np.float32))
    T, N, d = traj.shape
    print(f"SST 180d: T={T}, N={N}, d={d}")

    all_results = []
    accel_results = []

    # ═════════════════════════════════════════════════════════════
    # STEP 1: Adaptive tau + OT interpolation combined
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: Adaptive tau + OT interpolation combined")
    print("=" * 70)

    # OT interpolate 2x, then adaptive-tau resample to various lengths
    print("  OT interpolating 2x...")
    traj_ot2 = interpolate_measures_ot(traj, 2)
    print(f"  OT 2x shape: {traj_ot2.shape}")

    for target in [200, 330, 500]:
        traj_combined, _ = resample_adaptive_tau(traj_ot2, target)
        tag = f"ns1_ot2x_adapt_{target}"
        r = run_on_trajectory(traj_combined, tag, n_cycles=max(3, target // 55),
                              assignment="per_frame")
        r["label"] = f"OT-2x + adapt T={target}"
        all_results.append(r)
        status = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{status}] {r['label']}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    # Also try: linear 4x then adaptive tau
    traj_lin4 = interpolate_measures_linear(traj, 4)
    for target in [330, 500]:
        traj_combined, _ = resample_adaptive_tau(traj_lin4, target)
        tag = f"ns1_lin4x_adapt_{target}"
        r = run_on_trajectory(traj_combined, tag, n_cycles=max(3, target // 55),
                              assignment="per_frame")
        r["label"] = f"Lin-4x + adapt T={target}"
        all_results.append(r)
        status = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{status}] {r['label']}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    # ═════════════════════════════════════════════════════════════
    # STEP 2: Acceleration-based RC (predict Δv, double-integrate)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Acceleration-based RC")
    print("=" * 70)

    # On raw data
    r_accel = run_acceleration_rc(traj, "accel_raw_sst", n_cycles=3)
    if r_accel:
        accel_results.append(r_accel)
        print(f"  Raw SST: accel={r_accel['accel_rmse']:.4f} "
              f"vel={r_accel['vel_rmse']:.4f} pos={r_accel['pos_rmse']:.4f}")

    # On adaptive-tau upsampled data
    traj_adapt330, _ = resample_adaptive_tau(traj, 330)
    r_accel2 = run_acceleration_rc(traj_adapt330, "accel_adapt330", n_cycles=6)
    if r_accel2:
        accel_results.append(r_accel2)
        print(f"  Adaptive 330: accel={r_accel2['accel_rmse']:.4f} "
              f"vel={r_accel2['vel_rmse']:.4f} pos={r_accel2['pos_rmse']:.4f}")

    # On OT-interpolated + adaptive tau
    traj_ot_adapt, _ = resample_adaptive_tau(traj_ot2, 330)
    r_accel3 = run_acceleration_rc(traj_ot_adapt, "accel_ot2_adapt330", n_cycles=6)
    if r_accel3:
        accel_results.append(r_accel3)
        print(f"  OT2+adapt330: accel={r_accel3['accel_rmse']:.4f} "
              f"vel={r_accel3['vel_rmse']:.4f} pos={r_accel3['pos_rmse']:.4f}")

    # ═════════════════════════════════════════════════════════════
    # STEP 3: Optimal interpolation factor (3x, 5x)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Optimal interpolation factor search")
    print("=" * 70)

    for factor in [3, 5]:
        traj_interp = interpolate_measures_linear(traj, factor)
        tag = f"ns3_linear_{factor}x"
        n_cyc = max(3, factor * 3)
        r = run_on_trajectory(traj_interp, tag, n_cycles=n_cyc,
                              assignment="per_frame")
        r["label"] = f"Linear {factor}x (T={traj_interp.shape[0]})"
        all_results.append(r)
        status = "WIN" if r["velocity_wins"] else "loss"
        print(f"  [{status}] {r['label']}: vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f} {r['improvement_pct']:+.1f}%")

    # ═════════════════════════════════════════════════════════════
    # STEP 4: Longer real data (365-day SST)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: Longer real data (365-day SST)")
    print("=" * 70)

    if not (SST_365_DIR / "particles.npy").is_file():
        print("  Downloading 365-day SST data...")
        from data_utils.generate_sst_data import download_and_process_sst
        from datetime import datetime
        try:
            download_and_process_sst(
                start_date=datetime(2024, 4, 1),
                end_date=datetime(2025, 4, 1),
                n_particles=200,
                output_dir=SST_365_DIR,
                use_alternative=False,
                ensure_cyclicality=False,
            )
        except Exception as e:
            print(f"  NOAA download failed ({e}), using synthetic fallback")
            download_and_process_sst(
                start_date=datetime(2024, 4, 1),
                end_date=datetime(2025, 4, 1),
                n_particles=200,
                output_dir=SST_365_DIR,
                use_alternative=True,
                ensure_cyclicality=False,
            )

    if (SST_365_DIR / "particles.npy").is_file():
        traj_365_raw = np.load(SST_365_DIR / "particles.npy")
        traj_365 = _normalize01(traj_365_raw.astype(np.float32))
        T365 = traj_365.shape[0]
        print(f"  365d SST: T={T365}, N={traj_365.shape[1]}, d={traj_365.shape[2]}")

        # Baseline on 365d
        r365 = run_on_trajectory(traj_365, "ns4_sst365_baseline", n_cycles=4,
                                  assignment="per_frame")
        r365["label"] = f"365d baseline (T={T365})"
        all_results.append(r365)
        print(f"  Baseline: vel={r365['vel_rmse']:.4f} pos={r365['pos_rmse']:.4f} "
              f"{r365['improvement_pct']:+.1f}%")

        # Adaptive tau on 365d
        traj_365_adapt, _ = resample_adaptive_tau(traj_365, T365 * 2)
        r365a = run_on_trajectory(traj_365_adapt, "ns4_sst365_adapt",
                                   n_cycles=8, assignment="per_frame")
        r365a["label"] = f"365d adaptive (T={T365 * 2})"
        all_results.append(r365a)
        print(f"  Adaptive: vel={r365a['vel_rmse']:.4f} pos={r365a['pos_rmse']:.4f} "
              f"{r365a['improvement_pct']:+.1f}%")

        # OT interp 2x + adaptive tau on 365d
        print("  OT interpolating 365d 2x (may take a minute)...")
        traj_365_ot2 = interpolate_measures_ot(traj_365, 2)
        traj_365_combo, _ = resample_adaptive_tau(traj_365_ot2, T365 * 2)
        r365c = run_on_trajectory(traj_365_combo, "ns4_sst365_ot2_adapt",
                                   n_cycles=8, assignment="per_frame")
        r365c["label"] = f"365d OT2+adapt (T={T365 * 2})"
        all_results.append(r365c)
        print(f"  OT2+adapt: vel={r365c['vel_rmse']:.4f} pos={r365c['pos_rmse']:.4f} "
              f"{r365c['improvement_pct']:+.1f}%")

        # Acceleration RC on 365d
        r365_accel = run_acceleration_rc(traj_365, "accel_sst365", n_cycles=4)
        if r365_accel:
            accel_results.append(r365_accel)
            print(f"  Accel 365d: accel={r365_accel['accel_rmse']:.4f} "
                  f"vel={r365_accel['vel_rmse']:.4f} pos={r365_accel['pos_rmse']:.4f}")
    else:
        print("  [SKIP] 365d data not available")

    # ═════════════════════════════════════════════════════════════
    # SUMMARY & PLOTS
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FULL RESULTS SUMMARY")
    print("=" * 70)

    for r in sorted(all_results, key=lambda x: -x["improvement_pct"]):
        status = "WIN " if r["velocity_wins"] else "loss"
        print(f"  [{status}] {r['label']:42s}  vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f}  {r['improvement_pct']:+.1f}%")

    wins = [r for r in all_results if r["velocity_wins"]]
    print(f"\nVelocity wins: {len(wins)}/{len(all_results)}")

    if accel_results:
        print("\nAcceleration RC results:")
        for r in accel_results:
            best = "ACCEL" if r["accel_wins_vs_vel"] else "VEL"
            print(f"  {r['tag']:35s}  accel={r['accel_rmse']:.4f} "
                  f"vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f}  best={best}")

    # Plots
    if all_results:
        plot_comparison_bar(all_results, "Next-Steps Experiments — Real SST",
                            PLOTS_DIR / "next_steps_summary.png")
        plot_error_curves(all_results, "Per-step error curves — all experiments",
                          PLOTS_DIR / "next_steps_error_curves.png")

    if accel_results:
        plot_accel_comparison(accel_results,
                              PLOTS_DIR / "acceleration_rc_comparison.png")

    # Save report
    report = {
        "experiments": [{k: v for k, v in r.items()
                         if k not in ("vel_per_step", "pos_per_step")}
                        for r in all_results],
        "acceleration_experiments": [{k: v for k, v in r.items()
                                      if k not in ("accel_per_step", "vel_per_step", "pos_per_step")}
                                     for r in accel_results],
        "n_velocity_wins": len(wins),
        "n_total": len(all_results),
    }
    report_path = REPO_ROOT / "next_steps_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")
    print(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
