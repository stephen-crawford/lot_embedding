#!/usr/bin/env python3
"""
Research experiments: measure interpolation, adaptive tau, constant velocity
reparametrization, and snapshot vs non-snapshot references on real SST data.

Directions explored:
  1. Displacement interpolation between timestep measures (McCann-style in LOT space)
  2. Adaptive tau: more points where velocity is large, fewer where small
  3. Constant-velocity reparametrization: retime so ||v_t|| is constant
  4. Snapshot-begin reference vs gaussian_iso reference
  5. Acceleration analysis and visualization
"""
from __future__ import annotations

import importlib.util
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

SST_DIR = REPO_ROOT / "sst_data_180d"


# ── Metrics ────────────────────────────────────────────────────

def l2_sigma_per_step(pred, true):
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))

def l2_sigma_rmse(pred, true):
    per_step = l2_sigma_per_step(pred, true)
    return float(np.sqrt(np.mean(per_step ** 2)))


# ── Pipeline loader ────────────────────────────────────────────

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_on_trajectory(traj, tag, n_cycles, lot_kind="gaussian_iso",
                      assignment="per_frame", scale=1.0, sr=None, lk=None, rd=None):
    """Save trajectory to disk, run pipeline, return results."""
    results_root = REPO_ROOT / "results"
    traj_dir = results_root / f"{tag}_N_{traj.shape[1]}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / "trajectory.npy"
    np.save(traj_path, traj)

    # Create a temp goes-like dir with our trajectory
    tmp_dir = REPO_ROOT / f"_tmp_interp_{tag}"
    tmp_dir.mkdir(exist_ok=True)
    np.save(tmp_dir / "particles.npy", traj)

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
    vel_rmse = l2_sigma_rmse(vp[:Tf], vt[:Tf])
    pos_rmse = l2_sigma_rmse(pp[:Tf], pt[:Tf])

    # Clean up temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "tag": tag,
        "vel_rmse": vel_rmse,
        "pos_rmse": pos_rmse,
        "velocity_wins": vel_rmse < pos_rmse,
        "improvement_pct": (pos_rmse - vel_rmse) / pos_rmse * 100 if pos_rmse > 0 else 0,
        "T": traj.shape[0],
        "forecast_steps": Tf,
        "vel_dir": str(vel_dir),
        "pos_dir": str(pos_dir),
    }


# ═══════════════════════════════════════════════════════════════
# 1. DISPLACEMENT INTERPOLATION (McCann-style in particle space)
# ═══════════════════════════════════════════════════════════════

def interpolate_measures_linear(traj, factor):
    """
    Interpolate between consecutive measures using OT displacement
    interpolation in particle space: mu_{t+s} = (1-s)*mu_t + s*mu_{t+1}.

    Since particles are paired (same index = same "identity"), this is
    equivalent to McCann interpolation along the OT geodesic when the
    identity is the OT map.

    Parameters
    ----------
    traj : (T, N, d) array
    factor : int — number of sub-steps between each original pair

    Returns
    -------
    (T_new, N, d) array where T_new = (T-1)*factor + 1
    """
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
    """
    Interpolate using per-pair OT plans (handles non-identity correspondences).
    Solves OT(mu_t, mu_{t+1}), then interpolates along the displacement.
    """
    import ot as pot
    T, N, d = traj.shape
    T_new = (T - 1) * factor + 1
    out = np.empty((T_new, N, d), dtype=np.float64)
    w = np.full(N, 1.0 / N)

    for i in range(T - 1):
        src = traj[i].astype(np.float64)
        tgt = traj[i + 1].astype(np.float64)
        # Cost matrix
        M = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
        # OT plan
        G = pot.emd(w, w, M)
        # Barycentric map: T(x_j) = sum_k G_{jk} * y_k / sum_k G_{jk}
        row_sums = G.sum(axis=1, keepdims=True) + 1e-18
        T_map = (G @ tgt) / row_sums  # (N, d) — where each source particle goes

        for k in range(factor):
            s = k / factor
            out[i * factor + k] = (1 - s) * src + s * T_map
    out[-1] = traj[-1].astype(np.float64)
    return out.astype(traj.dtype)


# ═══════════════════════════════════════════════════════════════
# 2. ADAPTIVE TAU (variable timestep based on velocity)
# ═══════════════════════════════════════════════════════════════

def compute_velocity_norms(traj):
    """Per-step velocity norm: ||mu_{t+1} - mu_t||_{L2}."""
    diffs = traj[1:] - traj[:-1]  # (T-1, N, d)
    sq_norms = np.sum(diffs ** 2, axis=-1)  # (T-1, N)
    return np.sqrt(np.mean(sq_norms, axis=-1))  # (T-1,)


def resample_adaptive_tau(traj, target_T):
    """
    Resample trajectory with adaptive tau: more points where velocity is
    large, fewer where small. Creates a new trajectory of length target_T
    by allocating sub-steps proportional to velocity magnitude.
    """
    v_norms = compute_velocity_norms(traj)
    T_orig = traj.shape[0]

    # Cumulative arc length in L2(sigma) space
    arc = np.concatenate([[0], np.cumsum(v_norms)])
    total_arc = arc[-1]

    # Uniform in arc-length → adaptive in time
    target_arcs = np.linspace(0, total_arc, target_T)
    # Map back to original time indices (fractional)
    t_indices = np.interp(target_arcs, arc, np.arange(T_orig))

    # Interpolate trajectory at these fractional times
    out = np.empty((target_T, traj.shape[1], traj.shape[2]), dtype=traj.dtype)
    for i, ti in enumerate(t_indices):
        t_lo = int(np.floor(ti))
        t_hi = min(t_lo + 1, T_orig - 1)
        frac = ti - t_lo
        out[i] = (1 - frac) * traj[t_lo] + frac * traj[t_hi]
    return out, t_indices


# ═══════════════════════════════════════════════════════════════
# 3. CONSTANT VELOCITY REPARAMETRIZATION
# ═══════════════════════════════════════════════════════════════

def reparametrize_constant_velocity(traj, target_T):
    """
    Reparametrize time so ||v_t|| is approximately constant (arc-length
    parametrization). This is exactly adaptive tau — we resample uniformly
    in arc-length. The difference is conceptual: this transforms the
    *dynamics* into constant-speed, making velocity prediction trivial
    (only direction matters).
    """
    return resample_adaptive_tau(traj, target_T)


def reparametrize_constant_acceleration(traj, target_T):
    """
    Reparametrize so that the *change in velocity* (acceleration) is
    approximately constant. We compute the acceleration arc-length and
    resample uniformly along it.
    """
    v_norms = compute_velocity_norms(traj)
    T_orig = traj.shape[0]

    # Acceleration = change in velocity norms
    if len(v_norms) < 2:
        return resample_adaptive_tau(traj, target_T)

    accel = np.abs(np.diff(v_norms))  # (T-2,)
    accel = np.concatenate([[accel[0]], accel])  # pad to T-1

    # Cumulative "acceleration arc-length"
    arc = np.concatenate([[0], np.cumsum(accel + 1e-12)])  # +eps for flat regions
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


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_interpolation_check(traj_orig, traj_interp, factor, save_path):
    """Verify interpolation makes sense: show original + interpolated frames."""
    T_orig = traj_orig.shape[0]
    n_show = min(5, T_orig - 1)
    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))

    for col, t_orig in enumerate(np.linspace(0, T_orig - 2, n_show, dtype=int)):
        # Original frame
        ax = axes[0, col]
        ax.scatter(traj_orig[t_orig, :, 0], traj_orig[t_orig, :, 1],
                   s=6, alpha=0.5, c="black")
        ax.set_title(f"Original t={t_orig}", fontsize=9)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.grid(alpha=0.15)
        if col == 0:
            ax.set_ylabel("Original", fontsize=10, fontweight="bold")

        # Interpolated mid-frame
        t_interp = t_orig * factor + factor // 2
        if t_interp >= traj_interp.shape[0]:
            t_interp = traj_interp.shape[0] - 1
        ax = axes[1, col]
        ax.scatter(traj_interp[t_interp, :, 0], traj_interp[t_interp, :, 1],
                   s=6, alpha=0.5, c="#2166ac")
        # Overlay original endpoints as faint dots
        ax.scatter(traj_orig[t_orig, :, 0], traj_orig[t_orig, :, 1],
                   s=3, alpha=0.15, c="gray")
        ax.scatter(traj_orig[min(t_orig+1, T_orig-1), :, 0],
                   traj_orig[min(t_orig+1, T_orig-1), :, 1],
                   s=3, alpha=0.15, c="gray")
        ax.set_title(f"Interp t={t_interp} (mid)", fontsize=9, color="#2166ac")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal"); ax.grid(alpha=0.15)
        if col == 0:
            ax.set_ylabel(f"Interpolated ({factor}x)", fontsize=10,
                          fontweight="bold", color="#2166ac")

    fig.suptitle(f"Displacement interpolation verification ({factor}x upsampling)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_velocity_analysis(traj, label, save_path):
    """Velocity norms, acceleration, and distribution analysis."""
    v_norms = compute_velocity_norms(traj)
    accel = np.diff(v_norms) if len(v_norms) > 1 else np.array([0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Velocity norms over time
    ax = axes[0, 0]
    ax.plot(v_norms, color="#2166ac", lw=1.2)
    ax.set_xlabel("Timestep"); ax.set_ylabel(r"$\||v_t\||_{L^2(\sigma)}$")
    ax.set_title(f"Velocity magnitude — {label}"); ax.grid(alpha=0.25)
    ax.axhline(np.mean(v_norms), color="red", ls="--", lw=0.8,
               label=f"mean={np.mean(v_norms):.4f}")
    ax.legend(fontsize=9)

    # Acceleration over time
    ax = axes[0, 1]
    ax.plot(accel, color="#b2182b", lw=1.0)
    ax.set_xlabel("Timestep"); ax.set_ylabel(r"$\Delta\||v_t\||$")
    ax.set_title("Acceleration (velocity change)"); ax.grid(alpha=0.25)
    ax.axhline(0, color="black", lw=0.5)

    # Velocity histogram
    ax = axes[1, 0]
    ax.hist(v_norms, bins=30, color="#2166ac", alpha=0.7, edgecolor="white")
    ax.set_xlabel(r"$\||v_t\||$"); ax.set_ylabel("Count")
    ax.set_title("Velocity magnitude distribution"); ax.grid(alpha=0.25)
    cv = np.std(v_norms) / np.mean(v_norms) if np.mean(v_norms) > 0 else 0
    ax.text(0.95, 0.95, f"CV = {cv:.3f}\nstd/mean", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Cumulative arc length
    ax = axes[1, 1]
    arc = np.concatenate([[0], np.cumsum(v_norms)])
    ax.plot(arc, color="#4daf4a", lw=1.5)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Cumulative arc length")
    ax.set_title("Arc-length growth"); ax.grid(alpha=0.25)
    # Mark where it's linear (constant velocity) vs curved
    ax.plot([0, len(arc)-1], [0, arc[-1]], "r--", lw=0.8, alpha=0.5,
            label="Constant velocity reference")
    ax.legend(fontsize=9)

    fig.suptitle(f"Velocity field analysis — {label}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_reparametrization_comparison(v_orig, v_const_vel, v_const_acc, save_path):
    """Compare velocity profiles: original vs constant-vel vs constant-accel."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, v, title, color in [
        (axes[0], v_orig, "Original (variable speed)", "#b2182b"),
        (axes[1], v_const_vel, "Constant velocity reparam.", "#2166ac"),
        (axes[2], v_const_acc, "Constant acceleration reparam.", "#4daf4a"),
    ]:
        ax.plot(v, color=color, lw=1.2)
        ax.axhline(np.mean(v), color="gray", ls="--", lw=0.8)
        cv = np.std(v) / np.mean(v) if np.mean(v) > 0 else 0
        ax.set_title(f"{title}\nCV = {cv:.3f}", fontsize=10)
        ax.set_xlabel("Timestep"); ax.set_ylabel(r"$\||v_t\||$")
        ax.grid(alpha=0.25)

    fig.suptitle("Velocity profile comparison across reparametrizations", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_results_comparison(results_list, save_path):
    """Bar chart comparing all experiment configurations."""
    labels = [r["label"] for r in results_list]
    vel_vals = [r["vel_rmse"] for r in results_list]
    pos_vals = [r["pos_rmse"] for r in results_list]
    improv = [r["improvement_pct"] for r in results_list]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.4), 9),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - width/2, vel_vals, width, color="#2166ac", alpha=0.85,
            label="Velocity", edgecolor="white")
    ax1.bar(x + width/2, pos_vals, width, color="#b2182b", alpha=0.85,
            label="Positions", edgecolor="white")
    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE", fontsize=12)
    ax1.set_title("Real SST — Interpolation & Reparametrization Experiments", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.legend(fontsize=10); ax1.grid(axis="y", alpha=0.25)

    for i in range(n):
        ax1.text(x[i] - width/2, vel_vals[i] + 0.005, f"{vel_vals[i]:.3f}",
                 ha="center", va="bottom", fontsize=6.5, color="#2166ac")
        ax1.text(x[i] + width/2, pos_vals[i] + 0.005, f"{pos_vals[i]:.3f}",
                 ha="center", va="bottom", fontsize=6.5, color="#b2182b")

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Vel. improvement (%)", fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(improv):
        ax2.text(i, v + (0.8 if v > 0 else -1.5), f"{v:+.1f}%",
                 ha="center", va="bottom" if v > 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

def main():
    def _normalize01(t, margin=0.02):
        lo = t.min(axis=(0, 1), keepdims=True)
        hi = t.max(axis=(0, 1), keepdims=True)
        span = np.maximum(hi - lo, 1e-9)
        u = (t - lo) / span
        return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)

    traj_raw = np.load(SST_DIR / "particles.npy")
    traj = _normalize01(traj_raw.astype(np.float32))
    T, N, d = traj.shape
    print(f"SST 180d data: T={T}, N={N}, d={d}")

    all_results = []

    # ── Velocity analysis of raw data ──────────────────────────
    print("\n" + "=" * 60)
    print("VELOCITY FIELD ANALYSIS")
    print("=" * 60)
    plot_velocity_analysis(traj, "Raw SST 180d", PLOTS_DIR / "sst_velocity_analysis_raw.png")

    # ── Baseline: raw data, per_frame ──────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE: Raw SST data")
    print("=" * 60)
    r = run_on_trajectory(traj, "interp_baseline", n_cycles=3,
                          assignment="per_frame")
    r["label"] = "Baseline (raw)"
    all_results.append(r)
    print(f"  vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
          f"improvement={r['improvement_pct']:+.1f}%")

    # ── Experiment 1: Displacement interpolation ───────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Displacement interpolation")
    print("=" * 60)

    for factor in [2, 4, 8]:
        print(f"\n--- Linear interpolation {factor}x ---")
        traj_interp = interpolate_measures_linear(traj, factor)
        print(f"  Interpolated: {traj_interp.shape}")

        if factor == 4:
            plot_interpolation_check(traj, traj_interp, factor,
                                     PLOTS_DIR / f"sst_interp_check_{factor}x.png")
            plot_velocity_analysis(traj_interp, f"Linear interp {factor}x",
                                   PLOTS_DIR / f"sst_velocity_analysis_interp_{factor}x.png")

        n_cyc = max(3, factor * 3)
        r = run_on_trajectory(traj_interp, f"interp_linear_{factor}x",
                              n_cycles=n_cyc, assignment="per_frame")
        r["label"] = f"Linear {factor}x (T={traj_interp.shape[0]})"
        all_results.append(r)
        print(f"  vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
              f"improvement={r['improvement_pct']:+.1f}%")

    # OT-based interpolation (slower but more principled)
    print(f"\n--- OT interpolation 2x ---")
    traj_ot2 = interpolate_measures_ot(traj, 2)
    plot_interpolation_check(traj, traj_ot2, 2,
                             PLOTS_DIR / "sst_interp_check_ot_2x.png")
    r = run_on_trajectory(traj_ot2, "interp_ot_2x", n_cycles=6,
                          assignment="per_frame")
    r["label"] = f"OT interp 2x (T={traj_ot2.shape[0]})"
    all_results.append(r)
    print(f"  vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
          f"improvement={r['improvement_pct']:+.1f}%")

    # ── Experiment 2: Adaptive tau ─────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Adaptive tau (arc-length reparametrization)")
    print("=" * 60)

    for target_T in [165, 330, 500]:
        traj_adapt, t_idx = resample_adaptive_tau(traj, target_T)
        label = f"Adaptive tau T={target_T}"
        print(f"\n--- {label} ---")
        plot_velocity_analysis(traj_adapt, label,
                               PLOTS_DIR / f"sst_velocity_analysis_adaptive_{target_T}.png")
        n_cyc = max(3, target_T // 55)
        r = run_on_trajectory(traj_adapt, f"adaptive_tau_{target_T}",
                              n_cycles=n_cyc, assignment="per_frame")
        r["label"] = label
        all_results.append(r)
        print(f"  vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
              f"improvement={r['improvement_pct']:+.1f}%")

    # ── Experiment 3: Constant velocity reparametrization ──────
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Constant velocity reparametrization")
    print("=" * 60)

    traj_cv, t_cv = reparametrize_constant_velocity(traj, 330)
    traj_ca, t_ca = reparametrize_constant_acceleration(traj, 330)

    v_orig = compute_velocity_norms(traj)
    v_cv = compute_velocity_norms(traj_cv)
    v_ca = compute_velocity_norms(traj_ca)
    plot_reparametrization_comparison(v_orig, v_cv, v_ca,
                                      PLOTS_DIR / "sst_reparam_velocity_comparison.png")

    for traj_r, tag_r, label_r in [
        (traj_cv, "const_vel_330", "Const velocity T=330"),
        (traj_ca, "const_accel_330", "Const accel T=330"),
    ]:
        plot_velocity_analysis(traj_r, label_r,
                               PLOTS_DIR / f"sst_velocity_analysis_{tag_r}.png")
        r = run_on_trajectory(traj_r, tag_r, n_cycles=6, assignment="per_frame")
        r["label"] = label_r
        all_results.append(r)
        print(f"  {label_r}: vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
              f"improvement={r['improvement_pct']:+.1f}%")

    # ── Experiment 4: Snapshot vs non-snapshot reference ───────
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Snapshot vs non-snapshot reference")
    print("=" * 60)

    # Use adaptive-tau upsampled data (best chance for velocity win)
    traj_up = interpolate_measures_linear(traj, 4)
    for kind, assign in [
        ("snapshot_begin", "fixed"),
        ("snapshot_begin", "per_frame"),
        ("gaussian_iso", "fixed"),
        ("gaussian_iso", "per_frame"),
        ("uniform_square", "per_frame"),
    ]:
        tag_s = f"ref_{kind}_{assign}_4x"
        label_s = f"{kind} {assign} (4x)"
        r = run_on_trajectory(traj_up, tag_s, n_cycles=12,
                              lot_kind=kind, assignment=assign)
        r["label"] = label_s
        all_results.append(r)
        print(f"  {label_s}: vel={r['vel_rmse']:.4f} pos={r['pos_rmse']:.4f} "
              f"improvement={r['improvement_pct']:+.1f}%")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL RESULTS SUMMARY")
    print("=" * 70)
    wins = [r for r in all_results if r["velocity_wins"]]
    for r in sorted(all_results, key=lambda x: -x["improvement_pct"]):
        status = "WIN " if r["velocity_wins"] else "loss"
        print(f"  [{status}] {r['label']:40s}  vel={r['vel_rmse']:.4f} "
              f"pos={r['pos_rmse']:.4f}  {r['improvement_pct']:+.1f}%")

    print(f"\nTotal: {len(wins)}/{len(all_results)} velocity wins")

    # Plot comparison
    plot_results_comparison(all_results,
                            PLOTS_DIR / "sst_interpolation_experiments_summary.png")

    # Save JSON report
    report = {
        "data_source": "NOAA OISST Gulf of Mexico (165 days)",
        "metric": "L2_sigma_RMSE",
        "experiments": all_results,
        "n_velocity_wins": len(wins),
        "n_total": len(all_results),
    }
    report_path = REPO_ROOT / "interpolation_experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")
    print(f"Plots: {PLOTS_DIR}/sst_*")


if __name__ == "__main__":
    main()
