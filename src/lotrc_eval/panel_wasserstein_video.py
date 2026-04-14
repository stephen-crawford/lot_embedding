#!/usr/bin/env python3
"""
2×3 VELOCITY vs POSITIONS Wasserstein panel video.

Supports both 2D and 3D data. Auto-detects dimensionality from input arrays.

Public API
----------
- make_2x3_panel_wasserstein_video(
      velocity_dir, positions_dir, out_path,
      D_vel_W2=None, path_vel=None,
      D_pos_W2=None, path_pos=None,
      eps_vel=0.05, eps_pos=0.05,
      w_vel=None, w_pos=None,
      fps=20, target_duration_sec=None,
      respect_autonomous_window=True,
      plot_aligned=True,
      cost_label_vel="Wasserstein (Sinkhorn divergence)",
      cost_label_pos="Wasserstein (Sinkhorn divergence)",
  )
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import ot  # POT


# ---------- sinkhorn helpers ----------
def _ensure64(x): return np.asarray(x, dtype=np.float64)
def _pairwise_sqdist(A, B):
    A = _ensure64(A); B = _ensure64(B)
    diff = A[:, None, :] - B[None, :, :]
    return np.einsum("nmd,nmd->nm", diff, diff)

def _sinkhorn2_cost(w, M, eps, it=100000, thr=1e-9):
    out = ot.sinkhorn2(_ensure64(w), _ensure64(w), _ensure64(M),
                       reg=float(eps), numItermax=it, stopThr=thr, verbose=False)
    return float(out[0] if isinstance(out, (tuple, list)) else out)

def sinkhorn_divergence(Xa, Xb, w, eps):
    M_ab = _pairwise_sqdist(Xa, Xb)
    M_aa = _pairwise_sqdist(Xa, Xa)
    M_bb = _pairwise_sqdist(Xb, Xb)
    W_ab = _sinkhorn2_cost(w, M_ab, eps)
    W_aa = _sinkhorn2_cost(w, M_aa, eps)
    W_bb = _sinkhorn2_cost(w, M_bb, eps)
    return W_ab - 0.5 * (W_aa + W_bb)


# ---------- tiny utils ----------
def _axis_limits_from_sequences(*seqs):
    """Compute axis limits for 2D data."""
    pts = []
    for arr in seqs:
        if arr is None:
            continue
        pts.append(arr.reshape(-1, 2))
    pts = np.vstack(pts)
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    pad = 0.05 * max(dx, dy, 1.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    L = max(dx, dy) + 2 * pad
    return (cx - L/2, cx + L/2, cy - L/2, cy + L/2)


def _axis_limits_from_sequences_3d(*seqs):
    """Compute axis limits for 3D data."""
    pts = []
    for arr in seqs:
        if arr is None:
            continue
        pts.append(arr.reshape(-1, 3))
    pts = np.vstack(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    deltas = maxs - mins
    pad = 0.05 * max(deltas.max(), 1.0)
    centers = (mins + maxs) / 2
    L = deltas.max() + 2 * pad
    return tuple((c - L/2, c + L/2) for c in centers)

def _dtw_hard_path(D: np.ndarray):
    H = D.shape[0]
    C = np.full((H+1, H+1), np.inf, dtype=np.float64); C[0,0] = 0.0
    for i in range(1, H+1):
        for j in range(1, H+1):
            C[i,j] = D[i-1,j-1] + min(C[i-1,j], C[i,j-1], C[i-1,j-1])
    i, j, path = H, H, []
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        up, left, diag = C[i-1,j], C[i,j-1], C[i-1,j-1]
        k = int(np.argmin((up, left, diag)))
        if k == 0: i -= 1
        elif k == 1: j -= 1
        else: i -= 1; j -= 1
    path.reverse()
    return path

def _aligned_series_from_path(D, path, H):
    # path: list[(t_pred, s_true)]
    t2s = {int(t): int(s) for (t, s) in (path or [])}
    vals = np.full(H, np.nan); idxs = np.full(H, -1, dtype=int)
    if D is None:
        return vals, idxs
    for t in range(H):
        s = t2s.get(t, -1)
        if 0 <= s < D.shape[1]:
            vals[t] = float(D[t, s]); idxs[t] = s
    return vals, idxs

def _save_anim(fig, ani, path_base: Path, fps=25, dpi=150):
    mp4 = path_base.with_suffix(".mp4")
    try:
        ani.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        print(f"[video] Wrote {mp4}")
    except Exception as e:
        gif = path_base.with_suffix(".gif")
        print(f"[video] ffmpeg unavailable/failed ({e}); writing GIF -> {gif}")
        ani.save(str(gif), writer=PillowWriter(fps=fps), dpi=120)

def _fps_for_duration(H, fallback_fps, target_sec):
    if target_sec is None or target_sec <= 0:
        return max(1, int(fallback_fps))
    return max(1, int(round(H / float(target_sec))))


# ---------- main builder ----------
def make_2x3_panel_wasserstein_video(
    velocity_dir: Path,
    positions_dir: Path,
    out_path: Path,
    # Optional: provide precomputed Wasserstein Δ and paths
    D_vel_W2: np.ndarray = None, path_vel: list = None,
    D_pos_W2: np.ndarray = None, path_pos: list = None,
    # If Δ not given, compute unaligned Sinkhorn series with these:
    eps_vel: float = 0.05, eps_pos: float = 0.05,
    w_vel: np.ndarray = None, w_pos: np.ndarray = None,
    # Video timing
    fps: int = 20,
    target_duration_sec: float = None,   # if set, overrides fps to hit ~this length
    # Horizon control
    respect_autonomous_window: bool = True,
    # Display controls
    plot_aligned: bool = True,
    cost_label_vel: str = "Wasserstein (Sinkhorn divergence)",
    cost_label_pos: str = "Wasserstein (Sinkhorn divergence)",
):
    velocity_dir = Path(velocity_dir); positions_dir = Path(positions_dir)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- VELOCITY I/O ----------
    # Both methods now use the same file format
    pred_V = np.load(velocity_dir / "predicted_measures_X.npy")  # (H,R,2)
    true_V = np.load(velocity_dir / "true_measures_X.npy")       # (H,R,2)
    Hvel = min(pred_V.shape[0], true_V.shape[0])

    Hauto_vel = None
    warm_info_vel = velocity_dir / "warm_start_info.npy"
    if warm_info_vel.exists():
        warm, Hauto_val = np.load(warm_info_vel)
        Hauto_vel = int(Hauto_val)
        if respect_autonomous_window and Hauto_vel > 0:
            Hvel = min(Hvel, Hauto_vel)

    pred_V, true_V = pred_V[:Hvel], true_V[:Hvel]
    N_V = pred_V.shape[1]
    if w_vel is None:
        w_vel = np.full(N_V, 1.0 / N_V, dtype=np.float64)

    # ---------- POSITIONS I/O ----------
    pred_P = np.load(positions_dir / "predicted_measures_X.npy")  # (H,R,2)
    true_P = np.load(positions_dir / "true_measures_X.npy")       # (H,R,2)
    Hpos = min(pred_P.shape[0], true_P.shape[0])

    Hauto_pos = None
    warm_info_pos = positions_dir / "warm_start_info.npy"
    if warm_info_pos.exists():
        warm, Hauto_val = np.load(warm_info_pos)
        Hauto_pos = int(Hauto_val)
        if respect_autonomous_window and Hauto_pos > 0:
            Hpos = min(Hpos, Hauto_pos)

    pred_P, true_P = pred_P[:Hpos], true_P[:Hpos]
    N_P = pred_P.shape[1]
    if w_pos is None:
        w_pos = np.full(N_P, 1.0 / N_P, dtype=np.float64)

    # Final horizon = min across VELOCITY/POSITIONS so animation aligns
    H = min(Hvel, Hpos)
    pred_V, true_V = pred_V[:H], true_V[:H]
    pred_P, true_P = pred_P[:H], true_P[:H]

    # ---------- Wasserstein series & (optional) alignment ----------
    diag_V = np.zeros(H); diag_P = np.zeros(H)

    # Auto-compute DTW path if plotting aligned and Δ provided but no path
    if plot_aligned and D_vel_W2 is not None and path_vel is None:
        path_vel = _dtw_hard_path(_ensure64(D_vel_W2))
    if plot_aligned and D_pos_W2 is not None and path_pos is None:
        path_pos = _dtw_hard_path(_ensure64(D_pos_W2))

    # Build aligned series (or NaNs if not plotting)
    if plot_aligned:
        aligned_V, idx_V = _aligned_series_from_path(D_vel_W2, path_vel, H)
        aligned_P, idx_P = _aligned_series_from_path(D_pos_W2, path_pos, H)
    else:
        aligned_V = np.full(H, np.nan); idx_V = np.full(H, -1, dtype=int)
        aligned_P = np.full(H, np.nan); idx_P = np.full(H, -1, dtype=int)

    if D_vel_W2 is None:
        for t in range(H):
            diag_V[t] = sinkhorn_divergence(true_V[t], pred_V[t], w_vel, eps_vel)
    else:
        D_vel_W2 = _ensure64(D_vel_W2)
        diag_V = np.diag(D_vel_W2)[:H]

    if D_pos_W2 is None:
        for t in range(H):
            diag_P[t] = sinkhorn_divergence(true_P[t], pred_P[t], w_pos, eps_pos)
    else:
        D_pos_W2 = _ensure64(D_pos_W2)
        diag_P = np.diag(D_pos_W2)[:H]

    # ---------- axes (shared) ----------
    x0, x1, y0, y1 = _axis_limits_from_sequences(true_V, pred_V, true_P, pred_P)

    # ---------- figure & static artists ----------
    # POSITIONS on top (row 0), VELOCITY on bottom (row 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    ax_p_true, ax_p_pred, ax_p_err = axes[0]  # POSITIONS on top
    ax_v_true, ax_v_pred, ax_v_err = axes[1]  # VELOCITY on bottom

    for ax in (ax_v_true, ax_v_pred, ax_p_true, ax_p_pred):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=0.3)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    ax_p_true.set_title("POSITIONS — True")
    ax_p_pred.set_title("POSITIONS — Predicted")
    ax_v_true.set_title("VELOCITY — True")
    ax_v_pred.set_title("VELOCITY — Predicted")
    ax_p_err.set_title(f"POSITIONS — {cost_label_pos}")
    ax_v_err.set_title(f"VELOCITY — {cost_label_vel}")

    scat_v_true = ax_v_true.scatter([], [], s=5, alpha=0.7)
    scat_v_pred = ax_v_pred.scatter([], [], s=5, alpha=0.7)
    scat_p_true = ax_p_true.scatter([], [], s=5, alpha=0.7)
    scat_p_pred = ax_p_pred.scatter([], [], s=5, alpha=0.7)

    tgrid = np.arange(H)
    (line_vd,) = ax_v_err.plot(tgrid, diag_V, label="unaligned (diag)")
    (line_pd,) = ax_p_err.plot(tgrid, diag_P, label="unaligned (diag)")

    # Only add aligned curves/markers/labels if requested and we have finite values
    if plot_aligned and np.isfinite(aligned_V).any():
        (line_va,) = ax_v_err.plot(tgrid, aligned_V, label="aligned (DTW path)", alpha=0.95)
        mrk_va, = ax_v_err.plot([0], [aligned_V[0] if np.isfinite(aligned_V[0]) else np.nan], "o", ms=6)
    else:
        line_va = None; mrk_va = None

    if plot_aligned and np.isfinite(aligned_P).any():
        (line_pa,) = ax_p_err.plot(tgrid, aligned_P, label="aligned (DTW path)", alpha=0.95)
        mrk_pa, = ax_p_err.plot([0], [aligned_P[0] if np.isfinite(aligned_P[0]) else np.nan], "o", ms=6)
    else:
        line_pa = None; mrk_pa = None

    mrk_vd, = ax_v_err.plot([0], [diag_V[0]], "o", ms=6)
    mrk_pd, = ax_p_err.plot([0], [diag_P[0]], "o", ms=6)

    # share y-lims for comparability
    err_pool = [diag_V, diag_P]
    if plot_aligned:
        if np.isfinite(aligned_V).any(): err_pool.append(aligned_V[np.isfinite(aligned_V)])
        if np.isfinite(aligned_P).any(): err_pool.append(aligned_P[np.isfinite(aligned_P)])
    err_pool = np.concatenate([e if np.ndim(e) else np.array([e]) for e in err_pool if np.size(e) > 0])
    ymin = float(np.nanmin(err_pool)); ymax = float(np.nanmax(err_pool))
    pad = 0.05 * max(ymax - ymin, 1e-12)
    for ax in (ax_v_err, ax_p_err):
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("prediction time index t")
        ax.set_ylabel("cost")
        ax.grid(alpha=0.3)
        ax.legend()

    # dynamic readouts
    txt_v = ax_v_err.text(0.98, 0.90, "", transform=ax_v_err.transAxes, ha="right", va="top", fontsize=9)
    txt_p = ax_p_err.text(0.98, 0.90, "", transform=ax_p_err.transAxes, ha="right", va="top", fontsize=9)

    # ---------- animation update ----------
    def update(t):
        # VELOCITY/POSITIONS scatters
        scat_v_true.set_offsets(true_V[t]); scat_v_pred.set_offsets(pred_V[t])
        scat_p_true.set_offsets(true_P[t]); scat_p_pred.set_offsets(pred_P[t])

        # move markers for unaligned
        mrk_vd.set_data([t], [diag_V[t]])
        mrk_pd.set_data([t], [diag_P[t]])

        # optional aligned markers + s* readouts
        if plot_aligned:
            if mrk_va is not None:
                vVa = aligned_V[t]
                mrk_va.set_data([t], [vVa] if np.isfinite(vVa) else [np.nan])
            if mrk_pa is not None:
                vPa = aligned_P[t]
                mrk_pa.set_data([t], [vPa] if np.isfinite(vPa) else [np.nan])

            sV = int(idx_V[t]) if (idx_V.size and idx_V[t] >= 0) else -1
            sP = int(idx_P[t]) if (idx_P.size and idx_P[t] >= 0) else -1
            ax_v_err.set_xlabel(f"prediction time index t (VELOCITY): t={t}, s*={sV}")
            ax_p_err.set_xlabel(f"prediction time index t (POSITIONS): t={t}, s*={sP}")
            txt_v.set_text(
                f"diag={diag_V[t]:.3g}" + (f"  aligned={aligned_V[t]:.3g}" if np.isfinite(aligned_V[t]) else "")
            )
            txt_p.set_text(
                f"diag={diag_P[t]:.3g}" + (f"  aligned={aligned_P[t]:.3g}" if np.isfinite(aligned_P[t]) else "")
            )
        else:
            # cleaner labels when not plotting aligned
            ax_v_err.set_xlabel(f"prediction time index t (VELOCITY): t={t}")
            ax_p_err.set_xlabel(f"prediction time index t (POSITIONS): t={t}")
            txt_v.set_text(f"diag={diag_V[t]:.3g}")
            txt_p.set_text(f"diag={diag_P[t]:.3g}")

        return (scat_v_true, scat_v_pred, scat_p_true, scat_p_pred,
                mrk_vd, mrk_pd) + ((mrk_va, mrk_pa, txt_v, txt_p) if plot_aligned else (txt_v, txt_p))

    # compute output fps (optionally enforce a target duration)
    fps_out = _fps_for_duration(H, fps, target_duration_sec)
    print("[video] POSITIONS frames:", Hpos, "| VELOCITY frames:", Hvel,
          "| using H:", H,
          f"| Hauto_pos:{Hauto_pos if Hauto_pos is not None else 'n/a'}",
          f"| Hauto_vel:{Hauto_vel if Hauto_vel is not None else 'n/a'}",
          f"| fps_out:{fps_out} ⇒ duration≈{H/fps_out:.2f}s",
          f"| aligned={'on' if plot_aligned else 'off'}")

    ani = FuncAnimation(fig, update, frames=H, interval=int(1000 / max(fps_out, 1)), blit=False)
    _save_anim(fig, ani, out_path, fps=fps_out)
    plt.close(fig)


# ---------- 3D video builder ----------
def make_2x3_panel_wasserstein_video_3d(
    velocity_dir: Path,
    positions_dir: Path,
    out_path: Path,
    # Optional: provide precomputed Wasserstein Δ and paths
    D_vel_W2: np.ndarray = None, path_vel: list = None,
    D_pos_W2: np.ndarray = None, path_pos: list = None,
    # If Δ not given, compute unaligned Sinkhorn series with these:
    eps_vel: float = 0.05, eps_pos: float = 0.05,
    w_vel: np.ndarray = None, w_pos: np.ndarray = None,
    # Video timing
    fps: int = 20,
    target_duration_sec: float = None,
    # Horizon control
    respect_autonomous_window: bool = True,
    # Display controls
    plot_aligned: bool = True,
    cost_label_vel: str = "Wasserstein (Sinkhorn divergence)",
    cost_label_pos: str = "Wasserstein (Sinkhorn divergence)",
    # 3D specific
    elev: float = 20.0,
    azim: float = 45.0,
    rotate_view: bool = False,  # If True, slowly rotate camera during animation
):
    """
    Create 2x3 panel video for 3D point cloud data.

    Same interface as make_2x3_panel_wasserstein_video but renders 3D scatter plots.
    """
    velocity_dir = Path(velocity_dir); positions_dir = Path(positions_dir)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- VELOCITY I/O ----------
    pred_V = np.load(velocity_dir / "predicted_measures_X.npy")  # (H,R,3)
    true_V = np.load(velocity_dir / "true_measures_X.npy")       # (H,R,3)
    Hvel = min(pred_V.shape[0], true_V.shape[0])

    Hauto_vel = None
    warm_info_vel = velocity_dir / "warm_start_info.npy"
    if warm_info_vel.exists():
        warm, Hauto_val = np.load(warm_info_vel)
        Hauto_vel = int(Hauto_val)
        if respect_autonomous_window and Hauto_vel > 0:
            Hvel = min(Hvel, Hauto_vel)

    pred_V, true_V = pred_V[:Hvel], true_V[:Hvel]
    N_V = pred_V.shape[1]
    if w_vel is None:
        w_vel = np.full(N_V, 1.0 / N_V, dtype=np.float64)

    # ---------- POSITIONS I/O ----------
    pred_P = np.load(positions_dir / "predicted_measures_X.npy")  # (H,R,3)
    true_P = np.load(positions_dir / "true_measures_X.npy")       # (H,R,3)
    Hpos = min(pred_P.shape[0], true_P.shape[0])

    Hauto_pos = None
    warm_info_pos = positions_dir / "warm_start_info.npy"
    if warm_info_pos.exists():
        warm, Hauto_val = np.load(warm_info_pos)
        Hauto_pos = int(Hauto_val)
        if respect_autonomous_window and Hauto_pos > 0:
            Hpos = min(Hpos, Hauto_pos)

    pred_P, true_P = pred_P[:Hpos], true_P[:Hpos]
    N_P = pred_P.shape[1]
    if w_pos is None:
        w_pos = np.full(N_P, 1.0 / N_P, dtype=np.float64)

    # Final horizon = min across VELOCITY/POSITIONS
    H = min(Hvel, Hpos)
    pred_V, true_V = pred_V[:H], true_V[:H]
    pred_P, true_P = pred_P[:H], true_P[:H]

    # ---------- Wasserstein series & (optional) alignment ----------
    diag_V = np.zeros(H); diag_P = np.zeros(H)

    if plot_aligned and D_vel_W2 is not None and path_vel is None:
        path_vel = _dtw_hard_path(_ensure64(D_vel_W2))
    if plot_aligned and D_pos_W2 is not None and path_pos is None:
        path_pos = _dtw_hard_path(_ensure64(D_pos_W2))

    if plot_aligned:
        aligned_V, idx_V = _aligned_series_from_path(D_vel_W2, path_vel, H)
        aligned_P, idx_P = _aligned_series_from_path(D_pos_W2, path_pos, H)
    else:
        aligned_V = np.full(H, np.nan); idx_V = np.full(H, -1, dtype=int)
        aligned_P = np.full(H, np.nan); idx_P = np.full(H, -1, dtype=int)

    if D_vel_W2 is None:
        for t in range(H):
            diag_V[t] = sinkhorn_divergence(true_V[t], pred_V[t], w_vel, eps_vel)
    else:
        D_vel_W2 = _ensure64(D_vel_W2)
        diag_V = np.diag(D_vel_W2)[:H]

    if D_pos_W2 is None:
        for t in range(H):
            diag_P[t] = sinkhorn_divergence(true_P[t], pred_P[t], w_pos, eps_pos)
    else:
        D_pos_W2 = _ensure64(D_pos_W2)
        diag_P = np.diag(D_pos_W2)[:H]

    # ---------- 3D axis limits ----------
    (x0, x1), (y0, y1), (z0, z1) = _axis_limits_from_sequences_3d(true_V, pred_V, true_P, pred_P)

    # ---------- figure with 3D subplots ----------
    fig = plt.figure(figsize=(16, 9))

    # Create 2x3 grid: 4 3D scatter plots + 2 line plots
    # Row 0: POSITIONS (True, Pred, Error curve)
    # Row 1: VELOCITY (True, Pred, Error curve)

    ax_p_true = fig.add_subplot(2, 3, 1, projection='3d')
    ax_p_pred = fig.add_subplot(2, 3, 2, projection='3d')
    ax_p_err = fig.add_subplot(2, 3, 3)

    ax_v_true = fig.add_subplot(2, 3, 4, projection='3d')
    ax_v_pred = fig.add_subplot(2, 3, 5, projection='3d')
    ax_v_err = fig.add_subplot(2, 3, 6)

    # Configure 3D axes
    for ax in (ax_v_true, ax_v_pred, ax_p_true, ax_p_pred):
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)

    ax_p_true.set_title("POSITIONS — True")
    ax_p_pred.set_title("POSITIONS — Predicted")
    ax_v_true.set_title("VELOCITY — True")
    ax_v_pred.set_title("VELOCITY — Predicted")
    ax_p_err.set_title(f"POSITIONS — {cost_label_pos}")
    ax_v_err.set_title(f"VELOCITY — {cost_label_vel}")

    # Initial 3D scatter plots
    scat_v_true = ax_v_true.scatter(true_V[0, :, 0], true_V[0, :, 1], true_V[0, :, 2],
                                     s=5, alpha=0.7, c='blue')
    scat_v_pred = ax_v_pred.scatter(pred_V[0, :, 0], pred_V[0, :, 1], pred_V[0, :, 2],
                                     s=5, alpha=0.7, c='orange')
    scat_p_true = ax_p_true.scatter(true_P[0, :, 0], true_P[0, :, 1], true_P[0, :, 2],
                                     s=5, alpha=0.7, c='blue')
    scat_p_pred = ax_p_pred.scatter(pred_P[0, :, 0], pred_P[0, :, 1], pred_P[0, :, 2],
                                     s=5, alpha=0.7, c='orange')

    # Error curves
    tgrid = np.arange(H)
    (line_vd,) = ax_v_err.plot(tgrid, diag_V, label="unaligned (diag)")
    (line_pd,) = ax_p_err.plot(tgrid, diag_P, label="unaligned (diag)")

    if plot_aligned and np.isfinite(aligned_V).any():
        (line_va,) = ax_v_err.plot(tgrid, aligned_V, label="aligned (DTW path)", alpha=0.95)
        mrk_va, = ax_v_err.plot([0], [aligned_V[0] if np.isfinite(aligned_V[0]) else np.nan], "o", ms=6)
    else:
        line_va = None; mrk_va = None

    if plot_aligned and np.isfinite(aligned_P).any():
        (line_pa,) = ax_p_err.plot(tgrid, aligned_P, label="aligned (DTW path)", alpha=0.95)
        mrk_pa, = ax_p_err.plot([0], [aligned_P[0] if np.isfinite(aligned_P[0]) else np.nan], "o", ms=6)
    else:
        line_pa = None; mrk_pa = None

    mrk_vd, = ax_v_err.plot([0], [diag_V[0]], "o", ms=6)
    mrk_pd, = ax_p_err.plot([0], [diag_P[0]], "o", ms=6)

    # Y-limits for error plots
    err_pool = [diag_V, diag_P]
    if plot_aligned:
        if np.isfinite(aligned_V).any(): err_pool.append(aligned_V[np.isfinite(aligned_V)])
        if np.isfinite(aligned_P).any(): err_pool.append(aligned_P[np.isfinite(aligned_P)])
    err_pool = np.concatenate([e if np.ndim(e) else np.array([e]) for e in err_pool if np.size(e) > 0])
    ymin = float(np.nanmin(err_pool)); ymax = float(np.nanmax(err_pool))
    pad = 0.05 * max(ymax - ymin, 1e-12)
    for ax in (ax_v_err, ax_p_err):
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("prediction time index t")
        ax.set_ylabel("cost")
        ax.grid(alpha=0.3)
        ax.legend()

    txt_v = ax_v_err.text(0.98, 0.90, "", transform=ax_v_err.transAxes, ha="right", va="top", fontsize=9)
    txt_p = ax_p_err.text(0.98, 0.90, "", transform=ax_p_err.transAxes, ha="right", va="top", fontsize=9)

    # ---------- animation update ----------
    def update(t):
        # Update 3D scatter plots by removing and re-adding (matplotlib 3D limitation)
        ax_v_true.clear()
        ax_v_pred.clear()
        ax_p_true.clear()
        ax_p_pred.clear()

        # Reset limits and labels
        for ax in (ax_v_true, ax_v_pred, ax_p_true, ax_p_pred):
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

        # Compute view angle (rotate if requested)
        current_azim = azim + (t * 360 / H) if rotate_view else azim
        for ax in (ax_v_true, ax_v_pred, ax_p_true, ax_p_pred):
            ax.view_init(elev=elev, azim=current_azim)

        # Re-scatter
        ax_v_true.scatter(true_V[t, :, 0], true_V[t, :, 1], true_V[t, :, 2], s=5, alpha=0.7, c='blue')
        ax_v_pred.scatter(pred_V[t, :, 0], pred_V[t, :, 1], pred_V[t, :, 2], s=5, alpha=0.7, c='orange')
        ax_p_true.scatter(true_P[t, :, 0], true_P[t, :, 1], true_P[t, :, 2], s=5, alpha=0.7, c='blue')
        ax_p_pred.scatter(pred_P[t, :, 0], pred_P[t, :, 1], pred_P[t, :, 2], s=5, alpha=0.7, c='orange')

        ax_v_true.set_title("VELOCITY — True")
        ax_v_pred.set_title("VELOCITY — Predicted")
        ax_p_true.set_title("POSITIONS — True")
        ax_p_pred.set_title("POSITIONS — Predicted")

        # Move markers on error plots
        mrk_vd.set_data([t], [diag_V[t]])
        mrk_pd.set_data([t], [diag_P[t]])

        if plot_aligned:
            if mrk_va is not None:
                vVa = aligned_V[t]
                mrk_va.set_data([t], [vVa] if np.isfinite(vVa) else [np.nan])
            if mrk_pa is not None:
                vPa = aligned_P[t]
                mrk_pa.set_data([t], [vPa] if np.isfinite(vPa) else [np.nan])

            sV = int(idx_V[t]) if (idx_V.size and idx_V[t] >= 0) else -1
            sP = int(idx_P[t]) if (idx_P.size and idx_P[t] >= 0) else -1
            ax_v_err.set_xlabel(f"prediction time index t (VELOCITY): t={t}, s*={sV}")
            ax_p_err.set_xlabel(f"prediction time index t (POSITIONS): t={t}, s*={sP}")
            txt_v.set_text(
                f"diag={diag_V[t]:.3g}" + (f"  aligned={aligned_V[t]:.3g}" if np.isfinite(aligned_V[t]) else "")
            )
            txt_p.set_text(
                f"diag={diag_P[t]:.3g}" + (f"  aligned={aligned_P[t]:.3g}" if np.isfinite(aligned_P[t]) else "")
            )
        else:
            ax_v_err.set_xlabel(f"prediction time index t (VELOCITY): t={t}")
            ax_p_err.set_xlabel(f"prediction time index t (POSITIONS): t={t}")
            txt_v.set_text(f"diag={diag_V[t]:.3g}")
            txt_p.set_text(f"diag={diag_P[t]:.3g}")

        return []

    # Compute output fps
    fps_out = _fps_for_duration(H, fps, target_duration_sec)
    print("[video 3D] POSITIONS frames:", Hpos, "| VELOCITY frames:", Hvel,
          "| using H:", H,
          f"| Hauto_pos:{Hauto_pos if Hauto_pos is not None else 'n/a'}",
          f"| Hauto_vel:{Hauto_vel if Hauto_vel is not None else 'n/a'}",
          f"| fps_out:{fps_out} ⇒ duration≈{H/fps_out:.2f}s",
          f"| aligned={'on' if plot_aligned else 'off'}",
          f"| rotate={'on' if rotate_view else 'off'}")

    plt.tight_layout()
    ani = FuncAnimation(fig, update, frames=H, interval=int(1000 / max(fps_out, 1)), blit=False)
    _save_anim(fig, ani, out_path, fps=fps_out)
    plt.close(fig)


# ---------- auto-detecting wrapper ----------
def make_panel_wasserstein_video_auto(
    velocity_dir: Path,
    positions_dir: Path,
    out_path: Path,
    **kwargs
):
    """
    Auto-detect 2D vs 3D data and call appropriate video function.

    Checks dimensionality from predicted_measures_X.npy and dispatches
    to either make_2x3_panel_wasserstein_video (2D) or
    make_2x3_panel_wasserstein_video_3d (3D).
    """
    velocity_dir = Path(velocity_dir)
    pred = np.load(velocity_dir / "predicted_measures_X.npy")
    d = pred.shape[-1]

    if d == 2:
        print("[video] Detected 2D data, using 2D visualization")
        return make_2x3_panel_wasserstein_video(velocity_dir, positions_dir, out_path, **kwargs)
    elif d == 3:
        print("[video] Detected 3D data, using 3D visualization")
        # Filter out 2D-only kwargs if present
        kwargs_3d = {k: v for k, v in kwargs.items()
                     if k not in []}  # No 2D-only kwargs currently
        return make_2x3_panel_wasserstein_video_3d(velocity_dir, positions_dir, out_path, **kwargs_3d)
    else:
        raise ValueError(f"Unsupported dimensionality d={d}. Expected 2 or 3.")


# ---------- grid-based visualization for wind/field data ----------
def make_grid_comparison_video(
    true_grid: np.ndarray,
    velocity_pred: np.ndarray,
    positions_pred: np.ndarray,
    out_path: Path,
    grid_shape: tuple = None,
    title: str = "Forecast Comparison",
    cmap: str = "viridis",
    fps: int = 10,
    target_duration_sec: float = None,
    show_error_diff: bool = True,
    field_label: str = "Field magnitude",
):
    """
    Create comparison video for gridded field data (wind, SST, clouds, etc.).

    Shows spatial heatmaps instead of point cloud scatter plots.
    Layout:
        Row 1: Ground Truth | POSITIONS | VELOCITY
        Row 2: Error curves | Error difference map (blue=velocity better)

    Parameters
    ----------
    true_grid : (T, H, W) or (T, N) array
        Ground truth field values. If 2D, grid_shape must be provided.
    velocity_pred : (T, H, W) or (T, N) array
        VELOCITY pipeline predictions
    positions_pred : (T, H, W) or (T, N) array
        POSITIONS pipeline predictions
    out_path : Path
        Output video path (will add .gif or .mp4 suffix)
    grid_shape : tuple (H, W), optional
        If arrays are flattened (T, N), reshape to this grid
    title : str
        Video title
    cmap : str
        Colormap for field visualization
    fps : int
        Frames per second
    target_duration_sec : float, optional
        Target video duration (overrides fps)
    show_error_diff : bool
        Whether to show error difference map (blue=velocity better)
    field_label : str
        Label for colorbar
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle shape
    if true_grid.ndim == 2 and grid_shape is not None:
        T, N = true_grid.shape
        H_grid, W_grid = grid_shape
        true_grid = true_grid.reshape(T, H_grid, W_grid)
        velocity_pred = velocity_pred.reshape(T, H_grid, W_grid)
        positions_pred = positions_pred.reshape(T, H_grid, W_grid)
    elif true_grid.ndim == 3:
        T, H_grid, W_grid = true_grid.shape
    else:
        raise ValueError(f"true_grid must be (T, H, W) or (T, N) with grid_shape. Got {true_grid.shape}")

    # Compute errors over time
    errors_vel = np.sqrt(np.mean((velocity_pred - true_grid)**2, axis=(1, 2)))
    errors_pos = np.sqrt(np.mean((positions_pred - true_grid)**2, axis=(1, 2)))

    # Global color limits
    vmin = min(true_grid.min(), velocity_pred.min(), positions_pred.min())
    vmax = max(true_grid.max(), velocity_pred.max(), positions_pred.max())

    # Error diff limits
    err_diff_max = max(np.abs(positions_pred - true_grid).max(),
                       np.abs(velocity_pred - true_grid).max())

    # Setup figure
    if show_error_diff:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
        ax_true = fig.add_subplot(gs[0, 0])
        ax_pos = fig.add_subplot(gs[0, 1])
        ax_vel = fig.add_subplot(gs[0, 2])
        ax_err = fig.add_subplot(gs[1, 0:2])
        ax_diff = fig.add_subplot(gs[1, 2])
    else:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        ax_true, ax_pos, ax_vel = axes[0]
        ax_err = axes[1, 0]
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        ax_diff = None

    # Initial plots
    im_true = ax_true.imshow(true_grid[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax_true.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax_true.set_xticks([]); ax_true.set_yticks([])
    plt.colorbar(im_true, ax=ax_true, label=field_label, shrink=0.8)

    im_pos = ax_pos.imshow(positions_pred[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    pos_title = ax_pos.set_title('POSITIONS', fontsize=12, fontweight='bold', color='red')
    ax_pos.set_xticks([]); ax_pos.set_yticks([])
    plt.colorbar(im_pos, ax=ax_pos, label=field_label, shrink=0.8)

    im_vel = ax_vel.imshow(velocity_pred[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    vel_title = ax_vel.set_title('VELOCITY', fontsize=12, fontweight='bold', color='blue')
    ax_vel.set_xticks([]); ax_vel.set_yticks([])
    plt.colorbar(im_vel, ax=ax_vel, label=field_label, shrink=0.8)

    # Error curves
    line_pos, = ax_err.plot(errors_pos, 'r-', label='POSITIONS', linewidth=2)
    line_vel, = ax_err.plot(errors_vel, 'b-', label='VELOCITY', linewidth=2)
    marker_pos, = ax_err.plot([0], [errors_pos[0]], 'ro', markersize=10)
    marker_vel, = ax_err.plot([0], [errors_vel[0]], 'bo', markersize=10)
    ax_err.set_xlabel('Forecast Step', fontsize=11)
    ax_err.set_ylabel('RMSE', fontsize=11)
    ax_err.set_title('Error Evolution', fontsize=12)
    ax_err.legend(loc='upper left', fontsize=10)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_xlim(0, T-1)
    ax_err.set_ylim(0, max(errors_pos.max(), errors_vel.max()) * 1.1)

    # Error difference map
    if show_error_diff and ax_diff is not None:
        err_pos_0 = np.abs(positions_pred[0] - true_grid[0])
        err_vel_0 = np.abs(velocity_pred[0] - true_grid[0])
        diff_0 = err_pos_0 - err_vel_0  # Positive = velocity is better
        im_diff = ax_diff.imshow(diff_0, cmap='RdBu_r', vmin=-err_diff_max*0.5,
                                  vmax=err_diff_max*0.5, aspect='auto')
        ax_diff.set_title('Error Diff (blue=VELOCITY better)', fontsize=11)
        ax_diff.set_xticks([]); ax_diff.set_yticks([])
        plt.colorbar(im_diff, ax=ax_diff, shrink=0.8)
    else:
        im_diff = None

    # Text annotations
    txt_time = fig.suptitle(f'{title} | t=0/{T-1}', fontsize=14, fontweight='bold')

    def update(t):
        # Update heatmaps
        im_true.set_array(true_grid[t])
        im_pos.set_array(positions_pred[t])
        im_vel.set_array(velocity_pred[t])

        # Update titles with current RMSE
        rmse_pos = errors_pos[t]
        rmse_vel = errors_vel[t]
        ax_pos.set_title(f'POSITIONS (RMSE={rmse_pos:.2f})', fontsize=12, fontweight='bold', color='red')
        ax_vel.set_title(f'VELOCITY (RMSE={rmse_vel:.2f})', fontsize=12, fontweight='bold', color='blue')

        # Update error markers
        marker_pos.set_data([t], [errors_pos[t]])
        marker_vel.set_data([t], [errors_vel[t]])

        # Update error diff map
        if show_error_diff and im_diff is not None:
            err_pos_t = np.abs(positions_pred[t] - true_grid[t])
            err_vel_t = np.abs(velocity_pred[t] - true_grid[t])
            diff_t = err_pos_t - err_vel_t
            im_diff.set_array(diff_t)

        # Update title
        winner = "VELOCITY" if rmse_vel < rmse_pos else "POSITIONS"
        margin = abs(rmse_pos - rmse_vel) / max(rmse_pos, rmse_vel) * 100
        txt_time.set_text(f'{title} | t={t}/{T-1} | {winner} leads by {margin:.1f}%')

        return [im_true, im_pos, im_vel, marker_pos, marker_vel, txt_time] + ([im_diff] if im_diff else [])

    # Compute fps
    fps_out = _fps_for_duration(T, fps, target_duration_sec)
    print(f"[grid video] T={T}, grid={H_grid}x{W_grid}, fps={fps_out}")
    print(f"[grid video] POSITIONS mean RMSE: {errors_pos.mean():.4f}")
    print(f"[grid video] VELOCITY mean RMSE:  {errors_vel.mean():.4f}")
    print(f"[grid video] VELOCITY wins by {(1 - errors_vel.mean()/errors_pos.mean())*100:.1f}%")

    ani = FuncAnimation(fig, update, frames=T, interval=int(1000 / max(fps_out, 1)), blit=False)
    _save_anim(fig, ani, out_path, fps=fps_out)
    plt.close(fig)

    return {
        'errors_vel': errors_vel,
        'errors_pos': errors_pos,
        'mean_rmse_vel': float(errors_vel.mean()),
        'mean_rmse_pos': float(errors_pos.mean()),
        'velocity_improvement_pct': float((1 - errors_vel.mean()/errors_pos.mean()) * 100)
    }


def make_wind_comparison_video(
    true_u: np.ndarray,
    true_v: np.ndarray,
    vel_pred_u: np.ndarray,
    vel_pred_v: np.ndarray,
    pos_pred_u: np.ndarray,
    pos_pred_v: np.ndarray,
    out_path: Path,
    title: str = "Wind Forecast Comparison",
    fps: int = 10,
    target_duration_sec: float = None,
):
    """
    Specialized wind field comparison showing wind speed.

    Parameters
    ----------
    true_u, true_v : (T, H, W) arrays
        Ground truth u and v wind components
    vel_pred_u, vel_pred_v : (T, H, W) arrays
        VELOCITY predictions
    pos_pred_u, pos_pred_v : (T, H, W) arrays
        POSITIONS predictions
    out_path : Path
        Output video path
    title : str
        Video title
    """
    # Compute wind speed
    true_speed = np.sqrt(true_u**2 + true_v**2)
    vel_speed = np.sqrt(vel_pred_u**2 + vel_pred_v**2)
    pos_speed = np.sqrt(pos_pred_u**2 + pos_pred_v**2)

    return make_grid_comparison_video(
        true_grid=true_speed,
        velocity_pred=vel_speed,
        positions_pred=pos_speed,
        out_path=out_path,
        title=title,
        cmap='viridis',
        fps=fps,
        target_duration_sec=target_duration_sec,
        show_error_diff=True,
        field_label='Wind speed (m/s)'
    )



# # #!/usr/bin/env python3
# # """
# # 2×3 RAW vs LOT Wasserstein panel video.

# # Public API
# # ----------
# # - make_2x3_panel_wasserstein_video(
# #       raw_dir, lot_dir, out_path,
# #       D_raw_W2=None, path_raw=None,
# #       D_lot_W2=None, path_lot=None,
# #       eps_raw=0.05, eps_lot=0.05,
# #       w_raw=None, w_lot=None,
# #       fps=20, target_duration_sec=None,
# #       respect_autonomous_window=True,
# #       plot_aligned=True,
# #       cost_label_raw="Wasserstein (Sinkhorn divergence)",
# #       cost_label_lot="Wasserstein (Sinkhorn divergence)",
# #   )
# # """

# # from __future__ import annotations

# # from pathlib import Path
# # import json
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
# # import ot  # POT


# # # ---------- sinkhorn helpers ----------
# # def _ensure64(x): return np.asarray(x, dtype=np.float64)
# # def _pairwise_sqdist(A, B):
# #     A = _ensure64(A); B = _ensure64(B)
# #     diff = A[:, None, :] - B[None, :, :]
# #     return np.einsum("nmd,nmd->nm", diff, diff)

# # def _sinkhorn2_cost(w, M, eps, it=100000, thr=1e-9):
# #     out = ot.sinkhorn2(_ensure64(w), _ensure64(w), _ensure64(M),
# #                        reg=float(eps), numItermax=it, stopThr=thr, verbose=False)
# #     return float(out[0] if isinstance(out, (tuple, list)) else out)

# # def sinkhorn_divergence(Xa, Xb, w, eps):
# #     M_ab = _pairwise_sqdist(Xa, Xb)
# #     M_aa = _pairwise_sqdist(Xa, Xa)
# #     M_bb = _pairwise_sqdist(Xb, Xb)
# #     W_ab = _sinkhorn2_cost(w, M_ab, eps)
# #     W_aa = _sinkhorn2_cost(w, M_aa, eps)
# #     W_bb = _sinkhorn2_cost(w, M_bb, eps)
# #     return W_ab - 0.5 * (W_aa + W_bb)


# # # ---------- tiny utils ----------
# # def _axis_limits_from_sequences(*seqs):
# #     pts = []
# #     for arr in seqs:
# #         if arr is None:
# #             continue
# #         pts.append(arr.reshape(-1, 2))
# #     pts = np.vstack(pts)
# #     xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
# #     dx, dy = xmax - xmin, ymax - ymin
# #     pad = 0.05 * max(dx, dy, 1.0)
# #     cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
# #     L = max(dx, dy) + 2 * pad
# #     return (cx - L/2, cx + L/2, cy - L/2, cy + L/2)

# # def _dtw_hard_path(D: np.ndarray):
# #     H = D.shape[0]
# #     C = np.full((H+1, H+1), np.inf, dtype=np.float64); C[0,0] = 0.0
# #     for i in range(1, H+1):
# #         for j in range(1, H+1):
# #             C[i,j] = D[i-1,j-1] + min(C[i-1,j], C[i,j-1], C[i-1,j-1])
# #     i, j, path = H, H, []
# #     while i > 0 and j > 0:
# #         path.append((i-1, j-1))
# #         up, left, diag = C[i-1,j], C[i,j-1], C[i-1,j-1]
# #         k = int(np.argmin((up, left, diag)))
# #         if k == 0: i -= 1
# #         elif k == 1: j -= 1
# #         else: i -= 1; j -= 1
# #     path.reverse()
# #     return path

# # def _aligned_series_from_path(D, path, H):
# #     # path: list[(t_pred, s_true)]
# #     t2s = {int(t): int(s) for (t, s) in (path or [])}
# #     vals = np.full(H, np.nan); idxs = np.full(H, -1, dtype=int)
# #     if D is None:
# #         return vals, idxs
# #     for t in range(H):
# #         s = t2s.get(t, -1)
# #         if 0 <= s < D.shape[1]:
# #             vals[t] = float(D[t, s]); idxs[t] = s
# #     return vals, idxs

# # def _save_anim(fig, ani, path_base: Path, fps=25, dpi=150):
# #     mp4 = path_base.with_suffix(".mp4")
# #     try:
# #         ani.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
# #         print(f"[video] Wrote {mp4}")
# #     except Exception as e:
# #         gif = path_base.with_suffix(".gif")
# #         print(f"[video] ffmpeg unavailable/failed ({e}); writing GIF -> {gif}")
# #         ani.save(str(gif), writer=PillowWriter(fps=fps), dpi=120)

# # def _fps_for_duration(H, fallback_fps, target_sec):
# #     if target_sec is None or target_sec <= 0:
# #         return max(1, int(fallback_fps))
# #     return max(1, int(round(H / float(target_sec))))


# # # ---------- main builder ----------
# # def make_2x3_panel_wasserstein_video(
# #     raw_dir: Path,
# #     lot_dir: Path,
# #     out_path: Path,
# #     # Optional: provide precomputed Wasserstein Δ and paths
# #     D_raw_W2: np.ndarray = None, path_raw: list = None,
# #     D_lot_W2: np.ndarray = None, path_lot: list = None,
# #     # If Δ not given, compute unaligned Sinkhorn series with these:
# #     eps_raw: float = 0.05, eps_lot: float = 0.05,
# #     w_raw: np.ndarray = None, w_lot: np.ndarray = None,
# #     # Video timing
# #     fps: int = 20,
# #     target_duration_sec: float = None,   # if set, overrides fps to hit ~this length
# #     # Horizon control
# #     respect_autonomous_window: bool = True,  # if False, don't clamp RAW to Hauto
# #     # NEW: display controls
# #     plot_aligned: bool = True,
# #     cost_label_raw: str = "Wasserstein (Sinkhorn divergence)",
# #     cost_label_lot: str = "Wasserstein (Sinkhorn divergence)",
# # ):
# #     raw_dir = Path(raw_dir); lot_dir = Path(lot_dir)
# #     out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

# #     # ---------- RAW I/O ----------
# #     pred_R = np.load(raw_dir / "predicted_positions.npy")  # (H,N,2)
# #     true_R = np.load(raw_dir / "true_segment.npy")         # (H,N,2)
# #     Hraw = min(pred_R.shape[0], true_R.shape[0])

# #     Hauto = None
# #     warm_info = raw_dir / "warm_start_info.npy"
# #     if warm_info.exists():
# #         warm, Hauto_val = np.load(warm_info)
# #         Hauto = int(Hauto_val)
# #         if respect_autonomous_window and Hauto > 0:
# #             Hraw = min(Hraw, Hauto)

# #     pred_R, true_R = pred_R[:Hraw], true_R[:Hraw]
# #     N_R = pred_R.shape[1]
# #     if w_raw is None:
# #         w_raw = np.full(N_R, 1.0 / N_R, dtype=np.float64)

# #     # ---------- LOT I/O (reconstructed measures preferred) ----------
# #     if (lot_dir / "predicted_measures_X.npy").exists() and (lot_dir / "true_measures_X.npy").exists():
# #         pred_L = np.load(lot_dir / "predicted_measures_X.npy")  # (H,R,2)
# #         true_L_for_wass = np.load(lot_dir / "true_measures_X.npy")  # (H,R,2) — for Wasserstein only
# #         Hlot = min(pred_L.shape[0], true_L_for_wass.shape[0])
# #     else:
# #         # fallback to maps (seed + futures): drop seed
# #         predM = np.load(lot_dir / "predicted_maps.npy")
# #         trueM = np.load(lot_dir / "true_future_maps.npy")
# #         Hlot = min(predM.shape[0] - 1, trueM.shape[0] - 1)
# #         pred_L = predM[1:1 + Hlot]
# #         true_L_for_wass = trueM[1:1 + Hlot]
    
# #     # Load fixed reference σ for visualization (LOT True panel shows σ, not T_t(σ))
# #     sigma_path = lot_dir / "reference_sigma_X.npy"
# #     if sigma_path.exists():
# #         sigma_L = np.load(sigma_path)  # (R, 2) — fixed reference
# #     else:
# #         # Fallback: use first frame of true maps
# #         sigma_L = true_L_for_wass[0]

# #     # Final horizon = min across RAW/LOT so animation aligns
# #     H = min(Hraw, Hlot)
# #     pred_R, true_R = pred_R[:H], true_R[:H]
# #     pred_L = pred_L[:H]
# #     true_L_for_wass = true_L_for_wass[:H]
# #     R_L = pred_L.shape[1]
# #     if w_lot is None:
# #         w_lot = np.full(R_L, 1.0 / R_L, dtype=np.float64)

# #     # ---------- Wasserstein series & (optional) alignment ----------
# #     # If full Δ given, use its diag; aligned series only if plot_aligned is True.
# #     diag_R = np.zeros(H); diag_L = np.zeros(H)

# #     # Auto-compute DTW path if plotting aligned and Δ provided but no path
# #     if plot_aligned and D_raw_W2 is not None and path_raw is None:
# #         path_raw = _dtw_hard_path(_ensure64(D_raw_W2))
# #     if plot_aligned and D_lot_W2 is not None and path_lot is None:
# #         path_lot = _dtw_hard_path(_ensure64(D_lot_W2))

# #     # Build aligned series (or NaNs if not plotting)
# #     if plot_aligned:
# #         aligned_R, idx_R = _aligned_series_from_path(D_raw_W2, path_raw, H)
# #         aligned_L, idx_L = _aligned_series_from_path(D_lot_W2, path_lot, H)
# #     else:
# #         aligned_R = np.full(H, np.nan); idx_R = np.full(H, -1, dtype=int)
# #         aligned_L = np.full(H, np.nan); idx_L = np.full(H, -1, dtype=int)

# #     if D_raw_W2 is None:
# #         for t in range(H):
# #             diag_R[t] = sinkhorn_divergence(true_R[t], pred_R[t], w_raw, eps_raw)
# #     else:
# #         D_raw_W2 = _ensure64(D_raw_W2)
# #         diag_R = np.diag(D_raw_W2)[:H]

# #     if D_lot_W2 is None:
# #         for t in range(H):
# #             diag_L[t] = sinkhorn_divergence(true_L_for_wass[t], pred_L[t], w_lot, eps_lot)
# #     else:
# #         D_lot_W2 = _ensure64(D_lot_W2)
# #         diag_L = np.diag(D_lot_W2)[:H]

# #     # ---------- axes (shared) ----------
# #     x0, x1, y0, y1 = _axis_limits_from_sequences(true_R, pred_R, sigma_L, pred_L)

# #     # ---------- figure & static artists ----------
# #     fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
# #     ax_r_true, ax_r_pred, ax_r_err = axes[0]
# #     ax_l_true, ax_l_pred, ax_l_err = axes[1]

# #     for ax in (ax_r_true, ax_r_pred, ax_l_true, ax_l_pred):
# #         ax.set_aspect("equal", adjustable="box")
# #         ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=0.3)
# #         ax.set_xlabel("x"); ax.set_ylabel("y")

# #     ax_r_true.set_title("RAW — True")
# #     ax_r_pred.set_title("RAW — Predicted")
# #     ax_l_true.set_title("LOT — Reference σ")
# #     ax_l_pred.set_title("LOT — Predicted (reconstructed)")
# #     ax_r_err.set_title(f"RAW — {cost_label_raw}")
# #     ax_l_err.set_title(f"LOT — {cost_label_lot}")

# #     scat_r_true = ax_r_true.scatter([], [], s=5, alpha=0.7)
# #     scat_r_pred = ax_r_pred.scatter([], [], s=5, alpha=0.7)
# #     # LOT True panel: show fixed σ (static)
# #     scat_l_true = ax_l_true.scatter(sigma_L[:, 0], sigma_L[:, 1], s=5, alpha=0.7)
# #     scat_l_pred = ax_l_pred.scatter([], [], s=5, alpha=0.7)

# #     tgrid = np.arange(H)
# #     (line_rd,) = ax_r_err.plot(tgrid, diag_R, label="unaligned (diag)")
# #     (line_ld,) = ax_l_err.plot(tgrid, diag_L, label="unaligned (diag)")

# #     # Only add aligned curves/markers/labels if requested and we have finite values
# #     if plot_aligned and np.isfinite(aligned_R).any():
# #         (line_ra,) = ax_r_err.plot(tgrid, aligned_R, label="aligned (DTW path)", alpha=0.95)
# #         mrk_ra, = ax_r_err.plot([0], [aligned_R[0] if np.isfinite(aligned_R[0]) else np.nan], "o", ms=6)
# #     else:
# #         line_ra = None; mrk_ra = None

# #     if plot_aligned and np.isfinite(aligned_L).any():
# #         (line_la,) = ax_l_err.plot(tgrid, aligned_L, label="aligned (DTW path)", alpha=0.95)
# #         mrk_la, = ax_l_err.plot([0], [aligned_L[0] if np.isfinite(aligned_L[0]) else np.nan], "o", ms=6)
# #     else:
# #         line_la = None; mrk_la = None

# #     mrk_rd, = ax_r_err.plot([0], [diag_R[0]], "o", ms=6)
# #     mrk_ld, = ax_l_err.plot([0], [diag_L[0]], "o", ms=6)

# #     # share y-lims for comparability
# #     err_pool = [diag_R, diag_L]
# #     if plot_aligned:
# #         if np.isfinite(aligned_R).any(): err_pool.append(aligned_R[np.isfinite(aligned_R)])
# #         if np.isfinite(aligned_L).any(): err_pool.append(aligned_L[np.isfinite(aligned_L)])
# #     err_pool = np.concatenate([e if np.ndim(e) else np.array([e]) for e in err_pool if np.size(e) > 0])
# #     ymin = float(np.nanmin(err_pool)); ymax = float(np.nanmax(err_pool))
# #     pad = 0.05 * max(ymax - ymin, 1e-12)
# #     for ax in (ax_r_err, ax_l_err):
# #         ax.set_ylim(ymin - pad, ymax + pad)
# #         ax.set_xlabel("prediction time index t")
# #         ax.set_ylabel("cost")
# #         ax.grid(alpha=0.3)
# #         ax.legend()

# #     # dynamic readouts
# #     txt_r = ax_r_err.text(0.98, 0.90, "", transform=ax_r_err.transAxes, ha="right", va="top", fontsize=9)
# #     txt_l = ax_l_err.text(0.98, 0.90, "", transform=ax_l_err.transAxes, ha="right", va="top", fontsize=9)

# #     # ---------- animation update ----------
# #     def update(t):
# #         # RAW scatters (both update)
# #         scat_r_true.set_offsets(true_R[t])
# #         scat_r_pred.set_offsets(pred_R[t])
        
# #         # LOT scatters: true is static (σ), only pred updates
# #         # scat_l_true already shows σ (no update needed)
# #         scat_l_pred.set_offsets(pred_L[t])

# #         # move markers for unaligned
# #         mrk_rd.set_data([t], [diag_R[t]])
# #         mrk_ld.set_data([t], [diag_L[t]])

# #         # optional aligned markers + s* readouts
# #         if plot_aligned:
# #             if mrk_ra is not None:
# #                 vRa = aligned_R[t]
# #                 mrk_ra.set_data([t], [vRa] if np.isfinite(vRa) else [np.nan])
# #             if mrk_la is not None:
# #                 vLa = aligned_L[t]
# #                 mrk_la.set_data([t], [vLa] if np.isfinite(vLa) else [np.nan])

# #             sR = int(idx_R[t]) if (idx_R.size and idx_R[t] >= 0) else -1
# #             sL = int(idx_L[t]) if (idx_L.size and idx_L[t] >= 0) else -1
# #             ax_r_err.set_xlabel(f"prediction time index t (RAW): t={t}, s*={sR}")
# #             ax_l_err.set_xlabel(f"prediction time index t (LOT): t={t}, s*={sL}")
# #             txt_r.set_text(
# #                 f"diag={diag_R[t]:.3g}" + (f"  aligned={aligned_R[t]:.3g}" if np.isfinite(aligned_R[t]) else "")
# #             )
# #             txt_l.set_text(
# #                 f"diag={diag_L[t]:.3g}" + (f"  aligned={aligned_L[t]:.3g}" if np.isfinite(aligned_L[t]) else "")
# #             )
# #         else:
# #             # cleaner labels when not plotting aligned
# #             ax_r_err.set_xlabel(f"prediction time index t (RAW): t={t}")
# #             ax_l_err.set_xlabel(f"prediction time index t (LOT): t={t}")
# #             txt_r.set_text(f"diag={diag_R[t]:.3g}")
# #             txt_l.set_text(f"diag={diag_L[t]:.3g}")

# #         return (scat_r_true, scat_r_pred, scat_l_pred,
# #                 mrk_rd, mrk_ld) + ((mrk_ra, mrk_la, txt_r, txt_l) if plot_aligned else (txt_r, txt_l))

# #     # compute output fps (optionally enforce a target duration)
# #     fps_out = _fps_for_duration(H, fps, target_duration_sec)
# #     print("[video] RAW frames:", Hraw, "| LOT frames:", Hlot,
# #           "| using H:", H,
# #           f"| Hauto:{Hauto if Hauto is not None else 'n/a'}",
# #           f"| fps_out:{fps_out} ⇒ duration≈{H/fps_out:.2f}s",
# #           f"| aligned={'on' if plot_aligned else 'off'}")

# #     ani = FuncAnimation(fig, update, frames=H, interval=int(1000 / max(fps_out, 1)), blit=False)
# #     _save_anim(fig, ani, out_path, fps=fps_out)
# #     plt.close(fig)



# #!/usr/bin/env python3
# """
# 2×3 RAW vs LOT Wasserstein panel video.

# Public API
# ----------
# - make_2x3_panel_wasserstein_video(
#       raw_dir, lot_dir, out_path,
#       D_raw_W2=None, path_raw=None,
#       D_lot_W2=None, path_lot=None,
#       eps_raw=0.05, eps_lot=0.05,
#       w_raw=None, w_lot=None,
#       fps=20, target_duration_sec=None,
#       respect_autonomous_window=True,
#       plot_aligned=True,                      # <— NEW
#       cost_label_raw="Wasserstein (Sinkhorn divergence)",
#       cost_label_lot="Wasserstein (Sinkhorn divergence)",
#   )
# """

# from __future__ import annotations

# from pathlib import Path
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
# import ot  # POT


# # ---------- sinkhorn helpers ----------
# def _ensure64(x): return np.asarray(x, dtype=np.float64)
# def _pairwise_sqdist(A, B):
#     A = _ensure64(A); B = _ensure64(B)
#     diff = A[:, None, :] - B[None, :, :]
#     return np.einsum("nmd,nmd->nm", diff, diff)

# def _sinkhorn2_cost(w, M, eps, it=100000, thr=1e-9):
#     out = ot.sinkhorn2(_ensure64(w), _ensure64(w), _ensure64(M),
#                        reg=float(eps), numItermax=it, stopThr=thr, verbose=False)
#     return float(out[0] if isinstance(out, (tuple, list)) else out)

# def sinkhorn_divergence(Xa, Xb, w, eps):
#     M_ab = _pairwise_sqdist(Xa, Xb)
#     M_aa = _pairwise_sqdist(Xa, Xa)
#     M_bb = _pairwise_sqdist(Xb, Xb)
#     W_ab = _sinkhorn2_cost(w, M_ab, eps)
#     W_aa = _sinkhorn2_cost(w, M_aa, eps)
#     W_bb = _sinkhorn2_cost(w, M_bb, eps)
#     return W_ab - 0.5 * (W_aa + W_bb)


# # ---------- tiny utils ----------
# def _axis_limits_from_sequences(*seqs):
#     pts = []
#     for arr in seqs:
#         if arr is None:
#             continue
#         pts.append(arr.reshape(-1, 2))
#     pts = np.vstack(pts)
#     xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
#     dx, dy = xmax - xmin, ymax - ymin
#     pad = 0.05 * max(dx, dy, 1.0)
#     cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
#     L = max(dx, dy) + 2 * pad
#     return (cx - L/2, cx + L/2, cy - L/2, cy + L/2)

# def _dtw_hard_path(D: np.ndarray):
#     H = D.shape[0]
#     C = np.full((H+1, H+1), np.inf, dtype=np.float64); C[0,0] = 0.0
#     for i in range(1, H+1):
#         for j in range(1, H+1):
#             C[i,j] = D[i-1,j-1] + min(C[i-1,j], C[i,j-1], C[i-1,j-1])
#     i, j, path = H, H, []
#     while i > 0 and j > 0:
#         path.append((i-1, j-1))
#         up, left, diag = C[i-1,j], C[i,j-1], C[i-1,j-1]
#         k = int(np.argmin((up, left, diag)))
#         if k == 0: i -= 1
#         elif k == 1: j -= 1
#         else: i -= 1; j -= 1
#     path.reverse()
#     return path

# def _aligned_series_from_path(D, path, H):
#     # path: list[(t_pred, s_true)]
#     t2s = {int(t): int(s) for (t, s) in (path or [])}
#     vals = np.full(H, np.nan); idxs = np.full(H, -1, dtype=int)
#     if D is None:
#         return vals, idxs
#     for t in range(H):
#         s = t2s.get(t, -1)
#         if 0 <= s < D.shape[1]:
#             vals[t] = float(D[t, s]); idxs[t] = s
#     return vals, idxs

# def _save_anim(fig, ani, path_base: Path, fps=25, dpi=150):
#     mp4 = path_base.with_suffix(".mp4")
#     try:
#         ani.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
#         print(f"[video] Wrote {mp4}")
#     except Exception as e:
#         gif = path_base.with_suffix(".gif")
#         print(f"[video] ffmpeg unavailable/failed ({e}); writing GIF -> {gif}")
#         ani.save(str(gif), writer=PillowWriter(fps=fps), dpi=120)

# def _fps_for_duration(H, fallback_fps, target_sec):
#     if target_sec is None or target_sec <= 0:
#         return max(1, int(fallback_fps))
#     return max(1, int(round(H / float(target_sec))))


# # ---------- main builder ----------
# def make_2x3_panel_wasserstein_video(
#     raw_dir: Path,
#     lot_dir: Path,
#     out_path: Path,
#     # Optional: provide precomputed Wasserstein Δ and paths
#     D_raw_W2: np.ndarray = None, path_raw: list = None,
#     D_lot_W2: np.ndarray = None, path_lot: list = None,
#     # If Δ not given, compute unaligned Sinkhorn series with these:
#     eps_raw: float = 0.05, eps_lot: float = 0.05,
#     w_raw: np.ndarray = None, w_lot: np.ndarray = None,
#     # Video timing
#     fps: int = 20,
#     target_duration_sec: float = None,   # if set, overrides fps to hit ~this length
#     # Horizon control
#     respect_autonomous_window: bool = True,  # if False, don't clamp RAW to Hauto
#     # NEW: display controls
#     plot_aligned: bool = True,
#     cost_label_raw: str = "Wasserstein (Sinkhorn divergence)",
#     cost_label_lot: str = "Wasserstein (Sinkhorn divergence)",
# ):
#     raw_dir = Path(raw_dir); lot_dir = Path(lot_dir)
#     out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

#     # ---------- RAW I/O ----------
#     pred_R = np.load(raw_dir / "predicted_positions.npy")  # (H,N,2)
#     true_R = np.load(raw_dir / "true_segment.npy")         # (H,N,2)
#     Hraw = min(pred_R.shape[0], true_R.shape[0])

#     Hauto = None
#     warm_info = raw_dir / "warm_start_info.npy"
#     if warm_info.exists():
#         warm, Hauto_val = np.load(warm_info)
#         Hauto = int(Hauto_val)
#         if respect_autonomous_window and Hauto > 0:
#             Hraw = min(Hraw, Hauto)

#     pred_R, true_R = pred_R[:Hraw], true_R[:Hraw]
#     N_R = pred_R.shape[1]
#     if w_raw is None:
#         w_raw = np.full(N_R, 1.0 / N_R, dtype=np.float64)

#     # ---------- LOT I/O (reconstructed measures preferred) ----------
#     if (lot_dir / "predicted_measures_X.npy").exists() and (lot_dir / "true_measures_X.npy").exists():
#         pred_L = np.load(lot_dir / "predicted_measures_X.npy")  # (H,R,2)
#         true_L = np.load(lot_dir / "true_measures_X.npy")       # (H,R,2)
#         Hlot = min(pred_L.shape[0], true_L.shape[0])
#     else:
#         # fallback to maps (seed + futures): drop seed
#         predM = np.load(lot_dir / "predicted_maps.npy")
#         trueM = np.load(lot_dir / "true_future_maps.npy")
#         Hlot = min(predM.shape[0] - 1, trueM.shape[0] - 1)
#         pred_L = predM[1:1 + Hlot]
#         true_L = trueM[1:1 + Hlot]

#     # Final horizon = min across RAW/LOT so animation aligns
#     H = min(Hraw, Hlot)
#     pred_R, true_R = pred_R[:H], true_R[:H]
#     pred_L, true_L = pred_L[:H], true_L[:H]
#     R_L = pred_L.shape[1]
#     if w_lot is None:
#         w_lot = np.full(R_L, 1.0 / R_L, dtype=np.float64)

#     # ---------- Wasserstein series & (optional) alignment ----------
#     # If full Δ given, use its diag; aligned series only if plot_aligned is True.
#     diag_R = np.zeros(H); diag_L = np.zeros(H)

#     # Auto-compute DTW path if plotting aligned and Δ provided but no path
#     if plot_aligned and D_raw_W2 is not None and path_raw is None:
#         path_raw = _dtw_hard_path(_ensure64(D_raw_W2))
#     if plot_aligned and D_lot_W2 is not None and path_lot is None:
#         path_lot = _dtw_hard_path(_ensure64(D_lot_W2))

#     # Build aligned series (or NaNs if not plotting)
#     if plot_aligned:
#         aligned_R, idx_R = _aligned_series_from_path(D_raw_W2, path_raw, H)
#         aligned_L, idx_L = _aligned_series_from_path(D_lot_W2, path_lot, H)
#     else:
#         aligned_R = np.full(H, np.nan); idx_R = np.full(H, -1, dtype=int)
#         aligned_L = np.full(H, np.nan); idx_L = np.full(H, -1, dtype=int)

#     if D_raw_W2 is None:
#         for t in range(H):
#             diag_R[t] = sinkhorn_divergence(true_R[t], pred_R[t], w_raw, eps_raw)
#     else:
#         D_raw_W2 = _ensure64(D_raw_W2)
#         diag_R = np.diag(D_raw_W2)[:H]

#     if D_lot_W2 is None:
#         for t in range(H):
#             diag_L[t] = sinkhorn_divergence(true_L[t], pred_L[t], w_lot, eps_lot)
#     else:
#         D_lot_W2 = _ensure64(D_lot_W2)
#         diag_L = np.diag(D_lot_W2)[:H]

#     # ---------- axes (shared) ----------
#     x0, x1, y0, y1 = _axis_limits_from_sequences(true_R, pred_R, true_L, pred_L)

#     # ---------- figure & static artists ----------
#     fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
#     ax_r_true, ax_r_pred, ax_r_err = axes[0]
#     ax_l_true, ax_l_pred, ax_l_err = axes[1]

#     for ax in (ax_r_true, ax_r_pred, ax_l_true, ax_l_pred):
#         ax.set_aspect("equal", adjustable="box")
#         ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=0.3)
#         ax.set_xlabel("x"); ax.set_ylabel("y")

#     ax_r_true.set_title("RAW — True")
#     ax_r_pred.set_title("RAW — Predicted")
#     ax_l_true.set_title("LOT — True (reconstructed)")
#     ax_l_pred.set_title("LOT — Predicted (reconstructed)")
#     ax_r_err.set_title(f"RAW — {cost_label_raw}")
#     ax_l_err.set_title(f"LOT — {cost_label_lot}")

#     scat_r_true = ax_r_true.scatter([], [], s=5, alpha=0.7)
#     scat_r_pred = ax_r_pred.scatter([], [], s=5, alpha=0.7)
#     scat_l_true = ax_l_true.scatter([], [], s=5, alpha=0.7)
#     scat_l_pred = ax_l_pred.scatter([], [], s=5, alpha=0.7)

#     tgrid = np.arange(H)
#     (line_rd,) = ax_r_err.plot(tgrid, diag_R, label="unaligned (diag)")
#     (line_ld,) = ax_l_err.plot(tgrid, diag_L, label="unaligned (diag)")

#     # Only add aligned curves/markers/labels if requested and we have finite values
#     if plot_aligned and np.isfinite(aligned_R).any():
#         (line_ra,) = ax_r_err.plot(tgrid, aligned_R, label="aligned (DTW path)", alpha=0.95)
#         mrk_ra, = ax_r_err.plot([0], [aligned_R[0] if np.isfinite(aligned_R[0]) else np.nan], "o", ms=6)
#     else:
#         line_ra = None; mrk_ra = None

#     if plot_aligned and np.isfinite(aligned_L).any():
#         (line_la,) = ax_l_err.plot(tgrid, aligned_L, label="aligned (DTW path)", alpha=0.95)
#         mrk_la, = ax_l_err.plot([0], [aligned_L[0] if np.isfinite(aligned_L[0]) else np.nan], "o", ms=6)
#     else:
#         line_la = None; mrk_la = None

#     mrk_rd, = ax_r_err.plot([0], [diag_R[0]], "o", ms=6)
#     mrk_ld, = ax_l_err.plot([0], [diag_L[0]], "o", ms=6)

#     # share y-lims for comparability
#     err_pool = [diag_R, diag_L]
#     if plot_aligned:
#         if np.isfinite(aligned_R).any(): err_pool.append(aligned_R[np.isfinite(aligned_R)])
#         if np.isfinite(aligned_L).any(): err_pool.append(aligned_L[np.isfinite(aligned_L)])
#     err_pool = np.concatenate([e if np.ndim(e) else np.array([e]) for e in err_pool if np.size(e) > 0])
#     ymin = float(np.nanmin(err_pool)); ymax = float(np.nanmax(err_pool))
#     pad = 0.05 * max(ymax - ymin, 1e-12)
#     for ax in (ax_r_err, ax_l_err):
#         ax.set_ylim(ymin - pad, ymax + pad)
#         ax.set_xlabel("prediction time index t")
#         ax.set_ylabel("cost")
#         ax.grid(alpha=0.3)
#         ax.legend()

#     # dynamic readouts
#     txt_r = ax_r_err.text(0.98, 0.90, "", transform=ax_r_err.transAxes, ha="right", va="top", fontsize=9)
#     txt_l = ax_l_err.text(0.98, 0.90, "", transform=ax_l_err.transAxes, ha="right", va="top", fontsize=9)

#     # ---------- animation update ----------
#     def update(t):
#         # RAW/LOT scatters
#         scat_r_true.set_offsets(true_R[t]); scat_r_pred.set_offsets(pred_R[t])
#         scat_l_true.set_offsets(true_L[t]); scat_l_pred.set_offsets(pred_L[t])

#         # move markers for unaligned
#         mrk_rd.set_data([t], [diag_R[t]])
#         mrk_ld.set_data([t], [diag_L[t]])

#         # optional aligned markers + s* readouts
#         if plot_aligned:
#             if mrk_ra is not None:
#                 vRa = aligned_R[t]
#                 mrk_ra.set_data([t], [vRa] if np.isfinite(vRa) else [np.nan])
#             if mrk_la is not None:
#                 vLa = aligned_L[t]
#                 mrk_la.set_data([t], [vLa] if np.isfinite(vLa) else [np.nan])

#             sR = int(idx_R[t]) if (idx_R.size and idx_R[t] >= 0) else -1
#             sL = int(idx_L[t]) if (idx_L.size and idx_L[t] >= 0) else -1
#             ax_r_err.set_xlabel(f"prediction time index t (RAW): t={t}, s*={sR}")
#             ax_l_err.set_xlabel(f"prediction time index t (LOT): t={t}, s*={sL}")
#             txt_r.set_text(
#                 f"diag={diag_R[t]:.3g}" + (f"  aligned={aligned_R[t]:.3g}" if np.isfinite(aligned_R[t]) else "")
#             )
#             txt_l.set_text(
#                 f"diag={diag_L[t]:.3g}" + (f"  aligned={aligned_L[t]:.3g}" if np.isfinite(aligned_L[t]) else "")
#             )
#         else:
#             # cleaner labels when not plotting aligned
#             ax_r_err.set_xlabel(f"prediction time index t (RAW): t={t}")
#             ax_l_err.set_xlabel(f"prediction time index t (LOT): t={t}")
#             txt_r.set_text(f"diag={diag_R[t]:.3g}")
#             txt_l.set_text(f"diag={diag_L[t]:.3g}")

#         return (scat_r_true, scat_r_pred, scat_l_true, scat_l_pred,
#                 mrk_rd, mrk_ld) + ((mrk_ra, mrk_la, txt_r, txt_l) if plot_aligned else (txt_r, txt_l))

#     # compute output fps (optionally enforce a target duration)
#     fps_out = _fps_for_duration(H, fps, target_duration_sec)
#     print("[video] RAW frames:", Hraw, "| LOT frames:", Hlot,
#           "| using H:", H,
#           f"| Hauto:{Hauto if Hauto is not None else 'n/a'}",
#           f"| fps_out:{fps_out} ⇒ duration≈{H/fps_out:.2f}s",
#           f"| aligned={'on' if plot_aligned else 'off'}")

#     ani = FuncAnimation(fig, update, frames=H, interval=int(1000 / max(fps_out, 1)), blit=False)
#     _save_anim(fig, ani, out_path, fps=fps_out)
#     plt.close(fig)
