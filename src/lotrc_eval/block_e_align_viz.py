#!/usr/bin/env python3
"""
Block E — Alignment & Visualization utilities for LOT–RC evaluation.

Compatible with velocity_forecast.py and positions_forecast.py outputs.

Public API
----------
- soft_dtw_value(D: np.ndarray, gamma: float) -> float
- dtw_hard_path(D: np.ndarray) -> list[tuple[int, int]]
- alignment_metrics(D: np.ndarray, path: list[tuple[int, int]], bins: int = 11) -> dict
- plot_delta_with_path(D, path, title, savepath=None)
- plot_error_curves(D, path, title, savepath=None)
- save_alignment_artifacts(tag, D, out_dir, gamma_base) -> dict

Notes
-----
- soft_dtw_value implements the Cuturi–Blondel recursion (value only; no gradients).
- dtw_hard_path computes a classic (hard) DTW alignment path to visualize on Δ.
- alignment_metrics summarizes unaligned vs aligned costs, improvement, path stats.
- Plot helpers save compact, readable figures suitable for quick inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# SOFT-DTW VALUE
# ═══════════════════════════════════════════════════════════════

def soft_dtw_value(D: np.ndarray, gamma: float) -> float:
    """
    Compute soft-DTW value for a square cost matrix D ∈ R^{H×H}, gamma > 0.

    Parameters
    ----------
    D : np.ndarray
        Square cost matrix (H, H).
    gamma : float
        Smoothing (temperature) parameter; larger = smoother/softer.

    Returns
    -------
    float
        The soft-DTW value (scalar).
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"soft_dtw_value: D must be square; got shape {D.shape}")
    if gamma <= 0:
        raise ValueError("soft_dtw_value: gamma must be > 0.")

    H = D.shape[0]
    R = np.full((H + 1, H + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0

    # Stable softmin of (a, b, c)
    def softmin(a: float, b: float, c: float) -> float:
        m = min(a, b, c)
        return -gamma * np.log(
            np.exp((m - a) / gamma) + np.exp((m - b) / gamma) + np.exp((m - c) / gamma)
        ) + m

    for i in range(1, H + 1):
        for j in range(1, H + 1):
            R[i, j] = D[i - 1, j - 1] + softmin(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1])

    return float(R[H, H])


# ═══════════════════════════════════════════════════════════════
# HARD DTW PATH
# ═══════════════════════════════════════════════════════════════

def dtw_hard_path(D: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute a classic (hard) DTW alignment path on square matrix D.

    Parameters
    ----------
    D : np.ndarray
        Square cost matrix (H, H).

    Returns
    -------
    list[(t, s)]
        Path from (0,0) to (H-1, H-1).
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"dtw_hard_path: D must be square; got shape {D.shape}")

    H = D.shape[0]
    C = np.full((H + 1, H + 1), np.inf, dtype=np.float64)
    C[0, 0] = 0.0

    for i in range(1, H + 1):
        for j in range(1, H + 1):
            C[i, j] = D[i - 1, j - 1] + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])

    # Backtrack
    i, j = H, H
    path: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        up, left, diag = C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]
        k = int(np.argmin((up, left, diag)))
        if k == 0:
            i -= 1
        elif k == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return path


# ═══════════════════════════════════════════════════════════════
# ALIGNMENT METRICS
# ═══════════════════════════════════════════════════════════════

def alignment_metrics(
    D: np.ndarray,
    path: List[Tuple[int, int]],
    bins: int = 11,
) -> Dict:
    """
    Summarize unaligned (diag) vs aligned (path) costs and path geometry.

    Parameters
    ----------
    D : np.ndarray
        Square cost matrix (H, H).
    path : list[(t, s)]
        DTW alignment path.
    bins : int
        Number of bins for lead/lag histogram (t - s).

    Returns
    -------
    dict
        {
          'unaligned_mean': float,
          'aligned_mean': float,
          'improve_pct': float,
          'path_len': int,
          'dilation_ratio': float,
          'lead_lag_hist': list[int]
        }
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"alignment_metrics: D must be square; got {D.shape}")

    H = D.shape[0]
    if H == 0:
        raise ValueError("alignment_metrics: empty cost matrix")

    diag_vals = np.diag(D)
    unaligned = float(np.mean(diag_vals))
    
    if path:
        aligned_vals = np.array([D[t, s] for (t, s) in path], dtype=np.float64)
        aligned = float(aligned_vals.mean())
    else:
        aligned = unaligned

    improve = 100.0 * (unaligned - aligned) / max(unaligned, 1e-12)
    
    lead_lag = np.array([t - s for (t, s) in path], dtype=int) if path else np.array([], dtype=int)
    dilation = len(path) / float(H) if H > 0 else float("nan")

    if lead_lag.size > 0:
        hist, _edges = np.histogram(lead_lag, bins=bins)
        hist_list = hist.astype(int).tolist()
    else:
        hist_list = [0] * bins

    return dict(
        unaligned_mean=unaligned,
        aligned_mean=aligned,
        improve_pct=improve,
        path_len=len(path),
        dilation_ratio=dilation,
        lead_lag_hist=hist_list,
    )


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_error_curves(
    D: np.ndarray,
    path: List[Tuple[int, int]],
    title: str,
    savepath: Optional[Path] = None,
):
    """
    Plot unaligned (diag of Δ) vs aligned (values along path) error curves.
    """
    H = D.shape[0]
    diag_err = np.diag(D)
    aligned_err = np.array([D[t, s] for (t, s) in path], dtype=np.float64) if path else np.array([])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(H), diag_err, label="unaligned (diag Δ)")
    if path:
        ts = [t for (t, _s) in path]
        ax.plot(ts, aligned_err, label="aligned (path)")
    ax.set_xlabel("pred index t")
    ax.set_ylabel("cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.close(fig)
    
    return fig, ax


def plot_delta_with_path(
    D: np.ndarray,
    path: List[Tuple[int, int]],
    title: str,
    savepath: Optional[Path] = None,
):
    """
    Heatmap of Δ with the DTW path overlay.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(D, origin="lower", aspect="auto")
    ax.set_xlabel("true index s")
    ax.set_ylabel("pred index t")
    ax.set_title(title)
    
    if path:
        ys = [t for (t, s) in path]
        xs = [s for (t, s) in path]
        ax.plot(xs, ys, linewidth=1.0)
    
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.close(fig)
    
    return fig, ax


# ═══════════════════════════════════════════════════════════════
# SAVE ALIGNMENT ARTIFACTS
# ═══════════════════════════════════════════════════════════════

def save_alignment_artifacts(
    tag: str,
    D: np.ndarray,
    out_dir: Path,
    gamma_base: float,
) -> Dict:
    """
    Save Δ, soft-DTW value, DTW path, metrics, and two plots under out_dir.

    Files written:
      - Delta.npy
      - alignment_metrics.json
      - errors.png
      - delta_with_path.png
      - path.json

    Returns
    -------
    dict
        The metrics dict that was saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "Delta.npy", D)

    # Scale gamma with median(Δ) for unit-awareness
    gamma = gamma_base * max(float(np.median(D)), 1e-12)
    sdtw = soft_dtw_value(D, gamma)
    path = dtw_hard_path(D)
    metr = alignment_metrics(D, path)
    
    payload = {"tag": tag, "gamma": gamma, "soft_dtw_value": sdtw, **metr}
    (out_dir / "alignment_metrics.json").write_text(json.dumps(payload, indent=2))

    plot_error_curves(D, path, f"{tag}: diag vs aligned", out_dir / "errors.png")
    plot_delta_with_path(D, path, f"{tag}: Δ with path", out_dir / "delta_with_path.png")
    (out_dir / "path.json").write_text(json.dumps(path))

    # Console summary
    print(
        f"[align:{tag}] unaligned={metr['unaligned_mean']:.6g} | "
        f"aligned={metr['aligned_mean']:.6g} | improve={metr['improve_pct']:.2f}% | "
        f"gamma={gamma:.3g} | path_len={metr['path_len']}"
    )
    
    return payload