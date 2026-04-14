#!/usr/bin/env python3
"""
Block D — Sinkhorn (debiased entropic W₂²) time–time cost matrix Δ.

Compatible with velocity_forecast.py and positions_forecast.py outputs.

Public API
----------
- build_delta_sinkhorn_window(run: RunData, epsilon: float,
                              center: int | None = None, half_width: int = 6)
    -> (Delta: np.ndarray, aux: dict)
- build_delta_sinkhorn_full(run: RunData, epsilon: float)
    -> (Delta: np.ndarray, aux: dict)

Notes
-----
- We compute the *debiased* Sinkhorn divergence (aka Sinkhorn "divergence"):
      S_ε(α, β) = W_ε(α, β) - 1/2 W_ε(α, α) - 1/2 W_ε(β, β)
  with the *squared* Euclidean ground cost and common weights.
- Shapes:
    run.pred_X[t], run.true_X[s] ∈ R^{N×d} with the same (N, d) and weights w ∈ R^N.
- Complexity:
    Full H×H is O(H² · OT(N,N)) — use the windowed builder for quick validation.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np

try:
    import ot  # POT
    HAS_OT = True
except ImportError:
    HAS_OT = False

from .block_b_io import RunData


# ═══════════════════════════════════════════════════════════════
# NUMERIC HELPERS
# ═══════════════════════════════════════════════════════════════

def _ensure_float64(x: np.ndarray) -> np.ndarray:
    """Return a float64 view (no copy when already float64)."""
    return x.astype(np.float64, copy=False)


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Squared Euclidean distances:
        M_{ij} = ||A_i - B_j||²
    A: (n, d), B: (m, d) -> M: (n, m)
    """
    A = _ensure_float64(A)
    B = _ensure_float64(B)
    diff = A[:, None, :] - B[None, :, :]
    return np.einsum("nmd,nmd->nm", diff, diff)


def _normalized_weights(w: np.ndarray) -> np.ndarray:
    """Return w normalized to sum=1 with defensive checks."""
    w = _ensure_float64(w)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("[Sinkhorn] Invalid weights: non-finite or non-positive sum.")
    if abs(s - 1.0) > 1e-12:
        w = w / s
    return w


# ═══════════════════════════════════════════════════════════════
# CORE SINKHORN
# ═══════════════════════════════════════════════════════════════

def _sinkhorn2_cost(
    w_src: np.ndarray,
    w_tgt: np.ndarray,
    M_sq: np.ndarray,
    epsilon: float,
    num_iter: int = 100_000,
    stop_thr: float = 1e-9,
) -> float:
    """
    POT's entropic OT cost with squared ground cost.
    Robust to POT versions (returns scalar).
    """
    if not HAS_OT:
        raise ImportError("POT library required for Sinkhorn computation")
    
    w_src = _ensure_float64(w_src)
    w_tgt = _ensure_float64(w_tgt)
    M_sq = _ensure_float64(M_sq)
    
    out = ot.sinkhorn2(
        w_src, w_tgt, M_sq,
        reg=epsilon,
        numItermax=num_iter,
        stopThr=stop_thr,
        verbose=False,
    )
    
    if isinstance(out, (tuple, list)):
        cost = out[0]
    else:
        cost = out
    
    return float(cost)


def _debiased_sinkhorn_divergence(
    Xa: np.ndarray,
    Xb: np.ndarray,
    w: np.ndarray,
    epsilon: float,
) -> float:
    """
    Debiased Sinkhorn divergence (squared) between two empirical measures
    with *common* weights w and supports Xa, Xb (N×d).
    
        S_ε(α,β) = W_ε(α,β) - 1/2 W_ε(α,α) - 1/2 W_ε(β,β)
    """
    Xa = _ensure_float64(Xa)
    Xb = _ensure_float64(Xb)
    w = _normalized_weights(w)

    M_ab = _pairwise_sqdist(Xa, Xb)
    M_aa = _pairwise_sqdist(Xa, Xa)
    M_bb = _pairwise_sqdist(Xb, Xb)

    W_ab = _sinkhorn2_cost(w, w, M_ab, epsilon)
    W_aa = _sinkhorn2_cost(w, w, M_aa, epsilon)
    W_bb = _sinkhorn2_cost(w, w, M_bb, epsilon)

    return W_ab - 0.5 * (W_aa + W_bb)


# ═══════════════════════════════════════════════════════════════
# WINDOWING
# ═══════════════════════════════════════════════════════════════

def _pick_window_indices(
    H: int,
    center: Optional[int] = None,
    half_width: int = 6,
) -> np.ndarray:
    """
    Pick a contiguous subwindow of indices in [0, H-1].
    Default (None, 6) → a ~12×12 Δ submatrix centered in the middle.
    """
    if H <= 0:
        return np.zeros((0,), dtype=int)
    if center is None:
        center = H // 2
    lo = max(0, center - half_width)
    hi = min(H, center + half_width)  # exclusive end
    return np.arange(lo, hi, dtype=int)


# ═══════════════════════════════════════════════════════════════
# PUBLIC BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_delta_sinkhorn_window(
    run: RunData,
    epsilon: float,
    center: Optional[int] = None,
    half_width: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """
    Build Δ via debiased Sinkhorn on a *subwindow* of time indices to save compute.

    Δ_{i,j} = S_ε( pred_X[idx[i]], true_X[idx[j]] ), for i,j in the window indices.

    Parameters
    ----------
    run : RunData
    epsilon : float
    center : int | None
        Center index for the window (None → middle of the series).
    half_width : int
        Half-width of the window; final Δ is roughly (2*half_width)².

    Returns
    -------
    Delta : (Hsub, Hsub) ndarray
    aux   : dict with keys {'indices': idx (np.ndarray), 'epsilon': float}
    """
    H = run.H
    if H < 2:
        raise ValueError(f"[Sinkhorn] H={H} for {run.name}. Need H≥2.")

    idx = _pick_window_indices(H, center=center, half_width=half_width)
    Hsub = len(idx)
    
    if Hsub == 0:
        return np.zeros((0, 0), dtype=np.float64), dict(indices=idx, epsilon=float(epsilon))

    D = np.zeros((Hsub, Hsub), dtype=np.float64)
    w = _normalized_weights(run.w)

    print(f"[Sinkhorn] {run.name}: computing {Hsub}×{Hsub} window (center={center}, half_width={half_width})")
    
    for ii, t in enumerate(idx):
        Xa = run.pred_X[t]  # (N, d)
        for jj, s in enumerate(idx):
            Xb = run.true_X[s]
            D[ii, jj] = _debiased_sinkhorn_divergence(Xa, Xb, w, epsilon)

    aux = dict(indices=idx, epsilon=float(epsilon))
    
    print(f"[Sinkhorn] {run.name}: Δ median={np.median(D):.6g}")
    
    return D, aux


def build_delta_sinkhorn_full(
    run: RunData,
    epsilon: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Build Δ via debiased Sinkhorn on the *full* time grid (H×H).

    Δ_{t,s} = S_ε( pred_X[t], true_X[s] )

    Parameters
    ----------
    run : RunData
    epsilon : float

    Returns
    -------
    Delta : (H, H) ndarray
    aux   : dict with key {'epsilon': float}

    Notes
    -----
    Complexity is high for large H. Validate with the window builder first.
    """
    H = run.H
    if H < 2:
        raise ValueError(f"[Sinkhorn] H={H} for {run.name}. Need H≥2.")

    D = np.zeros((H, H), dtype=np.float64)
    w = _normalized_weights(run.w)

    print(f"[Sinkhorn] {run.name}: computing full {H}×{H} matrix (this may take a while)")
    
    for t in range(H):
        Xa = run.pred_X[t]
        for s in range(H):
            Xb = run.true_X[s]
            D[t, s] = _debiased_sinkhorn_divergence(Xa, Xb, w, epsilon)
        
        if (t + 1) % 10 == 0:
            print(f"[Sinkhorn] {run.name}: completed row {t+1}/{H}")

    print(f"[Sinkhorn] {run.name}: Δ median={np.median(D):.6g}")
    
    return D, dict(epsilon=float(epsilon))