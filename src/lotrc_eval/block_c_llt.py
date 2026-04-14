#!/usr/bin/env python3
"""
Block C — LLT@σ + L²(σ): build time–time cost matrix Δ from LOT barycentric embeddings.

Compatible with velocity_forecast.py and positions_forecast.py outputs.

Public API
----------
- build_embeddings_llt(run: RunData, epsilon: float) -> (phi_true, phi_pred, w)
- build_delta_llt_l2(run: RunData, epsilon: float | None = None, squared: bool = True)
    -> (Delta: np.ndarray, aux: dict)

Notes
-----
- If run.true_disp / run.pred_disp are present (saved by forecast scripts), those
  are used directly as embeddings (φ_true, φ_pred). This is the expected case.
- Otherwise, we compute φ_t = T_t(σ) - σ via Sinkhorn barycentric projection.
- L²(σ) cost:
    squared=True  : Δ_{t,s} = Σ_i w_i || φ_pred[t,i] - φ_true[s,i] ||²
    squared=False : Δ_{t,s} = sqrt( Σ_i w_i || φ_pred[t,i] - φ_true[s,i] ||² )

Important: Both VELOCITY and POSITIONS methods output the same displacement format,
so this block treats them uniformly.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import numpy as np

try:
    import ot  # POT
    HAS_OT = True
except ImportError:
    HAS_OT = False

from .block_b_io import RunData, pick_epsilon_from_sigma


# ═══════════════════════════════════════════════════════════════
# NUMERIC HELPERS
# ═══════════════════════════════════════════════════════════════

def _ensure_float64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64, copy=False)


def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return matrix of squared Euclidean distances:
        M_{ij} = ||A_i - B_j||²
    A: (n, d), B: (m, d) -> M: (n, m)
    """
    A = _ensure_float64(A)
    B = _ensure_float64(B)
    diff = A[:, None, :] - B[None, :, :]
    return np.einsum("nmd,nmd->nm", diff, diff)


def _sinkhorn_plan(
    w_src: np.ndarray,
    w_tgt: np.ndarray,
    M_sq: np.ndarray,
    epsilon: float,
    num_iter: int = 10_000,
    stop_thr: float = 1e-9,
) -> np.ndarray:
    """
    Entropic OT plan via POT's sinkhorn using *squared* ground cost matrix M_sq.
    Returns γ ∈ R^{N×N} (float64).
    """
    if not HAS_OT:
        raise ImportError("POT library required for Sinkhorn computation")
    
    w_src = _ensure_float64(w_src)
    w_tgt = _ensure_float64(w_tgt)
    M_sq = _ensure_float64(M_sq)
    
    gamma = ot.sinkhorn(
        w_src, w_tgt, M_sq,
        reg=epsilon,
        numItermax=num_iter,
        stopThr=stop_thr,
        verbose=False,
    )
    return _ensure_float64(gamma)


# ═══════════════════════════════════════════════════════════════
# LOT BARYCENTRIC EMBEDDING
# ═══════════════════════════════════════════════════════════════

def _barycentric_embedding_sigma_to(
    target_X: np.ndarray,
    sigma: np.ndarray,
    w_sigma: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Compute the LOT (LLT) embedding φ(μ) of a measure μ with support target_X
    on the reference σ with weights w_sigma via Sinkhorn barycentric projection:

        T(σ_i) = sum_j γ_{ij} x_j / sum_j γ_{ij},   φ = T - σ

    Inputs
    ------
    target_X : (N, d)   support points of μ (same N,d as σ)
    sigma    : (N, d)   reference support
    w_sigma  : (N,)     reference weights (sum=1)
    epsilon  : float    entropic regularization

    Returns
    -------
    phi : (N, d) LOT displacement field on σ
    """
    M_sq = _pairwise_sqdist(sigma, target_X)  # (N, N)
    P = _sinkhorn_plan(w_sigma, w_sigma, M_sq, epsilon)  # (N, N)
    rs = P.sum(axis=1, keepdims=True) + 1e-18
    T = (P @ _ensure_float64(target_X)) / rs  # (N, d)
    phi = T - _ensure_float64(sigma)
    return phi


def build_embeddings_llt(
    run: RunData,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LOT@σ embeddings for a run.

    If run.true_disp / run.pred_disp exist (from forecast scripts), they are used.
    Otherwise, compute φ_true[t], φ_pred[t] via barycentric projection for each t.

    Parameters
    ----------
    run : RunData
    epsilon : float

    Returns
    -------
    phi_true : (H, N, d)
    phi_pred : (H, N, d)
    w        : (N,) normalized weights on σ
    """
    H, N, d = run.H, run.N, run.d
    
    if H < 2:
        raise ValueError(f"[LLT] H={H} for {run.name}. Need H≥2.")

    # Normalize weights defensively
    w = _ensure_float64(run.w)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"[LLT] Invalid weights in {run.name} (sum={s}).")
    if abs(s - 1.0) > 1e-12:
        w = w / s

    # TRUE embeddings
    if run.true_disp is not None:
        phi_true = _ensure_float64(run.true_disp)
        print(f"[LLT] {run.name}: using pre-computed true_disp")
    else:
        print(f"[LLT] {run.name}: computing true embeddings via Sinkhorn (eps={epsilon:.3g})")
        phi_true = np.empty((H, N, d), dtype=np.float64)
        for t in range(H):
            phi_true[t] = _barycentric_embedding_sigma_to(
                run.true_X[t], run.sigma, w, epsilon
            )

    # PRED embeddings
    if run.pred_disp is not None:
        phi_pred = _ensure_float64(run.pred_disp)
        print(f"[LLT] {run.name}: using pre-computed pred_disp")
    else:
        print(f"[LLT] {run.name}: computing pred embeddings via Sinkhorn (eps={epsilon:.3g})")
        phi_pred = np.empty((H, N, d), dtype=np.float64)
        for t in range(H):
            phi_pred[t] = _barycentric_embedding_sigma_to(
                run.pred_X[t], run.sigma, w, epsilon
            )

    return phi_true, phi_pred, w


# ═══════════════════════════════════════════════════════════════
# DELTA BUILDER: L²(σ) OVER LOT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════

def build_delta_llt_l2(
    run: RunData,
    epsilon: Optional[float] = None,
    squared: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Build Δ on σ using LOT embeddings and L²(σ) geometry.

    Δ_{t,s} = || φ_pred[t,·] - φ_true[s,·] ||_{L²(σ)}²    (if squared=True)
            = sqrt( Σ_i w_i ||·||² )                      (if squared=False)

    Parameters
    ----------
    run : RunData
    epsilon : float | None
        If None, uses pick_epsilon_from_sigma(run.sigma, 0.05).
    squared : bool
        Whether to return squared L² or actual L².

    Returns
    -------
    Delta : (H, H) ndarray
    aux   : dict with keys:
            - 'phi_true' : (H,N,d)
            - 'phi_pred' : (H,N,d)
            - 'epsilon'  : float
            - 'weights'  : (N,)
    """
    if epsilon is None:
        epsilon = pick_epsilon_from_sigma(run.sigma, eps_scale=0.05)

    phi_true, phi_pred, w = build_embeddings_llt(run, float(epsilon))
    H, N, d = phi_true.shape

    Delta = np.empty((H, H), dtype=np.float64)

    if squared:
        # Δ_{t,s} = Σ_i w_i ||φ_pred[t,i] - φ_true[s,i]||²
        for t in range(H):
            # (H, N): sum over d then weight over i
            diff2_sum = ((phi_true - phi_pred[t]) ** 2).sum(axis=2)
            Delta[t] = (diff2_sum * w[None, :]).sum(axis=1)
    else:
        # ||·||_{L²(σ)} = sqrt( Σ_i w_i ||·||² )
        for t in range(H):
            diff2_sum = ((phi_true - phi_pred[t]) ** 2).sum(axis=2)
            Delta[t] = np.sqrt(np.maximum((diff2_sum * w[None, :]).sum(axis=1), 0.0))

    aux = dict(
        phi_true=phi_true,
        phi_pred=phi_pred,
        epsilon=float(epsilon),
        weights=w,
    )
    
    print(f"[LLT] {run.name}: Δ shape={Delta.shape}, median={np.median(Delta):.6g}")
    
    return Delta, aux