"""
GPU-friendly (and CPU) Sinkhorn optimal transport for uniform marginals.

Used for LOT barycentric maps: plan P ≈ argmin <P,C> + ε KL(P||a⊗b).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch


def sinkhorn_transport_plan(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    num_iter: int = 200,
    tol: float = 1e-9,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for balanced OT with Gibbs kernel K = exp(-C/reg).

    Args:
        cost: (..., R, N) nonnegative cost
        a: (..., R) first marginal, positive, sums to 1 per leading batch
        b: (..., N) second marginal, positive, sums to 1 per leading batch
        reg: entropic regularization ε
        num_iter: max iterations
        tol: stop if marginal residuals are below this (per batch)

    Returns:
        Transport plan P with same leading shape as cost, (..., R, N)
    """
    if reg <= 0:
        raise ValueError("reg must be positive for Sinkhorn")

    eps = torch.finfo(cost.dtype).tiny
    K = torch.exp(-cost / reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(num_iter):
        u_prev = u
        Kv = torch.einsum("...rn,...n->...r", K, v).clamp_min(eps)
        u = a / Kv
        Kut = torch.einsum("...rn,...r->...n", K, u).clamp_min(eps)
        v = b / Kut
        if tol > 0:
            err = (u - u_prev).abs().max()
            if err < tol:
                break

    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    return P


def plan_to_map(
    plan: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Barycentric projection: rows of plan normalized then applied to targets.

    Args:
        plan: (..., R, N)
        targets: (..., N, d)

    Returns:
        (..., R, d)
    """
    row_sums = plan.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = plan / row_sums
    return torch.einsum("...rn,...nd->...rd", weights, targets)


def lot_maps_from_reference_sinkhorn_batched(
    reference: torch.Tensor,
    targets_batch: torch.Tensor,
    reg: float,
    num_iter: int = 200,
) -> torch.Tensor:
    """
    Batched LOT maps T_t(σ) via Sinkhorn from fixed reference σ to each μ_t.

    Args:
        reference: (R, d)
        targets_batch: (B, N, d)
        reg: Sinkhorn regularization
        num_iter: Sinkhorn iterations

    Returns:
        (B, R, d) maps on same device/dtype as inputs
    """
    R, d = reference.shape
    B, N, d2 = targets_batch.shape
    if d != d2:
        raise ValueError(f"dimension mismatch reference d={d}, targets d={d2}")

    device = reference.device
    dtype = reference.dtype
    src = reference.unsqueeze(0).expand(B, R, d)
    diff = src.unsqueeze(2) - targets_batch.unsqueeze(1)
    cost = (diff * diff).sum(dim=-1)
    a = torch.full((B, R), 1.0 / R, device=device, dtype=dtype)
    b = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    plan = sinkhorn_transport_plan(cost, a, b, reg, num_iter=num_iter)
    return plan_to_map(plan, targets_batch)
