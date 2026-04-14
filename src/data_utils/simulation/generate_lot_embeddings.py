#!/usr/bin/env python3
"""
Compute LOT maps and velocities for all (N, reference_type, fraction) combinations.

Supports two OT assignment modes:
  - per_frame:  Solve OT(σ → μ_t) independently for each frame t (DEFAULT)
                This factors out rotation for rotationally-symmetric references,
                leaving only radial "breathing" dynamics.
  - fixed:      Solve OT(σ → μ_0) once, apply same barycentric weights to all frames
                Preserves temporal consistency but retains rotational dynamics.

Per-frame assignment is the theoretically correct LOT embedding that factors out
rotation when using rotationally-symmetric references (circles, Gaussians).

Creates:
    lot_maps/{system}_N_{N}/{kind}/frac{frac}/lot_maps.npy
    lot_maps/{system}_N_{N}/{kind}/frac{frac}/velocities.npy
    lot_maps/{system}_N_{N}/{kind}/frac{frac}/reference.npy
    lot_maps/{system}_N_{N}/{kind}/frac{frac}/metadata.json

Usage:
    # Per-frame assignment (default - factors out rotation)
    python generate_lot_embeddings.py geodesic_transport
    
    # Fixed-assignment (preserves temporal consistency)
    python generate_lot_embeddings.py geodesic_transport --assignment fixed
    
    # All systems
    python generate_lot_embeddings.py all
"""

import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════
# REFERENCE GENERATORS
# ═══════════════════════════════════════════════════════════════

def make_circle(R, radius=1.0, center=(0, 0)):
    """Generate R points uniformly on a circle."""
    theta = np.linspace(0, 2 * np.pi, R, endpoint=False)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    return np.stack((x, y), axis=1)


def make_triangle_uniform(R, scale=1.5, center=(0, 0)):
    """Generate R points uniformly on a triangle boundary."""
    # Triangle vertices
    v0 = np.array([0, scale]) + np.array(center)
    v1 = np.array([-scale * np.sqrt(3)/2, -scale/2]) + np.array(center)
    v2 = np.array([scale * np.sqrt(3)/2, -scale/2]) + np.array(center)
    
    # Perimeter lengths
    edges = [(v0, v1), (v1, v2), (v2, v0)]
    lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
    total_perimeter = sum(lengths)
    
    # Points per edge proportional to length
    points = []
    for (va, vb), length in zip(edges, lengths):
        n_edge = max(1, int(round(R * length / total_perimeter)))
        t = np.linspace(0, 1, n_edge, endpoint=False)
        edge_points = va + np.outer(t, vb - va)
        points.append(edge_points)
    
    points = np.vstack(points)
    
    # Adjust to exactly R points
    if len(points) > R:
        idx = np.linspace(0, len(points)-1, R, dtype=int)
        points = points[idx]
    elif len(points) < R:
        # Repeat some points
        idx = np.random.default_rng(42).choice(len(points), R, replace=True)
        points = points[idx]
    
    return points


def make_uniform_square(R, side=2.0, center=(0, 0)):
    """Generate R points uniformly in a square."""
    rng = np.random.default_rng(42)
    points = rng.uniform(-side/2, side/2, size=(R, 2))
    points += np.array(center)
    return points


def make_gaussian_iso(R, std=0.5, center=(0, 0)):
    """Generate R points from isotropic Gaussian."""
    rng = np.random.default_rng(42)
    points = rng.normal(0, std, size=(R, 2))
    points += np.array(center)
    return points


def make_gaussian_cloud_centered(R, trajectory, std_scale=0.5):
    """Generate R points from Gaussian centered on the cloud centroid.

    Uses the time-averaged centroid and spread of the actual particle
    trajectory to place reference points where clouds actually are.
    This ensures all reference points are near cloud features, so the
    OT map values (predicted particle positions) stay on clouds.
    """
    centroid = trajectory.mean(axis=(0, 1))  # (2,)
    cloud_std = trajectory.std(axis=(0, 1))  # (2,)
    std = cloud_std * std_scale
    rng = np.random.default_rng(42)
    points = rng.normal(0, 1, size=(R, 2)) * std + centroid
    return points


def make_gaussian_iso_centered(R, std=0.15, center=(0.5, 0.5)):
    """Generate R points from isotropic Gaussian centered on [0,1]^2 data domain.

    For real-world data normalized to [0,1]^2, placing the reference at
    (0.5, 0.5) with a moderate spread ensures the OT coupling is
    informative: reference points near a cloud feature are preferentially
    coupled to it, producing sharper LOT maps and cleaner velocities.
    """
    rng = np.random.default_rng(42)
    points = rng.normal(0, std, size=(R, 2))
    points += np.array(center)
    return points


# ═══════════════════════════════════════════════════════════════
# 3D REFERENCE GENERATORS
# ═══════════════════════════════════════════════════════════════

def make_uniform_cube(R, side=2.0, center=(0, 0, 0)):
    """Generate R points uniformly in a 3D cube."""
    rng = np.random.default_rng(42)
    points = rng.uniform(-side/2, side/2, size=(R, 3))
    points += np.array(center)
    return points


def make_gaussian_iso_3d(R, std=0.5, center=(0, 0, 0)):
    """Generate R points from 3D isotropic Gaussian."""
    rng = np.random.default_rng(42)
    points = rng.normal(0, std, size=(R, 3))
    points += np.array(center)
    return points


def make_sphere(R, radius=1.0, center=(0, 0, 0)):
    """Generate R points uniformly on a sphere surface using Fibonacci sphere."""
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    points = np.zeros((R, 3))
    for i in range(R):
        y = 1 - (i / float(R - 1)) * 2  # y goes from 1 to -1
        r = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * r
        z = np.sin(theta) * r

        points[i] = [x * radius + center[0],
                     y * radius + center[1],
                     z * radius + center[2]]

    return points


def generate_reference(kind: str, R: int, trajectory: np.ndarray = None, d: int = None) -> np.ndarray:
    """
    Generate reference measure of specified kind with R points.

    Args:
        kind: Reference type name
        R: Number of reference points
        trajectory: Full trajectory (needed for snapshot references)
        d: Spatial dimension (2 or 3). Auto-detected from trajectory if not provided.

    Returns:
        Reference points of shape (R, d)
    """
    # Auto-detect dimensionality from trajectory if not provided
    if d is None and trajectory is not None:
        d = trajectory.shape[-1]
    elif d is None:
        d = 2  # Default to 2D for backward compatibility

    # ═══════════════════════════════════════════════════════════════
    # 2D REFERENCES
    # ═══════════════════════════════════════════════════════════════
    if kind == "clean_circle_small":
        if d != 2:
            raise ValueError(f"clean_circle_small only supports d=2, got d={d}")
        return make_circle(R, radius=0.5)

    elif kind == "clean_circle_large":
        if d != 2:
            raise ValueError(f"clean_circle_large only supports d=2, got d={d}")
        return make_circle(R, radius=1.5)

    elif kind == "clean_circle":
        if d != 2:
            raise ValueError(f"clean_circle only supports d=2, got d={d}")
        return make_circle(R, radius=1.0)

    elif kind == "clean_triangle":
        if d != 2:
            raise ValueError(f"clean_triangle only supports d=2, got d={d}")
        return make_triangle_uniform(R, scale=1.5)

    # ═══════════════════════════════════════════════════════════════
    # DIMENSION-AGNOSTIC REFERENCES (work for both 2D and 3D)
    # ═══════════════════════════════════════════════════════════════
    elif kind == "uniform_square":
        # 2D: square, 3D: cube
        if d == 2:
            return make_uniform_square(R, side=2.0)
        elif d == 3:
            return make_uniform_cube(R, side=2.0)
        else:
            raise ValueError(f"uniform_square only supports d=2 or d=3, got d={d}")

    elif kind == "gaussian_iso":
        # Isotropic Gaussian in d dimensions, centered on the data domain
        if trajectory is not None:
            center = tuple(trajectory.mean(axis=(0, 1)).tolist())
            std = float(trajectory.std()) * 0.5
            std = max(std, 0.05)  # floor to avoid degenerate reference
        else:
            center = (0,) * d
            std = 0.5
        if d == 2:
            return make_gaussian_iso(R, std=std, center=center[:2])
        elif d == 3:
            return make_gaussian_iso_3d(R, std=std, center=center[:3])
        else:
            raise ValueError(f"gaussian_iso only supports d=2 or d=3, got d={d}")

    elif kind == "gaussian_cloud_centered":
        # Gaussian centered on the actual cloud centroid with cloud-scale spread
        if trajectory is None:
            raise ValueError("trajectory required for gaussian_cloud_centered")
        return make_gaussian_cloud_centered(R, trajectory)

    elif kind == "gaussian_data_centered":
        # Gaussian centered on [0,1]^2 for real-world data
        if d == 2:
            return make_gaussian_iso_centered(R, std=0.15, center=(0.5, 0.5))
        else:
            raise ValueError(f"gaussian_data_centered only supports d=2, got d={d}")

    # ═══════════════════════════════════════════════════════════════
    # 3D-ONLY REFERENCES
    # ═══════════════════════════════════════════════════════════════
    elif kind == "sphere":
        if d != 3:
            raise ValueError(f"sphere only supports d=3, got d={d}")
        return make_sphere(R, radius=1.0)

    elif kind == "uniform_cube":
        if d != 3:
            raise ValueError(f"uniform_cube only supports d=3, got d={d}")
        return make_uniform_cube(R, side=2.0)

    elif kind == "gaussian_iso_3d":
        if d != 3:
            raise ValueError(f"gaussian_iso_3d only supports d=3, got d={d}")
        return make_gaussian_iso_3d(R, std=0.5)

    # ═══════════════════════════════════════════════════════════════
    # SNAPSHOT REFERENCES (dimension-agnostic - uses trajectory)
    # ═══════════════════════════════════════════════════════════════
    elif kind == "snapshot_begin":
        if trajectory is None:
            raise ValueError("trajectory required for snapshot references")
        # Subsample from first frame
        rng = np.random.default_rng(42)
        N = trajectory.shape[1]
        idx = rng.choice(N, size=R, replace=(R > N))
        return trajectory[0, idx].copy()

    elif kind == "snapshot_middle":
        if trajectory is None:
            raise ValueError("trajectory required for snapshot references")
        T = trajectory.shape[0]
        mid = T // 2
        rng = np.random.default_rng(42)
        N = trajectory.shape[1]
        idx = rng.choice(N, size=R, replace=(R > N))
        return trajectory[mid, idx].copy()

    elif kind == "snapshot_end":
        if trajectory is None:
            raise ValueError("trajectory required for snapshot references")
        rng = np.random.default_rng(42)
        N = trajectory.shape[1]
        idx = rng.choice(N, size=R, replace=(R > N))
        return trajectory[-1, idx].copy()

    else:
        raise ValueError(f"Unknown reference kind: {kind}")


# ═══════════════════════════════════════════════════════════════
# LOT COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_ot_plan(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ot_method: str = "emd",
    ot_device: str = "cpu",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 150,
) -> np.ndarray:
    """
    Compute optimal transport plan from source to target.

    Args:
        source: Reference measure (R, d)
        target: Target measure (N, d)
        ot_method: "emd" (exact, via POT) or "sinkhorn" (entropic, via PyTorch on CPU/GPU)
        ot_device: "cpu" or "cuda" (only for sinkhorn)
        sinkhorn_reg: Entropic regularization ε for Sinkhorn
        sinkhorn_iters: Sinkhorn iterations

    Returns:
        transport_plan: (R, N) coupling (dense for sinkhorn)
    """
    R = source.shape[0]
    N = target.shape[0]

    a = np.ones(R, dtype=np.float64) / R
    b = np.ones(N, dtype=np.float64) / N

    if ot_method == "emd":
        import ot
        M = ot.dist(source, target, metric="sqeuclidean")
        return ot.emd(a, b, M)

    if ot_method == "sinkhorn":
        import torch
        from sinkhorn_ot import sinkhorn_transport_plan

        dev = torch.device(ot_device)
        s = torch.as_tensor(source, dtype=torch.float32, device=dev)
        t = torch.as_tensor(target, dtype=torch.float32, device=dev)
        diff = s.unsqueeze(1) - t.unsqueeze(0)
        cost = (diff * diff).sum(dim=-1).unsqueeze(0)
        aa = torch.full((1, R), 1.0 / R, device=dev, dtype=torch.float32)
        bb = torch.full((1, N), 1.0 / N, device=dev, dtype=torch.float32)
        P = sinkhorn_transport_plan(
            cost, aa, bb, float(sinkhorn_reg), num_iter=int(sinkhorn_iters)
        )
        return P.squeeze(0).detach().cpu().numpy()

    raise ValueError(f"Unknown ot_method: {ot_method!r}; use 'emd' or 'sinkhorn'")


# ═══════════════════════════════════════════════════════════════
# VELOCITY SCALING UTILITIES
# ═══════════════════════════════════════════════════════════════

def scale_velocities_by_m(velocities: np.ndarray, m: int) -> Tuple[np.ndarray, dict]:
    """
    Scale velocities by 1/m where m is the number of reference points.

    This normalizes the velocity magnitudes to a controlled scale that
    doesn't depend on the resolution of the reference measure.

    Args:
        velocities: (T-1, R, d) velocity array
        m: number of reference points (typically R)

    Returns:
        scaled_velocities: velocities / m
        scaling_params: dict with scaling info for unscaling
    """
    scaled = velocities / m
    params = {
        "scaling_method": "1/m",
        "m": int(m),
    }
    return scaled, params


def unscale_velocities_by_m(scaled_velocities: np.ndarray, params: dict) -> np.ndarray:
    """Unscale velocities that were scaled by 1/m."""
    m = params["m"]
    return scaled_velocities * m


def scale_velocities_minmax(
    velocities: np.ndarray,
    eps: float = 1e-8,
    min_floor: float = 0.1,
    window: str = "global"
) -> Tuple[np.ndarray, dict]:
    """
    Scale velocity magnitudes to [min_floor, 1.0] while preserving directions.

    This addresses nonstationarity by normalizing the dynamic range of velocities
    while ensuring slow-but-real velocities don't get pushed to zero.

    Args:
        velocities: (T-1, R, d) velocity array
        eps: small constant to avoid division by zero
        min_floor: minimum scaled magnitude (prevents zeroing slow points)
        window: "global" uses min/max over entire trajectory (recommended)

    Returns:
        scaled_velocities: direction-preserved, magnitude-scaled velocities
        scaling_params: dict with {mag_min, mag_max, min_floor} for unscaling
    """
    # Compute magnitudes: (T-1, R)
    mags = np.linalg.norm(velocities, axis=-1, keepdims=True)  # (T-1, R, 1)

    # Global min/max over entire trajectory
    mag_min = float(mags.min())
    mag_max = float(mags.max())
    mag_range = mag_max - mag_min

    # Handle edge case: constant velocity field
    if mag_range < eps:
        # All velocities have same magnitude, just normalize to 1
        scaled = velocities / (mag_max + eps)
        params = {
            "scaling_method": "minmax",
            "mag_min": mag_min,
            "mag_max": mag_max,
            "min_floor": min_floor,
            "constant_field": True,
        }
        return scaled, params

    # Unit directions (handle zero velocities gracefully)
    directions = velocities / (mags + eps)

    # Scale magnitudes to [min_floor, 1.0]
    # Formula: scaled_mag = min_floor + (1 - min_floor) * (mag - mag_min) / range
    normalized_mags = (mags - mag_min) / (mag_range + eps)  # [0, 1]
    scaled_mags = min_floor + (1.0 - min_floor) * normalized_mags  # [min_floor, 1]

    scaled_velocities = directions * scaled_mags

    params = {
        "scaling_method": "minmax",
        "mag_min": mag_min,
        "mag_max": mag_max,
        "min_floor": min_floor,
        "constant_field": False,
    }

    return scaled_velocities, params


def unscale_velocities_minmax(scaled_velocities: np.ndarray, params: dict, eps: float = 1e-8) -> np.ndarray:
    """
    Unscale velocities that were scaled with minmax method.

    Recovers original magnitudes while preserving the predicted directions.
    """
    if params.get("constant_field", False):
        return scaled_velocities * params["mag_max"]

    mag_min = params["mag_min"]
    mag_max = params["mag_max"]
    min_floor = params["min_floor"]
    mag_range = mag_max - mag_min

    # Get scaled magnitudes and directions
    scaled_mags = np.linalg.norm(scaled_velocities, axis=-1, keepdims=True)
    directions = scaled_velocities / (scaled_mags + eps)

    # Invert the scaling: original_mag = mag_min + (scaled_mag - min_floor) / (1 - min_floor) * range
    normalized_mags = (scaled_mags - min_floor) / (1.0 - min_floor + eps)
    original_mags = mag_min + normalized_mags * mag_range

    return directions * original_mags


def scale_velocities(
    velocities: np.ndarray,
    method: str = "none",
    m: int = None,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Scale velocities using the specified method.

    Args:
        velocities: (T-1, R, d) velocity array
        method: "none", "1/m", or "minmax"
        m: number of reference points (required for "1/m" method)
        **kwargs: additional args passed to specific scaling function

    Returns:
        scaled_velocities: scaled velocity array
        scaling_params: dict with method and parameters for unscaling
    """
    if method == "none" or method is None:
        return velocities, {"scaling_method": "none"}

    elif method == "1/m":
        if m is None:
            raise ValueError("m (number of reference points) required for 1/m scaling")
        return scale_velocities_by_m(velocities, m)

    elif method == "minmax":
        min_floor = kwargs.get("min_floor", 0.1)
        return scale_velocities_minmax(velocities, min_floor=min_floor)

    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'none', '1/m', or 'minmax'")


def unscale_velocities(scaled_velocities: np.ndarray, params: dict) -> np.ndarray:
    """
    Unscale velocities using the parameters saved during scaling.

    Args:
        scaled_velocities: scaled velocity array
        params: scaling parameters dict (from scale_velocities)

    Returns:
        unscaled_velocities: original-scale velocity array
    """
    method = params.get("scaling_method", "none")

    if method == "none":
        return scaled_velocities
    elif method == "1/m":
        return unscale_velocities_by_m(scaled_velocities, params)
    elif method == "minmax":
        return unscale_velocities_minmax(scaled_velocities, params)
    else:
        raise ValueError(f"Unknown scaling method in params: {method}")


def plan_to_barycentric_weights(plan: np.ndarray) -> np.ndarray:
    """
    Convert transport plan to normalized barycentric weights.
    
    Args:
        plan: (R, N) transport plan
    
    Returns:
        weights: (R, N) where each row sums to 1
    """
    row_sums = plan.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
    return plan / row_sums


def apply_barycentric_projection(weights: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Apply barycentric projection: weighted average of target points.
    
    Args:
        weights: (R, N) barycentric weights
        targets: (N, 2) target points
    
    Returns:
        projected: (R, 2) barycentric projections
    """
    return weights @ targets


def compute_ot_map(source: np.ndarray, target: np.ndarray, **ot_kwargs) -> np.ndarray:
    """
    Compute optimal transport map from source to target (per-frame version).

    Args:
        source: Reference measure (R, d)
        target: Target measure (N, d)
        **ot_kwargs: forwarded to ``compute_ot_plan`` (ot_method, ot_device, ...)

    Returns:
        Mapped points (R, d) — barycentric projection
    """
    plan = compute_ot_plan(source, target, **ot_kwargs)
    weights = plan_to_barycentric_weights(plan)
    return apply_barycentric_projection(weights, target)


def compute_periodic_velocities(lot_maps: np.ndarray) -> np.ndarray:
    """
    Compute velocities accounting for periodic boundaries in [0,1]^d.

    When a particle wraps from 0.99 to 0.01, the true velocity is +0.02,
    not the naive -0.98. This function computes the "shortest path" velocity.

    Args:
        lot_maps: (T, R, d) array of LOT maps

    Returns:
        velocities: (T-1, R, d) array of periodic-aware velocities
    """
    # Naive difference
    diff = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, d)

    # For periodic boundaries in [0,1], if |diff| > 0.5, wrap around
    # This gives the shortest path velocity
    diff = np.where(diff > 0.5, diff - 1.0, diff)
    diff = np.where(diff < -0.5, diff + 1.0, diff)

    return diff


def compute_lot_embedding_perframe(
    trajectory: np.ndarray,
    reference: np.ndarray,
    verbose: bool = False,
    periodic_boundary: bool = False,
    *,
    ot_method: str = "emd",
    ot_device: str = "cpu",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 150,
    sinkhorn_batch_frames: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LOT maps using per-frame OT (theoretically correct method).

    Solves OT(σ → μ_t) independently for each frame t.
    This factors out rotation when using rotationally-symmetric references.

    Args:
        trajectory: Particle positions (T, N, d) where d is 2 or 3
        reference: Reference measure (R, d)
        verbose: Print progress
        periodic_boundary: If True, compute velocities accounting for [0,1]^d wrap
        ot_method: "emd" or "sinkhorn"
        ot_device: "cpu" or "cuda" for batched Sinkhorn
        sinkhorn_reg, sinkhorn_iters: Sinkhorn hyperparameters
        sinkhorn_batch_frames: frames per GPU batch when ot_method=="sinkhorn"

    Returns:
        lot_maps: (T, R, d) - OT maps at each timestep
        velocities: (T-1, R, d) - temporal differences
    """
    T, N, d = trajectory.shape
    R = reference.shape[0]

    lot_maps = np.zeros((T, R, d), dtype=np.float32)

    if ot_method == "sinkhorn":
        import torch
        from sinkhorn_ot import lot_maps_from_reference_sinkhorn_batched

        dev = torch.device(ot_device)
        ref_t = torch.as_tensor(reference, dtype=torch.float32, device=dev)
        for start in range(0, T, sinkhorn_batch_frames):
            if verbose:
                print(f"  [per-frame sinkhorn {ot_device}] t={start}/{T}...")
            end = min(start + sinkhorn_batch_frames, T)
            tgt_t = torch.as_tensor(trajectory[start:end], dtype=torch.float32, device=dev)
            maps_b = lot_maps_from_reference_sinkhorn_batched(
                ref_t, tgt_t, float(sinkhorn_reg), int(sinkhorn_iters)
            )
            lot_maps[start:end] = maps_b.detach().cpu().numpy().astype(np.float32)
    else:
        for t in range(T):
            if verbose and t % 50 == 0:
                print(f"  [per-frame emd] Computing LOT map t={t}/{T}...")

            target = trajectory[t]
            lot_maps[t] = compute_ot_map(reference, target, ot_method="emd")

    # Compute velocities (periodic-aware if needed)
    if periodic_boundary:
        velocities = compute_periodic_velocities(lot_maps)
    else:
        velocities = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, d)

    return lot_maps, velocities


def temporal_smooth_maps(
    lot_maps: np.ndarray,
    window: int = 5,
    periodic_boundary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporally smooth LOT maps to reduce particle-resampling noise,
    then recompute velocities from the smoothed maps.

    For real-world data where particles are independently resampled each
    frame (e.g. GOES cloud data), the frame-to-frame LOT map differences
    are dominated by sampling noise.  A causal moving-average filter along
    the time axis preserves physical cloud evolution while averaging out
    random fluctuations.

    Args:
        lot_maps: (T, R, d) raw LOT maps
        window: Size of the causal moving-average window.
            window=1 means no smoothing.
        periodic_boundary: If True, use periodic-aware velocity computation.

    Returns:
        smoothed_maps: (T, R, d) temporally smoothed LOT maps
        velocities: (T-1, R, d) velocities from the smoothed maps
    """
    if window <= 1:
        vels = lot_maps[1:] - lot_maps[:-1]
        if periodic_boundary:
            vels = np.where(vels > 0.5, vels - 1.0, vels)
            vels = np.where(vels < -0.5, vels + 1.0, vels)
        return lot_maps.copy(), vels

    T, R, d = lot_maps.shape
    smoothed = np.empty_like(lot_maps)

    # Causal moving average (only uses past + current, no look-ahead)
    for t in range(T):
        start = max(0, t - window + 1)
        smoothed[t] = lot_maps[start:t + 1].mean(axis=0)

    if periodic_boundary:
        velocities = compute_periodic_velocities(smoothed)
    else:
        velocities = smoothed[1:] - smoothed[:-1]

    return smoothed, velocities


def compute_lot_embedding_fixed(
    trajectory: np.ndarray,
    reference: np.ndarray,
    anchor_frame: int = 0,
    verbose: bool = False,
    periodic_boundary: bool = False,
    *,
    ot_method: str = "emd",
    ot_device: str = "cpu",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LOT maps using fixed-assignment OT.

    Solves OT(σ → μ_anchor) ONCE, then applies the same barycentric weights
    to all frames. This preserves temporal consistency but retains rotation.

    Args:
        trajectory: Particle positions (T, N, d) where d is 2 or 3
        reference: Reference measure (R, d)
        anchor_frame: Which frame to compute OT assignment from (default: 0)
        verbose: Print progress
        periodic_boundary: If True, compute velocities accounting for [0,1]^d wrap

    Returns:
        lot_maps: (T, R, d) - OT maps at each timestep
        velocities: (T-1, R, d) - temporal differences
        barycentric_weights: (R, N) - fixed weights used for all frames
    """
    T, N, d = trajectory.shape
    R = reference.shape[0]

    # Compute OT plan ONCE at anchor frame
    anchor_target = trajectory[anchor_frame]
    if verbose:
        print(f"  [fixed] Computing OT plan at anchor frame {anchor_frame}...")

    plan = compute_ot_plan(
        reference,
        anchor_target,
        ot_method=ot_method,
        ot_device=ot_device,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_iters=sinkhorn_iters,
    )
    barycentric_weights = plan_to_barycentric_weights(plan)

    if verbose:
        sparsity = (plan > 1e-10).sum()
        print(f"  [fixed] Plan sparsity: {sparsity}/{plan.size} nonzero")

    # Apply same weights to ALL frames
    lot_maps = np.zeros((T, R, d), dtype=np.float32)

    for t in range(T):
        if verbose and t % 100 == 0:
            print(f"  [fixed] Applying projection t={t}/{T}...")
        lot_maps[t] = apply_barycentric_projection(barycentric_weights, trajectory[t])

    # Compute velocities (periodic-aware if needed)
    if periodic_boundary:
        velocities = compute_periodic_velocities(lot_maps)
    else:
        velocities = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, d)

    return lot_maps, velocities, barycentric_weights


def compute_lot_embedding(
    trajectory: np.ndarray,
    reference: np.ndarray,
    assignment: str = "per_frame",
    anchor_frame: int = 0,
    verbose: bool = False,
    periodic_boundary: bool = False,
    *,
    ot_method: str = "emd",
    ot_device: str = "cpu",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 150,
    sinkhorn_batch_frames: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute LOT maps and velocities for a trajectory.

    Args:
        trajectory: Particle positions (T, N, d) where d is 2 or 3
        reference: Reference measure (R, d)
        assignment: "per_frame" (default, factors out rotation) or "fixed"
        anchor_frame: For fixed assignment, which frame to anchor to
        verbose: Print progress
        periodic_boundary: If True, compute velocities accounting for [0,1]^d wrap
        ot_method: "emd" (POT, exact) or "sinkhorn" (PyTorch, batched on CPU/GPU)
        ot_device: "cpu" or "cuda" when using sinkhorn
        sinkhorn_reg, sinkhorn_iters: entropic OT parameters
        sinkhorn_batch_frames: number of time frames per GPU batch (per_frame + sinkhorn)

    Returns:
        lot_maps: (T, R, d) - OT maps at each timestep
        velocities: (T-1, R, d) - temporal differences
        barycentric_weights: (R, N) or None - weights if fixed assignment
    """
    if assignment == "per_frame":
        lot_maps, velocities = compute_lot_embedding_perframe(
            trajectory,
            reference,
            verbose,
            periodic_boundary,
            ot_method=ot_method,
            ot_device=ot_device,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_batch_frames=sinkhorn_batch_frames,
        )
        return lot_maps, velocities, None
    elif assignment == "fixed":
        return compute_lot_embedding_fixed(
            trajectory,
            reference,
            anchor_frame,
            verbose,
            periodic_boundary,
            ot_method=ot_method,
            ot_device=ot_device,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_iters=sinkhorn_iters,
        )
    else:
        raise ValueError(f"Unknown assignment mode: {assignment}. Use 'per_frame' or 'fixed'")


# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════

DEFAULT_KINDS = [
    "clean_circle_small",
    "clean_circle_large", 
    "uniform_square",
    "gaussian_iso",
    "snapshot_begin",
    "snapshot_middle",
    "snapshot_end",
]

# System-specific reference kinds
GEODESIC_KINDS = [
    "clean_circle",       # Circle matching source (radius=1.0)
    "clean_triangle",     # Triangle matching target (scale=1.5)
    "uniform_square",
    "gaussian_iso",
    "snapshot_begin",
    "snapshot_middle",
    "snapshot_end",
]

SWIRLING_KINDS = [
    "clean_circle_small",
    "clean_circle_large",
    "uniform_square",
    "gaussian_iso",
    "snapshot_begin",
    "snapshot_middle",
    "snapshot_end",
]

DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


def get_default_kinds(system: str) -> List[str]:
    """Get system-specific default reference kinds."""
    if "geodesic" in system.lower():
        return GEODESIC_KINDS
    elif "swirling" in system.lower():
        return SWIRLING_KINDS
    else:
        return DEFAULT_KINDS


def generate_lot_embeddings(
    system: str,
    N_list: List[int] = [100, 250, 500, 750, 1000, 1500],
    kinds: List[str] = None,
    fractions: List[int] = None,
    assignment: str = "per_frame",
    anchor_frame: int = 0,
    velocity_scaling: str = "none",
    velocity_scaling_min_floor: float = 0.1,
    periodic_boundary: bool = False,
    results_root: Path = Path("results"),
    lot_root: Path = Path("lot_maps"),
    verbose: bool = False,
    ot_method: str = "emd",
    ot_device: str = "cpu",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 150,
    sinkhorn_batch_frames: int = 64,
):
    """
    Generate LOT embeddings for all configurations.

    Args:
        system: System name ("geodesic_transport" or "swirling_cluster")
        N_list: List of particle counts
        kinds: Reference types (default: system-specific)
        fractions: Fraction percentages
        assignment: "per_frame" (default) or "fixed"
        anchor_frame: For fixed assignment, which frame to anchor to
        velocity_scaling: Velocity scaling method:
            - "none": No scaling (default, backward compatible)
            - "1/m": Scale by 1/m where m=R (number of reference points)
            - "minmax": Min-max scale magnitudes to [min_floor, 1.0], preserve directions
        velocity_scaling_min_floor: For minmax scaling, minimum scaled magnitude (default 0.1)
        periodic_boundary: If True, compute velocities accounting for [0,1]^d wrap.
            Use this for closed systems with periodic boundaries (tidal, SST).
        results_root: Directory containing trajectories
        lot_root: Output directory for LOT maps
        verbose: Print detailed progress
    """
    kinds = kinds or get_default_kinds(system)
    fractions = fractions or DEFAULT_FRACTIONS

    results_root = Path(results_root)
    lot_root = Path(lot_root)

    print(f"\n{'='*60}")
    print(f" Generating LOT embeddings: {system}")
    print(f" Assignment mode: {assignment}")
    print(f" OT method: {ot_method}" + (f" ({ot_device})" if ot_method == "sinkhorn" else ""))
    print(f" Velocity scaling: {velocity_scaling}")
    if periodic_boundary:
        print(f" Periodic boundary: ON (velocities wrap in [0,1]^d)")
    print(f"{'='*60}")
    
    total_runs = len(N_list) * len(kinds) * len(fractions)
    run_count = 0
    
    for N in N_list:
        # Load trajectory
        traj_path = results_root / f"{system}_N_{N}" / "trajectory_jittered.npy"
        if not traj_path.exists():
            traj_path = results_root / f"{system}_N_{N}" / "trajectory.npy"
        
        if not traj_path.exists():
            print(f"[MISS] {traj_path}")
            continue
        
        trajectory = np.load(traj_path)
        T, N_traj, d = trajectory.shape
        print(f"\n[INFO] N={N}: trajectory {trajectory.shape}")
        
        for kind in kinds:
            for frac in fractions:
                run_count += 1
                
                # Compute R (reference size) as fraction of N
                R = max(1, int(round(N * frac / 100)))
                
                # Generate reference
                try:
                    reference = generate_reference(kind, R, trajectory)
                    # REMOVED: np.save(out_dir / "reference.npy", ...) - out_dir not defined yet!
                except ValueError as e:
                    print(f"[SKIP] N={N} {kind} {frac}%: {e}")
                    continue
                
                print(f"[RUN] N={N} {kind} {frac}% -> R={R}, T={T}, d={d}, assignment={assignment}")
                
                # Compute LOT embedding
                lot_maps, velocities, bary_weights = compute_lot_embedding(
                    trajectory,
                    reference,
                    assignment=assignment,
                    anchor_frame=anchor_frame,
                    verbose=verbose,
                    periodic_boundary=periodic_boundary,
                    ot_method=ot_method,
                    ot_device=ot_device,
                    sinkhorn_reg=sinkhorn_reg,
                    sinkhorn_iters=sinkhorn_iters,
                    sinkhorn_batch_frames=sinkhorn_batch_frames,
                )

                # Velocity statistics (before scaling)
                vel_std_raw = velocities.std()
                vel_mag_raw = np.linalg.norm(velocities, axis=-1).mean()

                # Apply velocity scaling
                velocities_scaled, scaling_params = scale_velocities(
                    velocities,
                    method=velocity_scaling,
                    m=R,
                    min_floor=velocity_scaling_min_floor,
                )

                # Velocity statistics (after scaling)
                vel_std_scaled = velocities_scaled.std()
                vel_mag_scaled = np.linalg.norm(velocities_scaled, axis=-1).mean()

                # Save
                out_dir = lot_root / f"{system}_N_{N}" / kind / f"frac{frac}"
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(out_dir / "lot_maps.npy", lot_maps)
                np.save(out_dir / "velocities.npy", velocities_scaled)  # Save scaled velocities
                np.save(out_dir / "reference.npy", reference)

                # Also save raw velocities for reference/debugging if scaling is applied
                if velocity_scaling != "none":
                    np.save(out_dir / "velocities_raw.npy", velocities)

                if bary_weights is not None:
                    np.save(out_dir / "barycentric_weights.npy", bary_weights)

                # Save metadata (including scaling parameters)
                metadata = {
                    "system": system,
                    "N": N,
                    "R": R,
                    "T": T,
                    "d": d,
                    "kind": kind,
                    "frac_pct": frac,
                    "assignment": assignment,
                    "anchor_frame": anchor_frame if assignment == "fixed" else None,
                    # Raw velocity stats
                    "velocity_std_raw": float(vel_std_raw),
                    "velocity_mean_magnitude_raw": float(vel_mag_raw),
                    # Scaled velocity stats
                    "velocity_std": float(vel_std_scaled),
                    "velocity_mean_magnitude": float(vel_mag_scaled),
                    # Scaling parameters (for unscaling during rollout)
                    "velocity_scaling": scaling_params,
                    # Boundary handling
                    "periodic_boundary": periodic_boundary,
                    "trajectory_source": str(traj_path),
                    "ot_method": ot_method,
                    "ot_device": ot_device if ot_method == "sinkhorn" else None,
                    "sinkhorn_reg": sinkhorn_reg if ot_method == "sinkhorn" else None,
                    "sinkhorn_iters": sinkhorn_iters if ot_method == "sinkhorn" else None,
                    "sinkhorn_batch_frames": sinkhorn_batch_frames if ot_method == "sinkhorn" else None,
                }

                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                scaling_info = f" | scaling={velocity_scaling}" if velocity_scaling != "none" else ""
                print(f"[SAVED] {out_dir}/lot_maps.npy | vel_std_raw={vel_std_raw:.6f} vel_std_scaled={vel_std_scaled:.6f}{scaling_info}")
    
    print(f"\n[DONE] Generated {run_count} LOT embeddings for {system}")

    
# ═══════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════

"""
LOT Embedding Diagnostics for Forecasting.

This module provides diagnostic tools to assess whether LOT embeddings
are suitable for dynamics forecasting with reservoir computing.

Key insight: For forecasting, we want LOT velocities that capture the
true underlying dynamics. If rotation is "factored out" by per-frame
OT assignment, the velocities may not reflect the actual motion.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LOTDiagnostics:
    """Container for LOT embedding diagnostic results."""
    
    # Basic info
    lot_dir: Path
    system: Optional[str]
    assignment_mode: Optional[str]
    
    # Velocity statistics
    velocity_std: float
    velocity_mean_magnitude: float
    velocity_change_rate: float
    smoothness_ratio: float
    
    # Rotation analysis
    total_rotation_degrees: float
    rotation_factored_out: bool
    
    # Autocorrelation
    autocorr_lag5: float
    autocorr_lag10: float
    autocorr_lag25: float
    
    # Warnings
    warnings: List[str]
    forecasting_suitable: bool
    
    def print_report(self):
        """Print a formatted diagnostic report."""
        print(f"\n{'='*60}")
        print(f" LOT Embedding Diagnostics")
        print(f" {self.lot_dir}")
        print(f"{'='*60}")
        
        print(f"\nSystem: {self.system or 'unknown'}")
        print(f"Assignment mode: {self.assignment_mode or 'unknown'}")
        
        print(f"\nVelocity Statistics:")
        print(f"  Standard deviation: {self.velocity_std:.6f}")
        print(f"  Mean magnitude: {self.velocity_mean_magnitude:.6f}")
        print(f"  Change rate: {self.velocity_change_rate:.6f}")
        print(f"  Smoothness ratio: {self.smoothness_ratio:.2f} "
              f"{'✓' if self.smoothness_ratio > 1 else '⚠️ (noisy)'}")
        
        print(f"\nRotation Analysis:")
        print(f"  Total rotation: {self.total_rotation_degrees:.1f}°")
        if self.rotation_factored_out:
            print(f"  ⚠️  ROTATION FACTORED OUT")
            print(f"      This may indicate per-frame assignment eliminated rotational dynamics.")
            print(f"      For forecasting rotationally-symmetric systems, consider using")
            print(f"      fixed assignment mode instead.")
        else:
            print(f"  ✓ Rotation preserved in embedding")
        
        print(f"\nAutocorrelation (temporal structure):")
        print(f"  Lag 5:  {self.autocorr_lag5:.4f}")
        print(f"  Lag 10: {self.autocorr_lag10:.4f}")
        print(f"  Lag 25: {self.autocorr_lag25:.4f}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS FOR FORECASTING:")
            for w in self.warnings:
                print(f"  • {w}")
        
        print(f"\nForecasting Suitability: ", end="")
        if self.forecasting_suitable:
            print("✓ GOOD")
        else:
            print("⚠️  POTENTIAL ISSUES (see warnings)")
        
        print(f"{'='*60}\n")


def diagnose_lot_embedding(lot_dir: Path, system_name: Optional[str] = None) -> LOTDiagnostics:
    """
    Comprehensive diagnostics for a LOT embedding directory.
    
    Analyzes velocity structure, rotation, and temporal dynamics to assess
    whether the embedding is suitable for reservoir computing forecasting.
    
    Args:
        lot_dir: Path to LOT embedding directory (contains lot_maps.npy, velocities.npy)
        system_name: Optional system name for context-aware diagnostics
        
    Returns:
        LOTDiagnostics dataclass with all diagnostic results
    """
    lot_dir = Path(lot_dir)
    warnings = []
    
    # Load data
    velocities = np.load(lot_dir / "velocities.npy")
    lot_maps = np.load(lot_dir / "lot_maps.npy")
    T_vel, R, d = velocities.shape
    T_maps = lot_maps.shape[0]
    
    # Load metadata if available
    meta_path = lot_dir / "metadata.json"
    system = system_name
    assignment_mode = None
    
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        system = system or meta.get("system")
        assignment_mode = meta.get("assignment")
    
    # ═══════════════════════════════════════════════════════════════
    # VELOCITY STATISTICS
    # ═══════════════════════════════════════════════════════════════
    
    vel_std = float(velocities.std())
    vel_mag = float(np.linalg.norm(velocities, axis=-1).mean())
    
    # Change rate: how much velocities change between timesteps
    vel_diffs = np.diff(velocities, axis=0)
    change_rate = float(np.linalg.norm(vel_diffs, axis=-1).mean())
    
    # Smoothness ratio: high = smooth dynamics, low = noisy/discontinuous
    smoothness = vel_std / change_rate if change_rate > 1e-10 else float('inf')
    
    if smoothness < 1.0:
        warnings.append(
            f"Low smoothness ratio ({smoothness:.2f}): velocities change rapidly between "
            f"timesteps. This may indicate per-frame OT assignment causing discontinuities."
        )
    
    # ═══════════════════════════════════════════════════════════════
    # ROTATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    
    # Track rotation of first reference point around centroid
    centroids = lot_maps.mean(axis=1)  # (T, d)
    
    # Compute angle of first point relative to centroid
    rel_pos = lot_maps[:, 0, :] - centroids  # (T, d)
    angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
    angles_unwrap = np.unwrap(angles)
    total_rotation = angles_unwrap[-1] - angles_unwrap[0]
    total_rotation_deg = float(np.degrees(total_rotation))
    
    rotation_factored_out = abs(total_rotation) < 0.1  # Less than ~6 degrees total
    
    # Context-aware rotation warning
    if rotation_factored_out:
        if system and "swirl" in system.lower():
            warnings.append(
                f"Rotation factored out for swirling system! "
                f"Total rotation = {total_rotation_deg:.1f}° (expected ~720° for 2 cycles). "
                f"Per-frame OT assignment may have eliminated the primary rotational dynamics. "
                f"Consider using fixed assignment mode for forecasting."
            )
        elif system and "geodesic" in system.lower():
            # For geodesic transport, zero rotation is expected and correct
            pass
        else:
            warnings.append(
                f"Rotation factored out (total = {total_rotation_deg:.1f}°). "
                f"If the underlying system has rotational dynamics, they may not be "
                f"captured in the LOT velocities."
            )
    
    # ═══════════════════════════════════════════════════════════════
    # AUTOCORRELATION (temporal structure)
    # ═══════════════════════════════════════════════════════════════
    
    vel_flat = velocities.reshape(T_vel, -1)
    
    def safe_autocorr(lag):
        if lag >= T_vel:
            return np.nan
        v1 = vel_flat[:-lag].flatten()
        v2 = vel_flat[lag:].flatten()
        if v1.std() < 1e-10 or v2.std() < 1e-10:
            return np.nan
        return float(np.corrcoef(v1, v2)[0, 1])
    
    autocorr_5 = safe_autocorr(5)
    autocorr_10 = safe_autocorr(10)
    autocorr_25 = safe_autocorr(25)
    
    # Check for concerning autocorrelation patterns
    if not np.isnan(autocorr_5) and autocorr_5 < 0.3:
        warnings.append(
            f"Low short-term autocorrelation (lag-5 = {autocorr_5:.3f}). "
            f"Velocities may lack temporal structure needed for reservoir learning."
        )
    
    # ═══════════════════════════════════════════════════════════════
    # ASSIGNMENT MODE WARNINGS
    # ═══════════════════════════════════════════════════════════════
    
    if assignment_mode == "per_frame":
        if system and "swirl" in system.lower():
            warnings.append(
                "Using per-frame assignment for swirling system. "
                "This factors out rotation at each timestep, potentially "
                "eliminating the primary dynamics you want to forecast. "
                "Consider regenerating LOT embeddings with --assignment fixed"
            )
    
    # ═══════════════════════════════════════════════════════════════
    # OVERALL SUITABILITY
    # ═══════════════════════════════════════════════════════════════
    
    # Heuristic: suitable if no critical warnings
    critical_keywords = ["eliminating", "factored out", "primary dynamics"]
    has_critical_warning = any(
        any(kw in w.lower() for kw in critical_keywords)
        for w in warnings
    )
    forecasting_suitable = not has_critical_warning
    
    return LOTDiagnostics(
        lot_dir=lot_dir,
        system=system,
        assignment_mode=assignment_mode,
        velocity_std=vel_std,
        velocity_mean_magnitude=vel_mag,
        velocity_change_rate=change_rate,
        smoothness_ratio=smoothness,
        total_rotation_degrees=total_rotation_deg,
        rotation_factored_out=rotation_factored_out,
        autocorr_lag5=autocorr_5,
        autocorr_lag10=autocorr_10,
        autocorr_lag25=autocorr_25,
        warnings=warnings,
        forecasting_suitable=forecasting_suitable,
    )


def diagnose_lot_velocities(lot_dir: Path) -> None:
    """
    Legacy API: Print diagnostics for a LOT embedding.
    
    This function provides the same interface as the original, but with
    improved warnings for forecasting applications.
    
    Args:
        lot_dir: Directory containing lot_maps.npy and velocities.npy
    """
    diagnostics = diagnose_lot_embedding(lot_dir)
    diagnostics.print_report()


def batch_diagnose(
    lot_root: Path,
    system: str,
    N_list: Optional[List[int]] = None,
    kinds: Optional[List[str]] = None,
    fracs: Optional[List[int]] = None,
) -> Dict[str, LOTDiagnostics]:
    """
    Run diagnostics on multiple LOT embeddings.
    
    Args:
        lot_root: Root directory for LOT embeddings
        system: System name
        N_list: List of N values (default: all found)
        kinds: List of reference types (default: all found)
        fracs: List of fraction percentages (default: all found)
        
    Returns:
        Dictionary mapping config string to diagnostics
    """
    lot_root = Path(lot_root)
    results = {}
    
    # Find all matching directories
    system_dirs = list(lot_root.glob(f"{system}_N_*"))
    
    for sys_dir in system_dirs:
        N = int(sys_dir.name.split("_N_")[1])
        if N_list and N not in N_list:
            continue
        
        for kind_dir in sys_dir.iterdir():
            if not kind_dir.is_dir():
                continue
            kind = kind_dir.name
            if kinds and kind not in kinds:
                continue
            
            for frac_dir in kind_dir.iterdir():
                if not frac_dir.is_dir():
                    continue
                if not frac_dir.name.startswith("frac"):
                    continue
                
                frac = int(frac_dir.name[4:])
                if fracs and frac not in fracs:
                    continue
                
                lot_maps_path = frac_dir / "lot_maps.npy"
                if not lot_maps_path.exists():
                    continue
                
                try:
                    diag = diagnose_lot_embedding(frac_dir, system_name=system)
                    key = f"N={N}/{kind}/frac{frac}"
                    results[key] = diag
                except Exception as e:
                    print(f"[ERROR] {frac_dir}: {e}")
    
    return results


def summarize_batch_diagnostics(diagnostics: Dict[str, LOTDiagnostics]) -> None:
    """Print summary of batch diagnostics."""
    
    total = len(diagnostics)
    suitable = sum(1 for d in diagnostics.values() if d.forecasting_suitable)
    rotation_issues = sum(1 for d in diagnostics.values() if d.rotation_factored_out)
    smoothness_issues = sum(1 for d in diagnostics.values() if d.smoothness_ratio < 1.0)
    
    print(f"\n{'='*60}")
    print(f" BATCH DIAGNOSTICS SUMMARY")
    print(f"{'='*60}")
    print(f"Total embeddings analyzed: {total}")
    print(f"Suitable for forecasting: {suitable}/{total} ({100*suitable/total:.1f}%)")
    print(f"Rotation factored out: {rotation_issues}/{total}")
    print(f"Low smoothness: {smoothness_issues}/{total}")
    
    # List problematic embeddings
    problems = [k for k, v in diagnostics.items() if not v.forecasting_suitable]
    if problems:
        print(f"\nProblematic embeddings:")
        for p in problems[:10]:  # Limit output
            print(f"  • {p}")
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
    
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LOT embedding diagnostics for forecasting")
    parser.add_argument("path", type=Path, help="LOT directory or lot_root for batch mode")
    parser.add_argument("--batch", action="store_true", help="Run batch diagnostics")
    parser.add_argument("--system", type=str, default=None, help="System name for batch mode")
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--kinds", type=str, nargs="+", default=None)
    parser.add_argument("--fracs", type=int, nargs="+", default=None)
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.system:
            parser.error("--system required for batch mode")
        results = batch_diagnose(
            args.path, args.system, args.N, args.kinds, args.fracs
        )
        summarize_batch_diagnostics(results)
    else:
        diagnose_lot_velocities(args.path)



# def diagnose_lot_velocities(lot_dir: Path):
#     """
#     Diagnose velocity structure for a LOT embedding.
    
#     Args:
#         lot_dir: Directory containing lot_maps.npy and velocities.npy
#     """
#     lot_dir = Path(lot_dir)
    
#     velocities = np.load(lot_dir / "velocities.npy")
#     T, R, d = velocities.shape
    
#     print(f"\nDiagnostics for {lot_dir}")
#     print(f"  Velocities shape: {velocities.shape}")
    
#     # Smoothness
#     vel_std = velocities.std()
#     vel_diffs = np.diff(velocities, axis=0)
#     change_rate = np.linalg.norm(vel_diffs, axis=-1).mean()
#     smoothness = vel_std / change_rate if change_rate > 0 else float('inf')
    
#     print(f"  Velocity std: {vel_std:.6f}")
#     print(f"  Change rate: {change_rate:.6f}")
#     print(f"  Smoothness ratio: {smoothness:.2f} {'✓' if smoothness > 1 else '⚠️'}")
    
#     # Autocorrelation
#     vel_flat = velocities.reshape(T, -1)
#     print(f"  Autocorrelation:")
#     for lag in [5, 10, 25]:
#         if lag < T:
#             corr = np.corrcoef(vel_flat[:-lag].flatten(), vel_flat[lag:].flatten())[0, 1]
#             print(f"    lag {lag}: {corr:.4f}")
    
#     # Check metadata
#     meta_path = lot_dir / "metadata.json"
#     if meta_path.exists():
#         with open(meta_path) as f:
#             meta = json.load(f)
#         print(f"  Assignment: {meta.get('assignment', 'unknown')}")
    
#     # Check for rotation (specific to per-frame vs fixed)
#     lot_maps = np.load(lot_dir / "lot_maps.npy")
#     centroids = lot_maps.mean(axis=1)
#     angles = np.arctan2(lot_maps[:, 0, 1] - centroids[:, 1],
#                         lot_maps[:, 0, 0] - centroids[:, 0])
#     angles_unwrap = np.unwrap(angles)
#     total_rotation = angles_unwrap[-1] - angles_unwrap[0]
    
#     print(f"  Total rotation: {np.degrees(total_rotation):.1f}°")
#     if abs(total_rotation) < 0.1:
#         print(f"    → Rotation factored out (per-frame assignment)")
#     else:
#         print(f"    → Rotation preserved (fixed assignment)")


# # ═══════════════════════════════════════════════════════════════
# # CLI
# # ═══════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate LOT embeddings with per-frame or fixed assignment",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#     # Per-frame assignment (default - factors out rotation)
#     python generate_lot_embeddings.py geodesic_transport
    
#     # Fixed-assignment (preserves temporal consistency)
#     python generate_lot_embeddings.py swirling_cluster --assignment fixed
    
#     # Specific configurations
#     python generate_lot_embeddings.py geodesic_transport --N 500 --kinds gaussian_iso --fracs 100
    
#     # Diagnose existing embeddings
#     python generate_lot_embeddings.py diagnose lot_maps/geodesic_transport_N_500/gaussian_iso/frac100
#         """
#     )
    
#     parser.add_argument("system", help="System name or 'diagnose'")
#     parser.add_argument("--N", type=int, nargs="+", default=[100, 250, 500, 750, 1000, 1500])
#     parser.add_argument("--kinds", type=str, nargs="+", default=None)
#     parser.add_argument("--fracs", type=int, nargs="+", default=None)
#     parser.add_argument("--assignment", choices=["per_frame", "fixed"], default="per_frame",
#                        help="OT assignment mode (default: per_frame)")
#     parser.add_argument("--anchor-frame", type=int, default=0,
#                        help="Anchor frame for fixed assignment (default: 0)")
#     parser.add_argument("--results-root", type=Path, default=Path("results"))
#     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
#     parser.add_argument("--verbose", "-v", action="store_true")
    
#     args = parser.parse_args()
    
#     # Handle diagnose command
#     if args.system == "diagnose":
#         if args.N and len(args.N) == 1:
#             # Hack: first positional after "diagnose" gets parsed into --N
#             lot_dir = Path(str(args.N[0]))
#         else:
#             parser.error("diagnose requires a lot_dir path")
#         diagnose_lot_velocities(lot_dir)
#         return
    
#     # Handle generate
#     if args.system not in ["geodesic_transport", "swirling_cluster", "all"]:
#         # Maybe it's a path for diagnose
#         if Path(args.system).exists():
#             diagnose_lot_velocities(Path(args.system))
#             return
#         parser.error(f"Unknown system: {args.system}")
    
#     systems = ["geodesic_transport", "swirling_cluster"] if args.system == "all" else [args.system]
    
#     for system in systems:
#         generate_lot_embeddings(
#             system=system,
#             N_list=args.N,
#             kinds=args.kinds,
#             fractions=args.fracs,
#             assignment=args.assignment,
#             anchor_frame=args.anchor_frame,
#             results_root=args.results_root,
#             lot_root=args.lot_root,
#             verbose=args.verbose,
#         )


# if __name__ == "__main__":
#     main()








# #!/usr/bin/env python3
# """
# Compute LOT maps and velocities for all (N, reference_type, fraction) combinations.

# Supports two OT assignment modes:
#   - per_frame:  Solve OT(σ → μ_t) independently for each frame t (original behavior)
#   - fixed:      Solve OT(σ → μ_0) once, apply same barycentric weights to all frames

# Fixed-assignment preserves temporal structure for smooth dynamics, enabling
# velocity prediction via reservoir computing.

# Creates:
#     lot_maps/{system}_N_{N}/{kind}/frac{frac}/lot_maps.npy
#     lot_maps/{system}_N_{N}/{kind}/frac{frac}/velocities.npy
#     lot_maps/{system}_N_{N}/{kind}/frac{frac}/reference.npy
#     lot_maps/{system}_N_{N}/{kind}/frac{frac}/metadata.json

# Usage:
#     # Fixed-assignment (recommended for forecasting)
#     python generate_lot_embeddings.py geodesic_transport --assignment fixed
    
#     # Per-frame (original behavior)
#     python generate_lot_embeddings.py geodesic_transport --assignment per_frame
    
#     # All systems
#     python generate_lot_embeddings.py all --assignment fixed
# """

# import numpy as np
# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Tuple, Optional
# from dataclasses import dataclass, asdict


# # ═══════════════════════════════════════════════════════════════
# # REFERENCE GENERATORS
# # ═══════════════════════════════════════════════════════════════

# def make_circle(R, radius=1.0, center=(0, 0)):
#     """Generate R points uniformly on a circle."""
#     theta = np.linspace(0, 2 * np.pi, R, endpoint=False)
#     x = radius * np.cos(theta) + center[0]
#     y = radius * np.sin(theta) + center[1]
#     return np.stack((x, y), axis=1)


# def make_triangle_uniform(R, scale=1.5, center=(0, 0)):
#     """Generate R points uniformly on a triangle boundary."""
#     # Triangle vertices
#     v0 = np.array([0, scale]) + np.array(center)
#     v1 = np.array([-scale * np.sqrt(3)/2, -scale/2]) + np.array(center)
#     v2 = np.array([scale * np.sqrt(3)/2, -scale/2]) + np.array(center)
    
#     # Perimeter lengths
#     edges = [(v0, v1), (v1, v2), (v2, v0)]
#     lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
#     total_perimeter = sum(lengths)
    
#     # Points per edge proportional to length
#     points = []
#     for (va, vb), length in zip(edges, lengths):
#         n_edge = max(1, int(round(R * length / total_perimeter)))
#         t = np.linspace(0, 1, n_edge, endpoint=False)
#         edge_points = va + np.outer(t, vb - va)
#         points.append(edge_points)
    
#     points = np.vstack(points)
    
#     # Adjust to exactly R points
#     if len(points) > R:
#         idx = np.linspace(0, len(points)-1, R, dtype=int)
#         points = points[idx]
#     elif len(points) < R:
#         # Repeat some points
#         idx = np.random.default_rng(42).choice(len(points), R, replace=True)
#         points = points[idx]
    
#     return points


# def make_uniform_square(R, side=2.0, center=(0, 0)):
#     """Generate R points uniformly in a square."""
#     rng = np.random.default_rng(42)
#     points = rng.uniform(-side/2, side/2, size=(R, 2))
#     points += np.array(center)
#     return points


# def make_gaussian_iso(R, std=0.5, center=(0, 0)):
#     """Generate R points from isotropic Gaussian."""
#     rng = np.random.default_rng(42)
#     points = rng.normal(0, std, size=(R, 2))
#     points += np.array(center)
#     return points


# def generate_reference(kind: str, R: int, trajectory: np.ndarray = None) -> np.ndarray:
#     """
#     Generate reference measure of specified kind with R points.
    
#     Args:
#         kind: Reference type name
#         R: Number of reference points
#         trajectory: Full trajectory (needed for snapshot references)
    
#     Returns:
#         Reference points of shape (R, 2)
#     """
#     if kind == "clean_circle_small":
#         return make_circle(R, radius=0.5)
    
#     elif kind == "clean_circle_large":
#         return make_circle(R, radius=1.5)
    
#     elif kind == "clean_circle":
#         return make_circle(R, radius=1.0)
    
#     elif kind == "clean_triangle":
#         return make_triangle_uniform(R, scale=1.5)
    
#     elif kind == "uniform_square":
#         return make_uniform_square(R, side=2.0)
    
#     elif kind == "gaussian_iso":
#         return make_gaussian_iso(R, std=0.5)
    
#     elif kind == "snapshot_begin":
#         if trajectory is None:
#             raise ValueError("trajectory required for snapshot references")
#         # Subsample from first frame
#         rng = np.random.default_rng(42)
#         N = trajectory.shape[1]
#         idx = rng.choice(N, size=R, replace=(R > N))
#         return trajectory[0, idx].copy()
    
#     elif kind == "snapshot_middle":
#         if trajectory is None:
#             raise ValueError("trajectory required for snapshot references")
#         T = trajectory.shape[0]
#         mid = T // 2
#         rng = np.random.default_rng(42)
#         N = trajectory.shape[1]
#         idx = rng.choice(N, size=R, replace=(R > N))
#         return trajectory[mid, idx].copy()
    
#     elif kind == "snapshot_end":
#         if trajectory is None:
#             raise ValueError("trajectory required for snapshot references")
#         rng = np.random.default_rng(42)
#         N = trajectory.shape[1]
#         idx = rng.choice(N, size=R, replace=(R > N))
#         return trajectory[-1, idx].copy()
    
#     else:
#         raise ValueError(f"Unknown reference kind: {kind}")


# # ═══════════════════════════════════════════════════════════════
# # LOT COMPUTATION
# # ═══════════════════════════════════════════════════════════════

# def compute_ot_plan(source: np.ndarray, target: np.ndarray) -> np.ndarray:
#     """
#     Compute optimal transport plan from source to target.
    
#     Args:
#         source: Reference measure (R, 2)
#         target: Target measure (N, 2)
    
#     Returns:
#         transport_plan: (R, N) optimal coupling
#     """
#     import ot
    
#     R = source.shape[0]
#     N = target.shape[0]
    
#     # Uniform weights
#     a = np.ones(R) / R
#     b = np.ones(N) / N
    
#     # Cost matrix (squared Euclidean)
#     M = ot.dist(source, target, metric='sqeuclidean')
    
#     # Solve OT
#     transport_plan = ot.emd(a, b, M)  # (R, N)
    
#     return transport_plan


# def plan_to_barycentric_weights(plan: np.ndarray) -> np.ndarray:
#     """
#     Convert transport plan to normalized barycentric weights.
    
#     Args:
#         plan: (R, N) transport plan
    
#     Returns:
#         weights: (R, N) where each row sums to 1
#     """
#     row_sums = plan.sum(axis=1, keepdims=True)
#     row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
#     return plan / row_sums


# def apply_barycentric_projection(weights: np.ndarray, targets: np.ndarray) -> np.ndarray:
#     """
#     Apply barycentric projection: weighted average of target points.
    
#     Args:
#         weights: (R, N) barycentric weights
#         targets: (N, 2) target points
    
#     Returns:
#         projected: (R, 2) barycentric projections
#     """
#     return weights @ targets


# def compute_ot_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
#     """
#     Compute optimal transport map from source to target (per-frame version).
    
#     Args:
#         source: Reference measure (R, 2)
#         target: Target measure (N, 2)
    
#     Returns:
#         Mapped points (R, 2) - the barycentric projection
#     """
#     plan = compute_ot_plan(source, target)
#     weights = plan_to_barycentric_weights(plan)
#     return apply_barycentric_projection(weights, target)


# def compute_lot_embedding_perframe(
#     trajectory: np.ndarray,
#     reference: np.ndarray,
#     verbose: bool = False
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute LOT maps using per-frame OT (original method).
    
#     Solves OT(σ → μ_t) independently for each frame t.
#     WARNING: This destroys temporal structure!
    
#     Args:
#         trajectory: Particle positions (T, N, 2)
#         reference: Reference measure (R, 2)
#         verbose: Print progress
    
#     Returns:
#         lot_maps: (T, R, 2) - OT maps at each timestep
#         velocities: (T-1, R, 2) - temporal differences
#     """
#     T = trajectory.shape[0]
#     R = reference.shape[0]
    
#     lot_maps = np.zeros((T, R, 2), dtype=np.float32)
    
#     for t in range(T):
#         if verbose and t % 50 == 0:
#             print(f"  [per-frame] Computing LOT map t={t}/{T}...")
        
#         target = trajectory[t]  # (N, 2)
#         lot_maps[t] = compute_ot_map(reference, target)
    
#     # Compute velocities
#     velocities = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, 2)
    
#     return lot_maps, velocities


# def compute_lot_embedding_fixed(
#     trajectory: np.ndarray,
#     reference: np.ndarray,
#     anchor_frame: int = 0,
#     verbose: bool = False
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute LOT maps using fixed-assignment OT.
    
#     Solves OT(σ → μ_anchor) ONCE, then applies the same barycentric weights
#     to all frames. This preserves temporal structure!
    
#     Args:
#         trajectory: Particle positions (T, N, 2)
#         reference: Reference measure (R, 2)
#         anchor_frame: Which frame to compute OT assignment from (default: 0)
#         verbose: Print progress
    
#     Returns:
#         lot_maps: (T, R, 2) - OT maps at each timestep
#         velocities: (T-1, R, 2) - temporal differences
#         barycentric_weights: (R, N) - fixed weights used for all frames
#     """
#     T, N, d = trajectory.shape
#     R = reference.shape[0]
    
#     # Compute OT plan ONCE at anchor frame
#     anchor_target = trajectory[anchor_frame]
#     if verbose:
#         print(f"  [fixed] Computing OT plan at anchor frame {anchor_frame}...")
    
#     plan = compute_ot_plan(reference, anchor_target)
#     barycentric_weights = plan_to_barycentric_weights(plan)
    
#     if verbose:
#         sparsity = (plan > 1e-10).sum()
#         print(f"  [fixed] Plan sparsity: {sparsity}/{plan.size} nonzero")
    
#     # Apply same weights to ALL frames
#     lot_maps = np.zeros((T, R, d), dtype=np.float32)
    
#     for t in range(T):
#         if verbose and t % 100 == 0:
#             print(f"  [fixed] Applying projection t={t}/{T}...")
#         lot_maps[t] = apply_barycentric_projection(barycentric_weights, trajectory[t])
    
#     # Compute velocities
#     velocities = lot_maps[1:] - lot_maps[:-1]  # (T-1, R, 2)
    
#     return lot_maps, velocities, barycentric_weights


# def compute_lot_embedding(
#     trajectory: np.ndarray,
#     reference: np.ndarray,
#     assignment: str = "fixed",
#     anchor_frame: int = 0,
#     verbose: bool = False
# ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#     """
#     Compute LOT maps and velocities for a trajectory.
    
#     Args:
#         trajectory: Particle positions (T, N, 2)
#         reference: Reference measure (R, 2)
#         assignment: "fixed" (recommended) or "per_frame"
#         anchor_frame: For fixed assignment, which frame to anchor to
#         verbose: Print progress
    
#     Returns:
#         lot_maps: (T, R, 2) - OT maps at each timestep
#         velocities: (T-1, R, 2) - temporal differences
#         barycentric_weights: (R, N) or None - weights if fixed assignment
#     """
#     if assignment == "fixed":
#         return compute_lot_embedding_fixed(trajectory, reference, anchor_frame, verbose)
#     elif assignment == "per_frame":
#         lot_maps, velocities = compute_lot_embedding_perframe(trajectory, reference, verbose)
#         return lot_maps, velocities, None
#     else:
#         raise ValueError(f"Unknown assignment mode: {assignment}. Use 'fixed' or 'per_frame'")


# # ═══════════════════════════════════════════════════════════════
# # MAIN GENERATION LOGIC
# # ═══════════════════════════════════════════════════════════════

# DEFAULT_KINDS = [
#     "clean_circle_small",
#     "clean_circle_large", 
#     "uniform_square",
#     "gaussian_iso",
#     "snapshot_begin",
#     "snapshot_middle",
#     "snapshot_end",
# ]

# # System-specific reference kinds
# GEODESIC_KINDS = [
#     "clean_circle",       # Circle matching source (radius=1.0)
#     "clean_triangle",     # Triangle matching target (scale=1.5)
#     "clean_circle_small",
#     "clean_circle_large",
#     "uniform_square",
#     "gaussian_iso",
#     "snapshot_begin",
#     "snapshot_middle",
#     "snapshot_end",
# ]

# SWIRLING_KINDS = [
#     "clean_circle_small",
#     "clean_circle_large",
#     "uniform_square",
#     "gaussian_iso",
#     "snapshot_begin",
#     "snapshot_middle",
#     "snapshot_end",
# ]

# DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


# def get_default_kinds(system: str) -> List[str]:
#     """Get system-specific default reference kinds."""
#     if "geodesic" in system.lower():
#         return GEODESIC_KINDS
#     elif "swirling" in system.lower():
#         return SWIRLING_KINDS
#     else:
#         return DEFAULT_KINDS


# def generate_lot_embeddings(
#     system: str,
#     N_list: List[int] = [100, 250, 500, 750, 1000, 1500],
#     kinds: List[str] = None,
#     fractions: List[int] = None,
#     assignment: str = "fixed",
#     anchor_frame: int = 0,
#     results_root: Path = Path("results"),
#     lot_root: Path = Path("lot_maps"),
#     verbose: bool = False,
# ):
#     """
#     Generate LOT embeddings for all configurations.
    
#     Args:
#         system: System name ("geodesic_transport" or "swirling_cluster")
#         N_list: List of particle counts
#         kinds: Reference types (default: system-specific)
#         fractions: Fraction percentages
#         assignment: "fixed" (recommended) or "per_frame"
#         anchor_frame: For fixed assignment, which frame to anchor to
#         results_root: Directory containing trajectories
#         lot_root: Output directory for LOT maps
#         verbose: Print detailed progress
#     """
#     kinds = kinds or get_default_kinds(system)
#     fractions = fractions or DEFAULT_FRACTIONS
    
#     results_root = Path(results_root)
#     lot_root = Path(lot_root)
    
#     print(f"\n{'='*60}")
#     print(f" Generating LOT embeddings: {system}")
#     print(f" Assignment mode: {assignment}")
#     print(f"{'='*60}")
    
#     total_runs = len(N_list) * len(kinds) * len(fractions)
#     run_count = 0
    
#     for N in N_list:
#         # Load trajectory
#         traj_path = results_root / f"{system}_N_{N}" / "trajectory_jittered.npy"
#         if not traj_path.exists():
#             traj_path = results_root / f"{system}_N_{N}" / "trajectory.npy"
        
#         if not traj_path.exists():
#             print(f"[MISS] {traj_path}")
#             continue
        
#         trajectory = np.load(traj_path)
#         T, N_traj, d = trajectory.shape
#         print(f"\n[INFO] N={N}: trajectory {trajectory.shape}")
        
#         for kind in kinds:
#             for frac in fractions:
#                 run_count += 1
                
#                 # Compute R (reference size) as fraction of N
#                 R = max(1, int(round(N * frac / 100)))
                
#                 # Generate reference
#                 try:
#                     reference = generate_reference(kind, R, trajectory)
#                 except ValueError as e:
#                     print(f"[SKIP] N={N} {kind} {frac}%: {e}")
#                     continue
                
#                 print(f"[RUN] N={N} {kind} {frac}% -> R={R}, T={T}, d={d}, assignment={assignment}")
                
#                 # Compute LOT embedding
#                 lot_maps, velocities, bary_weights = compute_lot_embedding(
#                     trajectory, reference, 
#                     assignment=assignment, 
#                     anchor_frame=anchor_frame,
#                     verbose=verbose
#                 )
                
#                 # Velocity statistics
#                 vel_std = velocities.std()
#                 vel_mag = np.linalg.norm(velocities, axis=-1).mean()
                
#                 # Save
#                 out_dir = lot_root / f"{system}_N_{N}" / kind / f"frac{frac}"
#                 out_dir.mkdir(parents=True, exist_ok=True)
                
#                 np.save(out_dir / "lot_maps.npy", lot_maps)
#                 np.save(out_dir / "velocities.npy", velocities)
#                 np.save(out_dir / "reference.npy", reference)
                
#                 if bary_weights is not None:
#                     np.save(out_dir / "barycentric_weights.npy", bary_weights)
                
#                 # Save metadata
#                 metadata = {
#                     "system": system,
#                     "N": N,
#                     "R": R,
#                     "T": T,
#                     "d": d,
#                     "kind": kind,
#                     "frac_pct": frac,
#                     "assignment": assignment,
#                     "anchor_frame": anchor_frame if assignment == "fixed" else None,
#                     "velocity_std": float(vel_std),
#                     "velocity_mean_magnitude": float(vel_mag),
#                     "trajectory_source": str(traj_path),
#                 }
                
#                 with open(out_dir / "metadata.json", "w") as f:
#                     json.dump(metadata, f, indent=2)
                
#                 print(f"[SAVED] {out_dir}/lot_maps.npy | vel_std={vel_std:.6f}")
    
#     print(f"\n[DONE] Generated {run_count} LOT embeddings for {system}")


# # ═══════════════════════════════════════════════════════════════
# # DIAGNOSTICS
# # ═══════════════════════════════════════════════════════════════

# def diagnose_lot_velocities(lot_dir: Path):
#     """
#     Diagnose velocity structure for a LOT embedding.
    
#     Args:
#         lot_dir: Directory containing lot_maps.npy and velocities.npy
#     """
#     lot_dir = Path(lot_dir)
    
#     velocities = np.load(lot_dir / "velocities.npy")
#     T, R, d = velocities.shape
    
#     print(f"\nDiagnostics for {lot_dir}")
#     print(f"  Velocities shape: {velocities.shape}")
    
#     # Smoothness
#     vel_std = velocities.std()
#     vel_diffs = np.diff(velocities, axis=0)
#     change_rate = np.linalg.norm(vel_diffs, axis=-1).mean()
#     smoothness = vel_std / change_rate if change_rate > 0 else float('inf')
    
#     print(f"  Velocity std: {vel_std:.6f}")
#     print(f"  Change rate: {change_rate:.6f}")
#     print(f"  Smoothness ratio: {smoothness:.2f} {'✓' if smoothness > 1 else '⚠️'}")
    
#     # Autocorrelation
#     vel_flat = velocities.reshape(T, -1)
#     print(f"  Autocorrelation:")
#     for lag in [5, 10, 25]:
#         if lag < T:
#             corr = np.corrcoef(vel_flat[:-lag].flatten(), vel_flat[lag:].flatten())[0, 1]
#             print(f"    lag {lag}: {corr:.4f}")
    
#     # Check metadata
#     meta_path = lot_dir / "metadata.json"
#     if meta_path.exists():
#         with open(meta_path) as f:
#             meta = json.load(f)
#         print(f"  Assignment: {meta.get('assignment', 'unknown')}")


# # ═══════════════════════════════════════════════════════════════
# # CLI
# # ═══════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate LOT embeddings with fixed or per-frame assignment",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#     # Fixed-assignment (recommended for forecasting)
#     python generate_lot_embeddings.py geodesic_transport --assignment fixed
    
#     # Per-frame assignment (original behavior)
#     python generate_lot_embeddings.py swirling_cluster --assignment per_frame
    
#     # Specific configurations
#     python generate_lot_embeddings.py geodesic_transport --N 500 --kinds gaussian_iso --fracs 100
    
#     # Diagnose existing embeddings
#     python generate_lot_embeddings.py diagnose lot_maps/geodesic_transport_N_500/gaussian_iso/frac100
#         """
#     )
    
#     parser.add_argument("system", help="System name or 'diagnose'")
#     parser.add_argument("--N", type=int, nargs="+", default=[100, 250, 500, 750, 1000, 1500])
#     parser.add_argument("--kinds", type=str, nargs="+", default=None)
#     parser.add_argument("--fracs", type=int, nargs="+", default=None)
#     parser.add_argument("--assignment", choices=["fixed", "per_frame"], default="fixed",
#                        help="OT assignment mode (default: fixed)")
#     parser.add_argument("--anchor-frame", type=int, default=0,
#                        help="Anchor frame for fixed assignment (default: 0)")
#     parser.add_argument("--results-root", type=Path, default=Path("results"))
#     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
#     parser.add_argument("--verbose", "-v", action="store_true")
    
#     args = parser.parse_args()
    
#     # Handle diagnose command
#     if args.system == "diagnose":
#         if args.N and len(args.N) == 1:
#             # Hack: first positional after "diagnose" gets parsed into --N
#             lot_dir = Path(str(args.N[0]))
#         else:
#             parser.error("diagnose requires a lot_dir path")
#         diagnose_lot_velocities(lot_dir)
#         return
    
#     # Handle generate
#     if args.system not in ["geodesic_transport", "swirling_cluster", "all"]:
#         # Maybe it's a path for diagnose
#         if Path(args.system).exists():
#             diagnose_lot_velocities(Path(args.system))
#             return
#         parser.error(f"Unknown system: {args.system}")
    
#     systems = ["geodesic_transport", "swirling_cluster"] if args.system == "all" else [args.system]
    
#     for system in systems:
#         generate_lot_embeddings(
#             system=system,
#             N_list=args.N,
#             kinds=args.kinds,
#             fractions=args.fracs,
#             assignment=args.assignment,
#             anchor_frame=args.anchor_frame,
#             results_root=args.results_root,
#             lot_root=args.lot_root,
#             verbose=args.verbose,
#         )


# if __name__ == "__main__":
#     main()