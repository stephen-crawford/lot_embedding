#!/usr/bin/env python3
"""
Block A — Config & CLI parsing for the LOT–RC evaluator.

Updated for VELOCITY vs POSITIONS comparison:
- VELOCITY: predict LOT velocities, then integrate (velocity_forecast.py)
- POSITIONS: predict LOT maps directly (positions_forecast.py)

This module defines:
- AVAILABLE_BACKENDS: list of supported Δ backends
- DEFAULTS: default hyperparameters
- parse_args(): parses CLI arguments and returns an argparse.Namespace
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# SUPPORTED BACKENDS AND DEFAULTS
# ═══════════════════════════════════════════════════════════════

AVAILABLE_BACKENDS = ["llt_l2", "sinkhorn"]

DEFAULTS = {
    "backend": "llt_l2",      # 'llt_l2' (fast) or 'sinkhorn' (costly; use window)
    "gamma_base": 0.10,       # soft-DTW temperature: gamma = gamma_base * median(Δ)
    "eps_scale": 0.05,        # Sinkhorn ε = eps_scale × median squared spacing on σ
    "sinkhorn_center": None,  # window center index (None → middle)
    "sinkhorn_halfwidth": 6,  # window half-width → default 12×12 window
}


# ═══════════════════════════════════════════════════════════════
# CLI PARSER
# ═══════════════════════════════════════════════════════════════

def parse_args() -> Namespace:
    """
    Parse command-line arguments for the evaluator.
    
    Returns
    -------
    argparse.Namespace with fields:
        velocity_dir: Path to VELOCITY run folder
        positions_dir: Path to POSITIONS run folder
        eval_out: Path to output directory
        backend: one of AVAILABLE_BACKENDS
        gamma_base: float
        eps_scale: float
        sinkhorn_center: int|None
        sinkhorn_halfwidth: int
    """
    p = ArgumentParser(description="LOT–RC evaluator: VELOCITY vs POSITIONS comparison")
    
    # Required paths
    p.add_argument(
        "--velocity-dir",
        type=Path,
        required=True,
        help="Path to VELOCITY run directory (from velocity_forecast.py)",
    )
    p.add_argument(
        "--positions-dir",
        type=Path,
        required=True,
        help="Path to POSITIONS run directory (from positions_forecast.py)",
    )
    p.add_argument(
        "--eval-out",
        type=Path,
        required=True,
        help="Directory to write outputs (Δ, plots, metrics, paths).",
    )
    
    # Backend choice
    p.add_argument(
        "--backend",
        choices=AVAILABLE_BACKENDS,
        default=DEFAULTS["backend"],
        help=f"Which Δ backend to use: {AVAILABLE_BACKENDS}.",
    )
    
    # Shared hyperparameters
    p.add_argument(
        "--gamma-base",
        type=float,
        default=DEFAULTS["gamma_base"],
        help="soft-DTW smoothing scale factor (gamma = gamma_base * median(Δ)).",
    )
    p.add_argument(
        "--eps-scale",
        type=float,
        default=DEFAULTS["eps_scale"],
        help="Sinkhorn epsilon heuristic scale (ε = eps_scale × median squared spacing on σ).",
    )
    
    # Sinkhorn window (only used if backend='sinkhorn')
    p.add_argument(
        "--sinkhorn-center",
        type=int,
        default=DEFAULTS["sinkhorn_center"],
        help="Center index for Sinkhorn validation window (None → middle).",
    )
    p.add_argument(
        "--sinkhorn-halfwidth",
        type=int,
        default=DEFAULTS["sinkhorn_halfwidth"],
        help="Half-width for Sinkhorn validation window (6 → 12×12 window).",
    )
    
    args = p.parse_args()
    
    # Basic sanity checks (fail fast)
    for path_name in ("velocity_dir", "positions_dir"):
        path: Path = getattr(args, path_name.replace("-", "_"))
        if not path.exists():
            raise FileNotFoundError(f"--{path_name} not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"--{path_name} is not a directory: {path}")
    
    # Create output directory if needed
    args.eval_out.mkdir(parents=True, exist_ok=True)
    
    return args


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Config from forecast structure
# ═══════════════════════════════════════════════════════════════

def find_run_dirs(
    forecast_root: Path,
    system: str,
    N: int,
    kind: str,
    frac: int,
    reservoir_scale: float = 1.0,
) -> tuple[Path, Path]:
    """
    Find VELOCITY and POSITIONS run directories from standard structure.
    
    Args:
        forecast_root: Root of forecast_output/
        system: "geodesic_transport" or "swirling_cluster"
        N: Particle count
        kind: Reference type (e.g., "gaussian_iso")
        frac: Fraction percentage
        reservoir_scale: Reservoir size scaling
    
    Returns:
        (velocity_dir, positions_dir)
    
    Raises:
        FileNotFoundError if directories don't exist
    """
    forecast_root = Path(forecast_root)
    pct = int(reservoir_scale * 100)
    
    base = forecast_root / f"{system}_N_{N}" / kind / f"frac{frac}"
    
    # Try different naming patterns
    vel_patterns = [
        f"velocity_resFromOrigN_{pct}pct_velocity",
        f"velocity_resFromOrigN_{pct}pct",
    ]
    pos_patterns = [
        f"positions_resFromOrigN_{pct}pct_positions",
        f"positions_resFromOrigN_{pct}pct",
    ]
    
    velocity_dir = None
    positions_dir = None
    
    for pattern in vel_patterns:
        candidate = base / pattern
        if candidate.exists():
            velocity_dir = candidate
            break
    
    for pattern in pos_patterns:
        candidate = base / pattern
        if candidate.exists():
            positions_dir = candidate
            break
    
    if velocity_dir is None:
        raise FileNotFoundError(f"VELOCITY directory not found in {base}")
    if positions_dir is None:
        raise FileNotFoundError(f"POSITIONS directory not found in {base}")
    
    return velocity_dir, positions_dir


if __name__ == "__main__":
    args = parse_args()
    print(f"VELOCITY dir: {args.velocity_dir}")
    print(f"POSITIONS dir: {args.positions_dir}")
    print(f"Output dir: {args.eval_out}")