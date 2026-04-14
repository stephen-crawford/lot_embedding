#!/usr/bin/env python3
"""
Block B — IO: load artifacts from VELOCITY and POSITIONS forecast runs.

Compatible with outputs from:
- velocity_forecast.py (VELOCITY method: predict LOT velocities, integrate)
- positions_forecast.py (POSITIONS method: predict LOT maps directly)

Exposes:
- RunData (dataclass)
- require_run_dir(path)
- load_run_measures(path, name=None) -> RunData
- pick_epsilon_from_sigma(sigma, eps_scale) -> float

Expected files in each run directory (from velocity_forecast.py / positions_forecast.py):
    predicted_measures_X.npy   # (H, R, d) - predicted LOT maps (reconstructed measures)
    true_measures_X.npy        # (H, R, d) - ground truth LOT maps
    measures_w.npy             # (R,) - uniform weights on reference
    reference_sigma_X.npy      # (R, d) - reference points (maps[0] or actual σ)

Optional (saved by forecast scripts):
    pred_displacements.npy     # (H, R, d) - predicted displacement from reference
    true_displacements.npy     # (H, R, d) - true displacement from reference
    predicted_maps.npy         # (H+1, R, d) - full predicted maps including seed
    true_future_maps.npy       # (H+1, R, d) - full true maps including seed
    warm_start_info.npy        # [warm_steps, forecast_steps]
    metadata.json              # run configuration

Method detection:
    The method (VELOCITY vs POSITIONS) is detected from metadata.json if available,
    or from the directory name pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import numpy as np


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Required files (output by both velocity_forecast.py and positions_forecast.py)
REQUIRED_FILES = [
    "predicted_measures_X.npy",
    "true_measures_X.npy",
    "measures_w.npy",
    "reference_sigma_X.npy",
]

# Optional files
OPTIONAL_FILES = [
    "pred_displacements.npy",
    "true_displacements.npy",
    "predicted_maps.npy",
    "true_future_maps.npy",
    "warm_start_info.npy",
    "metadata.json",
    "predicted_velocities.npy",  # VELOCITY method only
    "true_future_vels.npy",       # VELOCITY method only
]


# ═══════════════════════════════════════════════════════════════
# DATA CONTAINER
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunData:
    """
    Container for a single forecast run's data.
    
    Attributes:
        name: Human-readable identifier (e.g., "VELOCITY" or "POSITIONS")
        method: Forecast method ("VELOCITY" or "POSITIONS")
        dir: Path to the run directory
        pred_X: (H, N, d) predicted LOT map images (reconstructed measures)
        true_X: (H, N, d) ground truth LOT map images
        w: (N,) weights on reference σ (uniform, sum=1)
        sigma: (N, d) reference support points
        pred_disp: (H, N, d) optional displacement φ_pred = T_pred - σ
        true_disp: (H, N, d) optional displacement φ_true = T_true - σ
        meta: dict of metadata from metadata.json
        H: forecast horizon (number of predicted timesteps)
        N: number of reference points (called R in forecast code, N here for eval)
        d: spatial dimension (typically 2)
        warm_steps: training warm-up steps (if available)
        forecast_steps: autonomous forecast steps (if available)
    """
    name: str
    method: str  # "VELOCITY" or "POSITIONS"
    dir: Path
    pred_X: np.ndarray            # (H, N, d)
    true_X: np.ndarray            # (H, N, d)
    w: np.ndarray                 # (N,)
    sigma: np.ndarray             # (N, d)
    pred_disp: Optional[np.ndarray] = None   # (H, N, d)
    true_disp: Optional[np.ndarray] = None   # (H, N, d)
    meta: Dict[str, Any] = field(default_factory=dict)
    H: int = 0
    N: int = 0
    d: int = 0
    warm_steps: Optional[int] = None
    forecast_steps: Optional[int] = None


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def _np_load64(path: Path) -> np.ndarray:
    """Load numpy array as float64 (no copy if already float64)."""
    arr = np.load(path)
    return arr.astype(np.float64, copy=False)


def _detect_method(run_dir: Path, meta: Dict[str, Any]) -> str:
    """
    Detect whether this is a VELOCITY or POSITIONS run.
    
    Priority:
        1. metadata.json "method" field
        2. Directory name pattern
        3. Presence of velocity files
        4. Default to "UNKNOWN"
    """
    # From metadata
    if meta.get("method"):
        return str(meta["method"]).upper()
    
    # From directory name
    dir_name = run_dir.name.lower()
    if "velocity" in dir_name:
        return "VELOCITY"
    if "position" in dir_name:
        return "POSITIONS"
    
    # From file presence
    if (run_dir / "predicted_velocities.npy").exists():
        return "VELOCITY"
    
    return "UNKNOWN"


def require_run_dir(run_dir: Path) -> None:
    """
    Validate that required artifacts exist.
    
    Raises FileNotFoundError with details if files are missing.
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {run_dir}")
    
    missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
    
    if missing:
        available = [f.name for f in run_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Missing required files in {run_dir}:\n"
            f"  Missing: {missing}\n"
            f"  Available: {available[:10]}{'...' if len(available) > 10 else ''}"
        )


def pick_epsilon_from_sigma(sigma: np.ndarray, eps_scale: float = 0.05) -> float:
    """
    Heuristic epsilon for Sinkhorn regularization.
    
    Computes: ε = eps_scale × median(||σ_i - σ_j||²) for i < j
    
    Args:
        sigma: (R, d) reference support points
        eps_scale: scaling factor (default 0.05)
    
    Returns:
        Regularization parameter ε
    """
    if sigma.ndim != 2:
        raise ValueError(f"sigma must be (R, d); got shape {sigma.shape}")
    
    R = sigma.shape[0]
    if R < 2:
        return eps_scale * 1.0  # Fallback for degenerate case
    
    # Pairwise squared distances (upper triangle only)
    diffs = sigma[:, None, :] - sigma[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    iu = np.triu_indices(R, k=1)
    med_sq = float(np.median(d2[iu]))
    
    return eps_scale * med_sq


# ═══════════════════════════════════════════════════════════════
# MAIN LOADER
# ═══════════════════════════════════════════════════════════════

def load_run_measures(
    run_dir: str | Path,
    name: Optional[str] = None,
) -> RunData:
    """
    Load a forecast run directory.
    
    Compatible with outputs from velocity_forecast.py and positions_forecast.py.
    
    Args:
        run_dir: Path to the run directory
        name: Optional display name (default: detected from method + dir)
    
    Returns:
        RunData with all loaded arrays and metadata
    
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If array shapes are inconsistent
    """
    run_dir = Path(run_dir)
    require_run_dir(run_dir)
    
    # ── Load required arrays ──
    pred_X = _np_load64(run_dir / "predicted_measures_X.npy")
    true_X = _np_load64(run_dir / "true_measures_X.npy")
    w = _np_load64(run_dir / "measures_w.npy")
    sigma = _np_load64(run_dir / "reference_sigma_X.npy")
    
    # ── Load optional arrays ──
    pred_disp = None
    true_disp = None
    
    if (run_dir / "pred_displacements.npy").exists():
        pred_disp = _np_load64(run_dir / "pred_displacements.npy")
    
    if (run_dir / "true_displacements.npy").exists():
        true_disp = _np_load64(run_dir / "true_displacements.npy")
    
    # ── Load metadata ──
    meta = {}
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            meta = {}
    
    # ── Load warm-up info ──
    warm_steps = None
    forecast_steps = None
    
    warm_info_path = run_dir / "warm_start_info.npy"
    if warm_info_path.exists():
        try:
            warm_info = np.load(warm_info_path)
            warm_steps = int(warm_info[0])
            forecast_steps = int(warm_info[1])
        except (IndexError, ValueError):
            pass
    
    # Fall back to metadata
    if warm_steps is None:
        warm_steps = meta.get("warm_steps")
    if forecast_steps is None:
        forecast_steps = meta.get("forecast_steps")
    
    # ── Shape validation ──
    if pred_X.ndim != 3 or true_X.ndim != 3:
        raise ValueError(
            f"pred_X and true_X must be 3D (H, R, d); "
            f"got {pred_X.shape} and {true_X.shape}"
        )
    
    H1, R1, d1 = pred_X.shape
    H2, R2, d2 = true_X.shape
    
    if (H1, R1, d1) != (H2, R2, d2):
        raise ValueError(
            f"pred_X shape {pred_X.shape} != true_X shape {true_X.shape}"
        )
    
    H, R, d = H1, R1, d1
    
    if sigma.shape != (R, d):
        raise ValueError(
            f"sigma shape {sigma.shape} incompatible with (R, d) = ({R}, {d})"
        )
    
    if w.shape != (R,):
        raise ValueError(
            f"weights shape {w.shape} incompatible with R = {R}"
        )
    
    # ── Normalize weights ──
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError(f"Invalid weights: sum = {w_sum}")
    if abs(w_sum - 1.0) > 1e-10:
        w = w / w_sum
    
    # ── Validate optional displacements ──
    if pred_disp is not None and pred_disp.shape != (H, R, d):
        raise ValueError(
            f"pred_displacements shape {pred_disp.shape} != ({H}, {R}, {d})"
        )
    if true_disp is not None and true_disp.shape != (H, R, d):
        raise ValueError(
            f"true_displacements shape {true_disp.shape} != ({H}, {R}, {d})"
        )
    
    # ── Detect method ──
    method = _detect_method(run_dir, meta)
    
    # ── Build name ──
    if name is None:
        name = f"{method}:{run_dir.name}"
    
    # ── Construct RunData ──
    rd = RunData(
        name=name,
        method=method,
        dir=run_dir,
        pred_X=pred_X,
        true_X=true_X,
        w=w,
        sigma=sigma,
        pred_disp=pred_disp,
        true_disp=true_disp,
        meta=meta,
        H=H,
        N=R,  # R in forecast code = N in eval code (number of reference points)
        d=d,
        warm_steps=warm_steps,
        forecast_steps=forecast_steps,
    )
    
    # ── Warnings ──
    if H < 2:
        print(f"[WARN] H={H} in {run_dir} — very short forecast horizon")
    
    print(
        f"[load] {rd.name}: method={rd.method} H={rd.H} N={rd.N} d={rd.d} "
        f"warm={rd.warm_steps} forecast={rd.forecast_steps}"
    )
    
    return rd


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: Load matched VELOCITY and POSITIONS runs
# ═══════════════════════════════════════════════════════════════

def find_paired_runs(
    forecast_root: Path,
    system: str,
    N: int,
    kind: str,
    frac: int,
    reservoir_scale: float = 1.0,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find matched VELOCITY and POSITIONS run directories.
    
    Args:
        forecast_root: Root of forecast_output/
        system: "geodesic_transport" or "swirling_cluster"
        N: particle count
        kind: reference type (e.g., "gaussian_iso")
        frac: fraction percentage
        reservoir_scale: reservoir size scaling
    
    Returns:
        (velocity_dir, positions_dir) - either may be None if not found
    """
    forecast_root = Path(forecast_root)
    pct = int(reservoir_scale * 100)
    
    base = forecast_root / f"{system}_N_{N}" / kind / f"frac{frac}"
    
    # Look for velocity and positions directories
    vel_patterns = [
        f"velocity_resFromOrigN_{pct}pct_velocity",
        f"velocity_resFromOrigN_{pct}pct",
    ]
    pos_patterns = [
        f"positions_resFromOrigN_{pct}pct_positions",
        f"positions_resFromOrigN_{pct}pct",
    ]
    
    vel_dir = None
    pos_dir = None
    
    for pattern in vel_patterns:
        candidate = base / pattern
        if candidate.exists():
            vel_dir = candidate
            break
    
    for pattern in pos_patterns:
        candidate = base / pattern
        if candidate.exists():
            pos_dir = candidate
            break
    
    return vel_dir, pos_dir


def load_paired_runs(
    forecast_root: Path,
    system: str,
    N: int,
    kind: str,
    frac: int,
    reservoir_scale: float = 1.0,
) -> Tuple[Optional[RunData], Optional[RunData]]:
    """
    Load matched VELOCITY and POSITIONS runs.
    
    Returns:
        (velocity_run, positions_run) - either may be None if not found
    """
    vel_dir, pos_dir = find_paired_runs(
        forecast_root, system, N, kind, frac, reservoir_scale
    )
    
    vel_run = None
    pos_run = None
    
    if vel_dir is not None:
        try:
            vel_run = load_run_measures(vel_dir, name="VELOCITY")
        except FileNotFoundError as e:
            print(f"[WARN] Could not load VELOCITY run: {e}")
    
    if pos_dir is not None:
        try:
            pos_run = load_run_measures(pos_dir, name="POSITIONS")
        except FileNotFoundError as e:
            print(f"[WARN] Could not load POSITIONS run: {e}")
    
    return vel_run, pos_run