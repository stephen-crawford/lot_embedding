#!/usr/bin/env python3
"""
POSITIONS Forecasting: Predict LOT transport maps directly.

Predicts future LOT transport maps (positions) directly without the 
velocity formulation. This is the baseline comparison for VELOCITY.

PARADIGM:
    1. Compute LOT embeddings: T_t = T_σ^{μ_t}
    2. Train reservoir: T_t → T_{t+1} (direct map prediction)
    3. Autonomous rollout: feed predicted maps back as input
    4. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

KEY CHARACTERISTICS:
    - Predicts transport maps (positions) directly
    - Non-stationary target: maps drift further from reference over time
    - Errors compound through autonomous rollout (exponential growth)
    - Baseline for comparison with VELOCITY approach

Compare to VELOCITY (velocity_forecast.py):
    - VELOCITY predicts v_t = T_{t+1} - T_t, then integrates
    - More stationary target, errors accumulate gradually (linear growth)

Supports both geodesic_transport and swirling_cluster systems.
Both systems use 7 reference types.

Usage:
    from positions_forecast import run_grid, SystemConfig
    
    cfg = SystemConfig(
        name="geodesic_transport",
        N_list=[500, 1000],
        reservoir_scales=[1.0, 2.0],
    )
    results = run_grid(cfg)
    
CLI:
    python positions_forecast.py geodesic_transport
    python positions_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
"""

from __future__ import annotations
import os
import time
import json
import csv
import shutil
import platform
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Import warm-up utilities
try:
    from .warmup_utils import compute_warm_steps_for_positions, validate_warm_steps
except ImportError:
    from warmup_utils import compute_warm_steps_for_positions, validate_warm_steps

try:
    from forecast_backend import create_reservoir
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from forecast_backend import create_reservoir

try:
    import torch as _torch
except ImportError:
    _torch = None


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
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

DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


@dataclass
class SystemConfig:
    """Configuration for POSITIONS forecasting experiments."""
    
    name: str  # "geodesic_transport" or "swirling_cluster"
    
    # Grid parameters
    N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
    reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
    # LOT reference configuration
    kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
    fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
    # Warm-up configuration (now uses system defaults from warmup_utils)
    cycle_length: Optional[int] = None  # Override system default if set
    
    # RC hyperparameters — paper defaults for RAW/positions (§5.2–5.3):
    #   spectral_radius=0.7, leak_rate=0.7, ridge=1e-4
    spectral_radius: float = 0.7
    input_scaling: float = 0.1
    leak_rate: float = 0.7
    ridge_param: float = 1e-4
    random_seed: int = 42

    # Hyperparameter sweep configuration
    hyperparameter_sweep: bool = False  # If True, sweep over lists below and pick best
    spectral_radius_list: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    leak_rate_list: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    ridge_param_list: List[float] = field(default_factory=lambda: [1e-4, 1e-2])

    # Paths
    results_root: Path = field(default_factory=lambda: Path("results"))
    forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
    lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
    # Save options
    save_dtype: type = np.float32
    save_maps: bool = True
    
    run_tag: str = "positions"

    # Boundary handling for predictions (prevents particles drifting outside domain)
    # "none": no boundary handling (particles can drift anywhere)
    # "periodic": wrap around [0, 1]^d (for closed systems like tidal, SST)
    # "clip": clamp to [0, 1]^d (for bounded domains)
    boundary_mode: str = "none"

    rc_backend: str = "auto"
    rc_device: str = "auto"

    def __post_init__(self):
        self.results_root = Path(self.results_root)
        self.forecast_root = Path(self.forecast_root)
        self.lot_root = Path(self.lot_root)
    
    def lot_dir(self, N: int, kind: str, frac: int) -> Path:
        return self.lot_root / f"{self.name}_N_{N}" / kind / f"frac{frac}"
    
    def output_dir(self, N: int, kind: str, frac: int, reservoir_scale: float) -> Path:
        pct = int(reservoir_scale * 100)
        tag = f"_{self.run_tag}" if self.run_tag else ""
        return (
            self.forecast_root / f"{self.name}_N_{N}" / kind / f"frac{frac}" /
            f"positions_resFromOrigN_{pct}pct{tag}"
        )


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def env_banner() -> None:
    torch_info = "no torch"
    if _torch is not None:
        cuda = _torch.cuda.is_available()
        torch_info = f"torch {_torch.__version__} cuda={cuda}"
    print(
        f"[env] Python {platform.python_version()} | NumPy {np.__version__} | {torch_info} | "
        f"OMP={os.environ.get('OMP_NUM_THREADS', '?')} MKL={os.environ.get('MKL_NUM_THREADS', '?')}"
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Atomic save to avoid corrupt partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    np.save(tmp, arr)
    tmp_actual = tmp if tmp.exists() else tmp.parent / (tmp.name + ".npy")
    os.replace(tmp_actual, path)


def minimal_run_exists(out_dir: Path) -> bool:
    """Check if essential outputs exist."""
    return all((out_dir / f).exists() for f in [
        "predicted_measures_X.npy", "true_measures_X.npy", "measures_w.npy"
    ])


def clean_incomplete_run(out_dir: Path) -> None:
    if out_dir.exists() and not minimal_run_exists(out_dir):
        print(f"[CLEAN] Removing incomplete: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)


def append_csv_row(path: Path, header: List[str], row: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def est_bytes(arr: np.ndarray, dtype=None) -> int:
    dt = np.dtype(dtype) if dtype is not None else arr.dtype
    return int(arr.size * dt.itemsize)


def preflight_space(dest_dir: Path, bytes_needed: int, margin: float = 1.15) -> Tuple[bool, Dict]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(dest_dir)
    need = int(bytes_needed * margin)
    return free >= need, {"free": free, "need": need, "where": str(dest_dir)}


def choose_save_plan(
    out_dir: Path,
    pred_maps: np.ndarray,
    true_maps: np.ndarray,
    pred_meas: np.ndarray,
    true_meas: np.ndarray,
    weights: np.ndarray,
    cfg: SystemConfig
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Choose what to save based on available disk space."""
    dtype = cfg.save_dtype
    
    def estimate(save_maps: bool) -> int:
        total = 0
        if save_maps:
            total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
        total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
        total += est_bytes(weights, np.float32)
        total += 2 * est_bytes(pred_meas, dtype)
        return total
    
    ok, info = preflight_space(out_dir, estimate(True))
    if ok:
        return {"save_maps": True, "save_meas": True}, info
    
    ok, info = preflight_space(out_dir, estimate(False))
    if ok:
        return {"save_maps": False, "save_meas": True}, info
    
    raise OSError(f"Insufficient disk space at {out_dir}")


# ═══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_single(
    lot_dir: Path,
    out_dir: Path,
    N: int,
    kind: str,
    frac: int,
    reservoir_scale: float,
    cfg: SystemConfig,
    skip_existing: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run POSITIONS forecast for single configuration.
    
    POSITIONS paradigm: predict maps directly T_t → T_{t+1}.
    This has non-stationary targets and exponential error growth.
    
    Returns dict with timing/metrics, or None if skipped/missing.
    """
    if not (lot_dir / "lot_maps.npy").exists():
        print(f"[MISS] {lot_dir}")
        return None
    
    if skip_existing and minimal_run_exists(out_dir):
        print(f"[SKIP] {out_dir}")
        return None
      
    clean_incomplete_run(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from rc_computer import ReservoirConfig, InitScheme

    # ── LOAD ──
    t0_load = time.perf_counter()
    maps = np.load(lot_dir / "lot_maps.npy")  # (T, R, d)
    time_load = time.perf_counter() - t0_load
    
    T, R, d = maps.shape
    # Support both 2D (d=2) and 3D (d=3) data
    assert d in (2, 3), f"Expected d=2 or d=3, got d={d}"
    
    # ── PREP ──
    t0_prep = time.perf_counter()
    
    # POSITIONS paradigm: predict map[t+1] from map[t]
    # This is the direct position prediction approach
    map_flat = maps.reshape(T, R * d)
    input_seq = map_flat[:-1]   # T_0, ..., T_{T-2}
    target_seq = map_flat[1:]   # T_1, ..., T_{T-1}
    max_steps = input_seq.shape[0]
    
    # Use warmup_utils for correct warm-up calculation
    warm_steps = compute_warm_steps_for_positions(
        T=T,
        system_name=cfg.name,
        cycle_length=cfg.cycle_length,
        lot_dir=lot_dir,
    )
    
    train_input = input_seq[:warm_steps]
    train_target = target_seq[:warm_steps]
    
    time_prep = time.perf_counter() - t0_prep
    
    # ── TRAIN ──
    t0_train = time.perf_counter()
    reservoir_size = max(1, int(round(reservoir_scale * N)))

    # Forecast horizon (needed for sweep evaluation)
    forecast_steps = T - 1 - warm_steps
    true_future_maps = maps[warm_steps + 1:T]

    # ── HYPERPARAMETER SWEEP ──
    # Sweep over hyperparameter combinations like the tornado analysis
    if cfg.hyperparameter_sweep:
        hp_source = "sweep"
        best_map_rmse = float('inf')
        best_hp = None
        best_rc = None
        best_pred_maps = None
        best_train_rmse = None
        best_backend_used = "numpy"
        best_dev_str = "cpu"

        sweep_results = []
        for sr in cfg.spectral_radius_list:
            for leak in cfg.leak_rate_list:
                for ridge in cfg.ridge_param_list:
                    rc_config = ReservoirConfig(
                        input_size=R * d,
                        reservoir_size=reservoir_size,
                        output_size=R * d,
                        spectral_radius=sr,
                        input_scaling=cfg.input_scaling,
                        leak_rate=leak,
                        ridge_param=ridge,
                        activation="tanh",
                        init_scheme=InitScheme.SPARSE,
                        random_seed=cfg.random_seed,
                    )
                    rc, backend_used, dev_str = create_reservoir(
                        rc_config, cfg.rc_backend, cfg.rc_device
                    )

                    states = rc.run(train_input)
                    rc.train(states, train_target)

                    train_pred = rc.predict(states)
                    tr_rmse = rmse(train_pred, train_target)

                    _, all_pred_flat = rc.run_autonomous(train_input, forecast_steps + 1)
                    pred_maps_flat = all_pred_flat[1:]
                    pred_maps_candidate = pred_maps_flat.reshape(forecast_steps, R, d)

                    m_rmse = rmse(pred_maps_candidate, true_future_maps)
                    sweep_results.append((sr, leak, ridge, m_rmse, tr_rmse))

                    if m_rmse < best_map_rmse:
                        best_map_rmse = m_rmse
                        best_hp = (sr, leak, ridge)
                        best_rc = rc
                        best_pred_maps = pred_maps_candidate
                        best_train_rmse = tr_rmse
                        best_backend_used = backend_used
                        best_dev_str = dev_str

        spectral_radius, leak_rate, ridge_param = best_hp
        input_scaling = cfg.input_scaling
        rc = best_rc
        pred_maps = best_pred_maps
        train_rmse = best_train_rmse
        map_rmse = best_map_rmse
        backend_used, dev_str = best_backend_used, best_dev_str

        # Log sweep results
        sweep_results.sort(key=lambda x: x[3])  # Sort by map_rmse
        print(f"  [SWEEP] Tested {len(sweep_results)} configs, best: sr={spectral_radius}, leak={leak_rate}, ridge={ridge_param}")
    else:
        hp_source = "config"
        spectral_radius = cfg.spectral_radius
        input_scaling = cfg.input_scaling
        leak_rate = cfg.leak_rate
        ridge_param = cfg.ridge_param

        rc_config = ReservoirConfig(
            input_size=R * d,
            reservoir_size=reservoir_size,
            output_size=R * d,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            ridge_param=ridge_param,
            activation="tanh",
            init_scheme=InitScheme.SPARSE,
            random_seed=cfg.random_seed,
        )
        rc, backend_used, dev_str = create_reservoir(
            rc_config, cfg.rc_backend, cfg.rc_device
        )

        states = rc.run(train_input)
        rc.train(states, train_target)

        # Training RMSE
        train_pred = rc.predict(states)
        train_rmse = rmse(train_pred, train_target)

        # Run autonomous for forecast_steps + 1 (includes T[warm_steps])
        _, all_pred_flat = rc.run_autonomous(train_input, forecast_steps + 1)

        # Skip first prediction to align with VELOCITY
        # Output: T[warm_steps+1], ..., T[T-1]
        pred_maps_flat = all_pred_flat[1:]
        pred_maps = pred_maps_flat.reshape(forecast_steps, R, d)

        # ── METRICS ──
        map_rmse = rmse(pred_maps, true_future_maps)

    # Apply boundary handling to keep predictions in domain
    if cfg.boundary_mode == "periodic":
        pred_maps = pred_maps % 1.0  # Wrap to [0, 1)^d
    elif cfg.boundary_mode == "clip":
        pred_maps = np.clip(pred_maps, 0.0, 1.0)  # Clamp to [0, 1]^d
    # "none": no boundary handling (default, allows particles to drift)

    time_train = time.perf_counter() - t0_train
    time_rollout = 0.0  # Rollout is included in time_train for POSITIONS
    
    # ── PREPARE OUTPUT ──
    pred_measures_X = pred_maps
    true_measures_X = true_future_maps
    weights = np.full(R, 1.0 / R, dtype=np.float32)
    
    # Load actual reference σ (if available), otherwise fall back to maps[0]
    # CRITICAL: Displacements should be φ_t = T_t(σ) - σ, NOT T_t(σ) - T_0(σ)
    ref_path = lot_dir / "reference.npy"
    if ref_path.exists():
        sigma_points = np.load(ref_path).astype(maps.dtype)
        using_actual_reference = True
    else:
        # Fallback for legacy LOT embeddings without reference.npy
        sigma_points = maps[0].copy()
        using_actual_reference = False
        print(f"[WARN] {lot_dir}/reference.npy not found, using maps[0] as σ")
    
    true_disp = true_measures_X - sigma_points
    pred_disp = pred_measures_X - sigma_points
    
    plan, space_info = choose_save_plan(
        out_dir, pred_maps, true_future_maps,
        pred_measures_X, true_measures_X, weights, cfg
    )
    
    # ── SAVE ──
    t0_save = time.perf_counter()
    
    atomic_save_npy(out_dir / "predicted_measures_X.npy", pred_measures_X.astype(cfg.save_dtype))
    atomic_save_npy(out_dir / "true_measures_X.npy", true_measures_X.astype(cfg.save_dtype))
    atomic_save_npy(out_dir / "measures_w.npy", weights)
    atomic_save_npy(out_dir / "reference_sigma_X.npy", sigma_points.astype(cfg.save_dtype))
    atomic_save_npy(out_dir / "true_displacements.npy", true_disp.astype(cfg.save_dtype))
    atomic_save_npy(out_dir / "pred_displacements.npy", pred_disp.astype(cfg.save_dtype))
    
    if plan["save_maps"]:
        atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
        atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
    atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
    try:
        rc.save_configuration(out_dir)
    except Exception:
        pass
    
    time_save = time.perf_counter() - t0_save
    time_total = time_load + time_prep + time_train + time_rollout + time_save
    
    # ── RESULT ──
    result = {
        "system": cfg.name,
        "method": "POSITIONS",
        "domain": "lot_positions",
        "N": N,
        "R": R,
        "kind": kind,
        "frac_pct": frac,
        "reservoir_scale": reservoir_scale,
        "reservoir_size": reservoir_size,
        "T": T,
        "d": d,
        "cycle_length": cfg.cycle_length,
        "warm_steps": warm_steps,
        "forecast_steps": forecast_steps,
        "train_rmse": train_rmse,
        "forecast_map_rmse": map_rmse,
        "using_actual_reference": using_actual_reference,
        "time_load": time_load,
        "time_prep": time_prep,
        "time_train": time_train,
        "time_rollout": time_rollout,
        "time_save": time_save,
        "time_total": time_total,
        "saved_maps": plan["save_maps"],
        "output_dir": str(out_dir),
        "rc_backend": backend_used,
        "rc_device": dev_str,
    }
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"[POSITIONS] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
          f"train={train_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
    return result


def run_grid(
    cfg: SystemConfig,
    skip_existing: bool = False,
    csv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run full parameter grid: N × kinds × fractions × reservoir_scales."""
    env_banner()
    np.random.seed(cfg.random_seed)
    
    results = []
    
    for N in cfg.N_list:
        for kind in cfg.kinds:
            for frac in cfg.fractions:
                lot_dir = cfg.lot_dir(N, kind, frac)
                
                for scale in cfg.reservoir_scales:
                    out_dir = cfg.output_dir(N, kind, frac, scale)
                    
                    result = run_single(
                        lot_dir, out_dir, N, kind, frac, scale, cfg, skip_existing
                    )
                    
                    if result is not None:
                        results.append(result)
                        if csv_path:
                            append_csv_row(csv_path, list(result.keys()), list(result.values()))
    
    print(f"\n[DONE] POSITIONS grid: {len(results)} runs")
    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="POSITIONS forecasting (direct map prediction)")
    parser.add_argument("system", choices=["geodesic_transport", "swirling_cluster"])
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=None)
    parser.add_argument("--kinds", type=str, nargs="+", default=None)
    parser.add_argument("--fracs", type=int, nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
    parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
    parser.add_argument("--run-tag", type=str, default="positions")
    parser.add_argument("--cycle-length", type=int, default=None,
                        help="Override cycle length (default: auto-detect from system)")
    parser.add_argument(
        "--rc-backend",
        choices=["auto", "numpy", "torch"],
        default="auto",
        help="Reservoir implementation: PyTorch (GPU-capable) or NumPy",
    )
    parser.add_argument(
        "--rc-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device when using torch backend",
    )
    args = parser.parse_args()
    
    cfg = SystemConfig(
        name=args.system,
        N_list=args.N or [100, 250, 500, 750, 1000, 1500],
        reservoir_scales=args.scales or [0.5, 1.0, 1.5, 2.0, 5.0],
        kinds=args.kinds or DEFAULT_KINDS,
        fractions=args.fracs or DEFAULT_FRACTIONS,
        results_root=args.results_root,
        forecast_root=args.output_root,
        lot_root=args.lot_root,
        run_tag=args.run_tag,
        cycle_length=args.cycle_length,
        rc_backend=args.rc_backend,
        rc_device=args.rc_device,
    )
    
    run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


if __name__ == "__main__":
    main()





# #!/usr/bin/env python3
# """
# POSITIONS Forecasting: Predict LOT transport maps directly.

# Predicts future LOT transport maps (positions) directly without the 
# velocity formulation. This is the baseline comparison for VELOCITY.

# PARADIGM:
#     1. Compute LOT embeddings: T_t = T_σ^{μ_t}
#     2. Train reservoir: T_t → T_{t+1} (direct map prediction)
#     3. Autonomous rollout: feed predicted maps back as input
#     4. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

# KEY CHARACTERISTICS:
#     - Predicts transport maps (positions) directly
#     - Non-stationary target: maps drift further from reference over time
#     - Errors compound through autonomous rollout (exponential growth)
#     - Baseline for comparison with VELOCITY approach

# Compare to VELOCITY (velocity_forecast.py):
#     - VELOCITY predicts v_t = T_{t+1} - T_t, then integrates
#     - More stationary target, errors accumulate gradually (linear growth)

# Supports both geodesic_transport and swirling_cluster systems.
# Both systems use 7 reference types.

# Usage:
#     from positions_forecast import run_grid, SystemConfig
    
#     cfg = SystemConfig(
#         name="geodesic_transport",
#         N_list=[500, 1000],
#         reservoir_scales=[1.0, 2.0],
#     )
#     results = run_grid(cfg)
    
# CLI:
#     python positions_forecast.py geodesic_transport
#     python positions_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
# """

# from __future__ import annotations
# import os
# import time
# import json
# import csv
# import shutil
# import platform
# import argparse
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import List, Optional, Dict, Any, Tuple
# import numpy as np

# # Import warm-up utilities
# from .warmup_utils import compute_warm_steps_for_positions, validate_warm_steps


# # ═══════════════════════════════════════════════════════════════
# # CONFIGURATION
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

# DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


# @dataclass
# class SystemConfig:
#     """Configuration for POSITIONS forecasting experiments."""
    
#     name: str  # "geodesic_transport" or "swirling_cluster"
    
#     # Grid parameters
#     N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
#     reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
#     # LOT reference configuration
#     kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
#     fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
#     # Warm-up configuration (now uses system defaults from warmup_utils)
#     cycle_length: Optional[int] = None  # Override system default if set
    
#     # RC hyperparameters (can use higher spectral radius for position prediction)
#     spectral_radius: float = 0.9        # Higher for direct map prediction
#     input_scaling: float = 0.1
#     leak_rate: float = 1.0              # No leaky integration
#     ridge_param: float = 1e-6
#     random_seed: int = 42
    
#     # Paths
#     results_root: Path = field(default_factory=lambda: Path("results"))
#     forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
#     lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
#     # Save options
#     save_dtype: type = np.float32
#     save_maps: bool = True
    
#     run_tag: str = "positions"
    
#     def __post_init__(self):
#         self.results_root = Path(self.results_root)
#         self.forecast_root = Path(self.forecast_root)
#         self.lot_root = Path(self.lot_root)
    
#     def lot_dir(self, N: int, kind: str, frac: int) -> Path:
#         return self.lot_root / f"{self.name}_N_{N}" / kind / f"frac{frac}"
    
#     def output_dir(self, N: int, kind: str, frac: int, reservoir_scale: float) -> Path:
#         pct = int(reservoir_scale * 100)
#         tag = f"_{self.run_tag}" if self.run_tag else ""
#         return (
#             self.forecast_root / f"{self.name}_N_{N}" / kind / f"frac{frac}" /
#             f"positions_resFromOrigN_{pct}pct{tag}"
#         )


# # ═══════════════════════════════════════════════════════════════
# # UTILITIES
# # ═══════════════════════════════════════════════════════════════

# def env_banner() -> None:
#     print(f"[env] Python {platform.python_version()} | NumPy {np.__version__} | "
#           f"OMP={os.environ.get('OMP_NUM_THREADS', '?')} MKL={os.environ.get('MKL_NUM_THREADS', '?')}")


# def rmse(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.sqrt(np.mean((a - b) ** 2)))


# def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
#     """Atomic save to avoid corrupt partial files."""
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.parent / f".tmp_{path.name}"
#     np.save(tmp, arr)
#     tmp_actual = tmp if tmp.exists() else tmp.parent / (tmp.name + ".npy")
#     os.replace(tmp_actual, path)


# def minimal_run_exists(out_dir: Path) -> bool:
#     """Check if essential outputs exist."""
#     return all((out_dir / f).exists() for f in [
#         "predicted_measures_X.npy", "true_measures_X.npy", "measures_w.npy"
#     ])


# def clean_incomplete_run(out_dir: Path) -> None:
#     if out_dir.exists() and not minimal_run_exists(out_dir):
#         print(f"[CLEAN] Removing incomplete: {out_dir}")
#         shutil.rmtree(out_dir, ignore_errors=True)


# def append_csv_row(path: Path, header: List[str], row: List[Any]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     write_header = not path.exists()
#     with open(path, "a", newline="") as f:
#         w = csv.writer(f)
#         if write_header:
#             w.writerow(header)
#         w.writerow(row)


# def est_bytes(arr: np.ndarray, dtype=None) -> int:
#     dt = np.dtype(dtype) if dtype is not None else arr.dtype
#     return int(arr.size * dt.itemsize)


# def preflight_space(dest_dir: Path, bytes_needed: int, margin: float = 1.15) -> Tuple[bool, Dict]:
#     dest_dir = Path(dest_dir)
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     total, used, free = shutil.disk_usage(dest_dir)
#     need = int(bytes_needed * margin)
#     return free >= need, {"free": free, "need": need, "where": str(dest_dir)}


# def choose_save_plan(
#     out_dir: Path,
#     pred_maps: np.ndarray,
#     true_maps: np.ndarray,
#     pred_meas: np.ndarray,
#     true_meas: np.ndarray,
#     weights: np.ndarray,
#     cfg: SystemConfig
# ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
#     """Choose what to save based on available disk space."""
#     dtype = cfg.save_dtype
    
#     def estimate(save_maps: bool) -> int:
#         total = 0
#         if save_maps:
#             total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
#         total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
#         total += est_bytes(weights, np.float32)
#         total += 2 * est_bytes(pred_meas, dtype)
#         return total
    
#     ok, info = preflight_space(out_dir, estimate(True))
#     if ok:
#         return {"save_maps": True, "save_meas": True}, info
    
#     ok, info = preflight_space(out_dir, estimate(False))
#     if ok:
#         return {"save_maps": False, "save_meas": True}, info
    
#     raise OSError(f"Insufficient disk space at {out_dir}")


# # ═══════════════════════════════════════════════════════════════
# # CORE PIPELINE
# # ═══════════════════════════════════════════════════════════════

# def run_single(
#     lot_dir: Path,
#     out_dir: Path,
#     N: int,
#     kind: str,
#     frac: int,
#     reservoir_scale: float,
#     cfg: SystemConfig,
#     skip_existing: bool = False,
# ) -> Optional[Dict[str, Any]]:
#     """
#     Run POSITIONS forecast for single configuration.
    
#     POSITIONS paradigm: predict maps directly T_t → T_{t+1}.
#     This has non-stationary targets and exponential error growth.
    
#     Returns dict with timing/metrics, or None if skipped/missing.
#     """
#     if not (lot_dir / "lot_maps.npy").exists():
#         print(f"[MISS] {lot_dir}")
#         return None
    
#     if skip_existing and minimal_run_exists(out_dir):
#         print(f"[SKIP] {out_dir}")
#         return None
      
#     clean_incomplete_run(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     # Import RC with correct API
#     from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
    
#     # ── LOAD ──
#     t0_load = time.perf_counter()
#     maps = np.load(lot_dir / "lot_maps.npy")  # (T, R, d)
#     time_load = time.perf_counter() - t0_load
    
#     T, R, d = maps.shape
#     assert d == 2, f"Expected d=2, got d={d}"
    
#     # ── PREP ──
#     t0_prep = time.perf_counter()
    
#     # POSITIONS paradigm: predict map[t+1] from map[t]
#     # This is the direct position prediction approach
#     map_flat = maps.reshape(T, R * d)
#     input_seq = map_flat[:-1]   # T_0, ..., T_{T-2}
#     target_seq = map_flat[1:]   # T_1, ..., T_{T-1}
#     max_steps = input_seq.shape[0]
    
#     # Use warmup_utils for correct warm-up calculation
#     warm_steps = compute_warm_steps_for_positions(
#         T=T,
#         system_name=cfg.name,
#         cycle_length=cfg.cycle_length,
#         lot_dir=lot_dir,
#     )
    
#     train_input = input_seq[:warm_steps]
#     train_target = target_seq[:warm_steps]
    
#     time_prep = time.perf_counter() - t0_prep
    
#     # ── TRAIN ──
#     t0_train = time.perf_counter()
#     reservoir_size = max(1, int(round(reservoir_scale * N)))
    
#     rc_config = ReservoirConfig(
#         input_size=R * d,
#         reservoir_size=reservoir_size,
#         output_size=R * d,
#         spectral_radius=cfg.spectral_radius,
#         input_scaling=cfg.input_scaling,
#         leak_rate=cfg.leak_rate,
#         ridge_param=cfg.ridge_param,
#         activation="tanh",
#         init_scheme=InitScheme.SPARSE,
#         random_seed=cfg.random_seed,
#     )
#     rc = ReservoirComputer(rc_config)
    
#     states = rc.run(train_input)
#     rc.train(states, train_target)
    
#     # Training RMSE
#     train_pred = rc.predict(states)
#     train_rmse = rmse(train_pred, train_target)
    
#     time_train = time.perf_counter() - t0_train
    
#     # ── ROLLOUT ──
#     t0_roll = time.perf_counter()
    
#     # Forecast horizon: predict maps T[warm_steps+1], ..., T[T-1]
#     # Note: After warm-up with T[0:warm_steps], first prediction is T[warm_steps]
#     # To align with VELOCITY output, we skip first prediction
#     forecast_steps = T - 1 - warm_steps
    
#     # Run autonomous for forecast_steps + 1 (includes T[warm_steps])
#     _, all_pred_flat = rc.run_autonomous(train_input, forecast_steps + 1)
    
#     # Skip first prediction to align with VELOCITY
#     # Output: T[warm_steps+1], ..., T[T-1]
#     pred_maps_flat = all_pred_flat[1:]
#     pred_maps = pred_maps_flat.reshape(forecast_steps, R, d)
    
#     time_rollout = time.perf_counter() - t0_roll
    
#     # ── GROUND TRUTH ──
#     # True maps for comparison: maps[warm_steps+1:T]
#     true_future_maps = maps[warm_steps + 1:T]
    
#     # ── METRICS ──
#     map_rmse = rmse(pred_maps, true_future_maps)
    
#     # ── PREPARE OUTPUT ──
#     pred_measures_X = pred_maps
#     true_measures_X = true_future_maps
#     weights = np.full(R, 1.0 / R, dtype=np.float32)
    
#     sigma_points = maps[0].copy()
#     true_disp = true_measures_X - sigma_points
#     pred_disp = pred_measures_X - sigma_points
    
#     plan, space_info = choose_save_plan(
#         out_dir, pred_maps, true_future_maps,
#         pred_measures_X, true_measures_X, weights, cfg
#     )
    
#     # ── SAVE ──
#     t0_save = time.perf_counter()
    
#     atomic_save_npy(out_dir / "predicted_measures_X.npy", pred_measures_X.astype(cfg.save_dtype))
#     atomic_save_npy(out_dir / "true_measures_X.npy", true_measures_X.astype(cfg.save_dtype))
#     atomic_save_npy(out_dir / "measures_w.npy", weights)
#     atomic_save_npy(out_dir / "reference_sigma_X.npy", sigma_points.astype(cfg.save_dtype))
#     atomic_save_npy(out_dir / "true_displacements.npy", true_disp.astype(cfg.save_dtype))
#     atomic_save_npy(out_dir / "pred_displacements.npy", pred_disp.astype(cfg.save_dtype))
    
#     if plan["save_maps"]:
#         atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
#         atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
#     atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
#     try:
#         rc.save_configuration(out_dir)
#     except Exception:
#         pass
    
#     time_save = time.perf_counter() - t0_save
#     time_total = time_load + time_prep + time_train + time_rollout + time_save
    
#     # ── RESULT ──
#     result = {
#         "system": cfg.name,
#         "method": "POSITIONS",
#         "domain": "lot_positions",
#         "N": N,
#         "R": R,
#         "kind": kind,
#         "frac_pct": frac,
#         "reservoir_scale": reservoir_scale,
#         "reservoir_size": reservoir_size,
#         "T": T,
#         "d": d,
#         "cycle_length": cfg.cycle_length,
#         "warm_steps": warm_steps,
#         "forecast_steps": forecast_steps,
#         "train_rmse": train_rmse,
#         "forecast_map_rmse": map_rmse,
#         "time_load": time_load,
#         "time_prep": time_prep,
#         "time_train": time_train,
#         "time_rollout": time_rollout,
#         "time_save": time_save,
#         "time_total": time_total,
#         "saved_maps": plan["save_maps"],
#         "output_dir": str(out_dir),
#     }
    
#     with open(out_dir / "metadata.json", "w") as f:
#         json.dump(result, f, indent=2)
    
#     print(f"[POSITIONS] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
#           f"train={train_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
#     return result


# def run_grid(
#     cfg: SystemConfig,
#     skip_existing: bool = False,
#     csv_path: Optional[Path] = None,
# ) -> List[Dict[str, Any]]:
#     """Run full parameter grid: N × kinds × fractions × reservoir_scales."""
#     env_banner()
#     np.random.seed(cfg.random_seed)
    
#     results = []
    
#     for N in cfg.N_list:
#         for kind in cfg.kinds:
#             for frac in cfg.fractions:
#                 lot_dir = cfg.lot_dir(N, kind, frac)
                
#                 for scale in cfg.reservoir_scales:
#                     out_dir = cfg.output_dir(N, kind, frac, scale)
                    
#                     result = run_single(
#                         lot_dir, out_dir, N, kind, frac, scale, cfg, skip_existing
#                     )
                    
#                     if result is not None:
#                         results.append(result)
#                         if csv_path:
#                             append_csv_row(csv_path, list(result.keys()), list(result.values()))
    
#     print(f"\n[DONE] POSITIONS grid: {len(results)} runs")
#     return results


# # ═══════════════════════════════════════════════════════════════
# # CLI
# # ═══════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description="POSITIONS forecasting (direct map prediction)")
#     parser.add_argument("system", choices=["geodesic_transport", "swirling_cluster"])
#     parser.add_argument("--N", type=int, nargs="+", default=None)
#     parser.add_argument("--scales", type=float, nargs="+", default=None)
#     parser.add_argument("--kinds", type=str, nargs="+", default=None)
#     parser.add_argument("--fracs", type=int, nargs="+", default=None)
#     parser.add_argument("--skip-existing", action="store_true")
#     parser.add_argument("--csv", type=Path, default=None)
#     parser.add_argument("--results-root", type=Path, default=Path("results"))
#     parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
#     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
#     parser.add_argument("--run-tag", type=str, default="positions")
#     parser.add_argument("--cycle-length", type=int, default=None,
#                         help="Override cycle length (default: auto-detect from system)")
#     args = parser.parse_args()
    
#     cfg = SystemConfig(
#         name=args.system,
#         N_list=args.N or [100, 250, 500, 750, 1000, 1500],
#         reservoir_scales=args.scales or [0.5, 1.0, 1.5, 2.0, 5.0],
#         kinds=args.kinds or DEFAULT_KINDS,
#         fractions=args.fracs or DEFAULT_FRACTIONS,
#         results_root=args.results_root,
#         forecast_root=args.output_root,
#         lot_root=args.lot_root,
#         run_tag=args.run_tag,
#         cycle_length=args.cycle_length,
#     )
    
#     run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


# if __name__ == "__main__":
#     main()



# # #!/usr/bin/env python3
# # """
# # POSITIONS Forecasting: Predict LOT transport maps directly.

# # Predicts future LOT transport maps (positions) directly without the 
# # velocity formulation. This is the baseline comparison for VELOCITY.

# # PARADIGM:
# #     1. Compute LOT embeddings: T_t = T_σ^{μ_t}
# #     2. Train reservoir: T_t → T_{t+1} (direct map prediction)
# #     3. Autonomous rollout: feed predicted maps back as input
# #     4. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

# # KEY CHARACTERISTICS:
# #     - Predicts transport maps (positions) directly
# #     - Non-stationary target: maps drift further from reference over time
# #     - Errors compound through autonomous rollout (exponential growth)
# #     - Baseline for comparison with VELOCITY approach

# # Compare to VELOCITY (velocity_forecast.py):
# #     - VELOCITY predicts v_t = T_{t+1} - T_t, then integrates
# #     - More stationary target, errors accumulate gradually (linear growth)

# # Supports both geodesic_transport and swirling_cluster systems.
# # Both systems use 7 reference types.

# # Usage:
# #     from positions_forecast import run_grid, SystemConfig
    
# #     cfg = SystemConfig(
# #         name="geodesic_transport",
# #         N_list=[500, 1000],
# #         reservoir_scales=[1.0, 2.0],
# #         cycle_length=100,
# #     )
# #     results = run_grid(cfg)
    
# # CLI:
# #     python positions_forecast.py geodesic_transport
# #     python positions_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
# # """

# # from __future__ import annotations
# # import os
# # import time
# # import json
# # import csv
# # import shutil
# # import platform
# # import argparse
# # from pathlib import Path
# # from dataclasses import dataclass, field
# # from typing import List, Optional, Dict, Any, Tuple
# # import numpy as np


# # # ═══════════════════════════════════════════════════════════════
# # # CONFIGURATION
# # # ═══════════════════════════════════════════════════════════════

# # DEFAULT_KINDS = [
# #     "clean_circle_small",
# #     "clean_circle_large",
# #     "uniform_square",
# #     "gaussian_iso",
# #     "snapshot_begin",
# #     "snapshot_middle",
# #     "snapshot_end",
# # ]

# # DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


# # @dataclass
# # class SystemConfig:
# #     """Configuration for POSITIONS forecasting experiments."""
    
# #     name: str  # "geodesic_transport" or "swirling_cluster"
    
# #     # Grid parameters
# #     N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
# #     reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
# #     # LOT reference configuration
# #     kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
# #     fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
# #     # Warm-up configuration
# #     cycle_length: Optional[int] = None  # L: length of one cycle (e.g., 100 for one forward sweep)
# #     align_to_boundary: bool = True      # align autonomous start to cycle boundary
# #     warm_steps_default: int = 50        # fallback if cycle_length not set
    
# #     # RC hyperparameters (can use higher spectral radius for position prediction)
# #     spectral_radius: float = 0.9        # Higher for direct map prediction
# #     input_scaling: float = 0.1
# #     leak_rate: float = 1.0              # No leaky integration
# #     ridge_param: float = 1e-6
# #     random_seed: int = 42
    
# #     # Paths
# #     results_root: Path = field(default_factory=lambda: Path("results"))
# #     forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
# #     lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
# #     # Save options
# #     save_dtype: type = np.float32
# #     save_maps: bool = True
    
# #     run_tag: str = "positions"
    
# #     def __post_init__(self):
# #         self.results_root = Path(self.results_root)
# #         self.forecast_root = Path(self.forecast_root)
# #         self.lot_root = Path(self.lot_root)
    
# #     def lot_dir(self, N: int, kind: str, frac: int) -> Path:
# #         return self.lot_root / f"{self.name}_N_{N}" / kind / f"frac{frac}"
    
# #     def output_dir(self, N: int, kind: str, frac: int, reservoir_scale: float) -> Path:
# #         pct = int(reservoir_scale * 100)
# #         tag = f"_{self.run_tag}" if self.run_tag else ""
# #         return (
# #             self.forecast_root / f"{self.name}_N_{N}" / kind / f"frac{frac}" /
# #             f"positions_resFromOrigN_{pct}pct{tag}"
# #         )


# # # ═══════════════════════════════════════════════════════════════
# # # UTILITIES
# # # ═══════════════════════════════════════════════════════════════

# # def env_banner() -> None:
# #     print(f"[env] Python {platform.python_version()} | NumPy {np.__version__} | "
# #           f"OMP={os.environ.get('OMP_NUM_THREADS', '?')} MKL={os.environ.get('MKL_NUM_THREADS', '?')}")


# # def rmse(a: np.ndarray, b: np.ndarray) -> float:
# #     return float(np.sqrt(np.mean((a - b) ** 2)))


# # def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
# #     """Atomic save to avoid corrupt partial files."""
# #     path = Path(path)
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     tmp = path.parent / f".tmp_{path.name}"
# #     np.save(tmp, arr)
# #     tmp_actual = tmp if tmp.exists() else tmp.parent / (tmp.name + ".npy")
# #     os.replace(tmp_actual, path)


# # def minimal_run_exists(out_dir: Path) -> bool:
# #     """Check if essential outputs exist."""
# #     return all((out_dir / f).exists() for f in [
# #         "predicted_measures_X.npy", "true_measures_X.npy", "measures_w.npy"
# #     ])


# # def clean_incomplete_run(out_dir: Path) -> None:
# #     if out_dir.exists() and not minimal_run_exists(out_dir):
# #         print(f"[CLEAN] Removing incomplete: {out_dir}")
# #         shutil.rmtree(out_dir, ignore_errors=True)


# # def append_csv_row(path: Path, header: List[str], row: List[Any]) -> None:
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     write_header = not path.exists()
# #     with open(path, "a", newline="") as f:
# #         w = csv.writer(f)
# #         if write_header:
# #             w.writerow(header)
# #         w.writerow(row)


# # def est_bytes(arr: np.ndarray, dtype=None) -> int:
# #     dt = np.dtype(dtype) if dtype is not None else arr.dtype
# #     return int(arr.size * dt.itemsize)


# # def preflight_space(dest_dir: Path, bytes_needed: int, margin: float = 1.15) -> Tuple[bool, Dict]:
# #     dest_dir = Path(dest_dir)
# #     dest_dir.mkdir(parents=True, exist_ok=True)
# #     total, used, free = shutil.disk_usage(dest_dir)
# #     need = int(bytes_needed * margin)
# #     return free >= need, {"free": free, "need": need, "where": str(dest_dir)}


# # def choose_save_plan(
# #     out_dir: Path,
# #     pred_maps: np.ndarray,
# #     true_maps: np.ndarray,
# #     pred_meas: np.ndarray,
# #     true_meas: np.ndarray,
# #     weights: np.ndarray,
# #     cfg: SystemConfig
# # ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
# #     """Choose what to save based on available disk space."""
# #     dtype = cfg.save_dtype
    
# #     def estimate(save_maps: bool) -> int:
# #         total = 0
# #         if save_maps:
# #             total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
# #         total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
# #         total += est_bytes(weights, np.float32)
# #         total += 2 * est_bytes(pred_meas, dtype)
# #         return total
    
# #     ok, info = preflight_space(out_dir, estimate(True))
# #     if ok:
# #         return {"save_maps": True, "save_meas": True}, info
    
# #     ok, info = preflight_space(out_dir, estimate(False))
# #     if ok:
# #         return {"save_maps": False, "save_meas": True}, info
    
# #     raise OSError(f"Insufficient disk space at {out_dir}")


# # def compute_warm_steps(max_steps: int, cfg: SystemConfig) -> int:
# #     """
# #     Compute warm_steps based on cycle configuration.
    
# #     For POSITIONS (direct map prediction), we use cycle_length - 1 since:
# #     - map[t] -> map[t+1] is one transition
# #     - One cycle of L maps has L-1 transitions
    
# #     If cycle_length is set and align_to_boundary is True:
# #         warm_steps = cycle_length - 1 (train on one full cycle)
# #     Otherwise:
# #         warm_steps = warm_steps_default (fallback, typically 50)
# #     """
# #     if cfg.cycle_length is not None and cfg.align_to_boundary:
# #         return max(1, min(cfg.cycle_length - 1, max_steps - 1))
# #     return max(1, min(cfg.warm_steps_default, max_steps - 1))


# # # ═══════════════════════════════════════════════════════════════
# # # CORE PIPELINE
# # # ═══════════════════════════════════════════════════════════════

# # def run_single(
# #     lot_dir: Path,
# #     out_dir: Path,
# #     N: int,
# #     kind: str,
# #     frac: int,
# #     reservoir_scale: float,
# #     cfg: SystemConfig,
# #     skip_existing: bool = False,
# # ) -> Optional[Dict[str, Any]]:
# #     """
# #     Run POSITIONS forecast for single configuration.
    
# #     POSITIONS paradigm: predict maps directly T_t → T_{t+1}.
# #     This has non-stationary targets and exponential error growth.
    
# #     Returns dict with timing/metrics, or None if skipped/missing.
# #     """
# #     if not (lot_dir / "lot_maps.npy").exists():
# #         print(f"[MISS] {lot_dir}")
# #         return None
    
# #     if skip_existing and minimal_run_exists(out_dir):
# #         print(f"[SKIP] {out_dir}")
# #         return None
      
# #     clean_incomplete_run(out_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)
    
# #     # Import RC with correct API
# #     from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
    
# #     # ── LOAD ──
# #     t0_load = time.perf_counter()
# #     maps = np.load(lot_dir / "lot_maps.npy")  # (T, R, d)
# #     time_load = time.perf_counter() - t0_load
    
# #     T, R, d = maps.shape
# #     assert d == 2, f"Expected d=2, got d={d}"
    
# #     # ── PREP ──
# #     t0_prep = time.perf_counter()
    
# #     # POSITIONS paradigm: predict map[t+1] from map[t]
# #     # This is the direct position prediction approach
# #     map_flat = maps.reshape(T, R * d)
# #     input_seq = map_flat[:-1]   # T_0, ..., T_{T-2}
# #     target_seq = map_flat[1:]   # T_1, ..., T_{T-1}
# #     max_steps = input_seq.shape[0]
    
# #     warm_steps = compute_warm_steps(max_steps, cfg)
    
# #     train_input = input_seq[:warm_steps]
# #     train_target = target_seq[:warm_steps]
    
# #     time_prep = time.perf_counter() - t0_prep
    
# #     # ── TRAIN ──
# #     t0_train = time.perf_counter()
# #     reservoir_size = max(1, int(round(reservoir_scale * N)))
    
# #     rc_config = ReservoirConfig(
# #         input_size=R * d,
# #         reservoir_size=reservoir_size,
# #         output_size=R * d,
# #         spectral_radius=cfg.spectral_radius,
# #         input_scaling=cfg.input_scaling,
# #         leak_rate=cfg.leak_rate,
# #         ridge_param=cfg.ridge_param,
# #         activation="tanh",
# #         init_scheme=InitScheme.SPARSE,
# #         random_seed=cfg.random_seed,
# #     )
# #     rc = ReservoirComputer(rc_config)
    
# #     states = rc.run(train_input)
# #     rc.train(states, train_target)
    
# #     # Training RMSE
# #     train_pred = rc.predict(states)
# #     train_rmse = rmse(train_pred, train_target)
    
# #     time_train = time.perf_counter() - t0_train
    
# #     # ── ROLLOUT ──
# #     t0_roll = time.perf_counter()
    
# #     # Forecast horizon: predict maps T[warm_steps+1], ..., T[T-1]
# #     # Note: After warm-up with T[0:warm_steps], first prediction is T[warm_steps]
# #     # To align with VELOCITY output, we skip first prediction
# #     forecast_steps = T - 1 - warm_steps
    
# #     # Run autonomous for forecast_steps + 1 (includes T[warm_steps])
# #     _, all_pred_flat = rc.run_autonomous(train_input, forecast_steps + 1)
    
# #     # Skip first prediction to align with VELOCITY
# #     # Output: T[warm_steps+1], ..., T[T-1]
# #     pred_maps_flat = all_pred_flat[1:]
# #     pred_maps = pred_maps_flat.reshape(forecast_steps, R, d)
    
# #     time_rollout = time.perf_counter() - t0_roll
    
# #     # ── GROUND TRUTH ──
# #     # True maps for comparison: maps[warm_steps+1:T]
# #     true_future_maps = maps[warm_steps + 1:T]
    
# #     # ── METRICS ──
# #     map_rmse = rmse(pred_maps, true_future_maps)
    
# #     # ── PREPARE OUTPUT ──
# #     pred_measures_X = pred_maps
# #     true_measures_X = true_future_maps
# #     weights = np.full(R, 1.0 / R, dtype=np.float32)
    
# #     sigma_points = maps[0].copy()
# #     true_disp = true_measures_X - sigma_points
# #     pred_disp = pred_measures_X - sigma_points
    
# #     plan, space_info = choose_save_plan(
# #         out_dir, pred_maps, true_future_maps,
# #         pred_measures_X, true_measures_X, weights, cfg
# #     )
    
# #     # ── SAVE ──
# #     t0_save = time.perf_counter()
    
# #     atomic_save_npy(out_dir / "predicted_measures_X.npy", pred_measures_X.astype(cfg.save_dtype))
# #     atomic_save_npy(out_dir / "true_measures_X.npy", true_measures_X.astype(cfg.save_dtype))
# #     atomic_save_npy(out_dir / "measures_w.npy", weights)
# #     atomic_save_npy(out_dir / "reference_sigma_X.npy", sigma_points.astype(cfg.save_dtype))
# #     atomic_save_npy(out_dir / "true_displacements.npy", true_disp.astype(cfg.save_dtype))
# #     atomic_save_npy(out_dir / "pred_displacements.npy", pred_disp.astype(cfg.save_dtype))
    
# #     if plan["save_maps"]:
# #         atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
# #         atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
# #     atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
# #     try:
# #         rc.save_configuration(out_dir)
# #     except Exception:
# #         pass
    
# #     time_save = time.perf_counter() - t0_save
# #     time_total = time_load + time_prep + time_train + time_rollout + time_save
    
# #     # ── RESULT ──
# #     result = {
# #         "system": cfg.name,
# #         "method": "POSITIONS",
# #         "domain": "lot_positions",
# #         "N": N,
# #         "R": R,
# #         "kind": kind,
# #         "frac_pct": frac,
# #         "reservoir_scale": reservoir_scale,
# #         "reservoir_size": reservoir_size,
# #         "T": T,
# #         "d": d,
# #         "cycle_length": cfg.cycle_length,
# #         "warm_steps": warm_steps,
# #         "forecast_steps": forecast_steps,
# #         "train_rmse": train_rmse,
# #         "forecast_map_rmse": map_rmse,
# #         "time_load": time_load,
# #         "time_prep": time_prep,
# #         "time_train": time_train,
# #         "time_rollout": time_rollout,
# #         "time_save": time_save,
# #         "time_total": time_total,
# #         "saved_maps": plan["save_maps"],
# #         "output_dir": str(out_dir),
# #     }
    
# #     with open(out_dir / "metadata.json", "w") as f:
# #         json.dump(result, f, indent=2)
    
# #     print(f"[POSITIONS] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
# #           f"train={train_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
# #     return result


# # def run_grid(
# #     cfg: SystemConfig,
# #     skip_existing: bool = False,
# #     csv_path: Optional[Path] = None,
# # ) -> List[Dict[str, Any]]:
# #     """Run full parameter grid: N × kinds × fractions × reservoir_scales."""
# #     env_banner()
# #     np.random.seed(cfg.random_seed)
    
# #     results = []
    
# #     for N in cfg.N_list:
# #         for kind in cfg.kinds:
# #             for frac in cfg.fractions:
# #                 lot_dir = cfg.lot_dir(N, kind, frac)
                
# #                 for scale in cfg.reservoir_scales:
# #                     out_dir = cfg.output_dir(N, kind, frac, scale)
                    
# #                     result = run_single(
# #                         lot_dir, out_dir, N, kind, frac, scale, cfg, skip_existing
# #                     )
                    
# #                     if result is not None:
# #                         results.append(result)
# #                         if csv_path:
# #                             append_csv_row(csv_path, list(result.keys()), list(result.values()))
    
# #     print(f"\n[DONE] POSITIONS grid: {len(results)} runs")
# #     return results


# # # ═══════════════════════════════════════════════════════════════
# # # CLI
# # # ═══════════════════════════════════════════════════════════════

# # def main():
# #     parser = argparse.ArgumentParser(description="POSITIONS forecasting (direct map prediction)")
# #     parser.add_argument("system", choices=["geodesic_transport", "swirling_cluster"])
# #     parser.add_argument("--N", type=int, nargs="+", default=None)
# #     parser.add_argument("--scales", type=float, nargs="+", default=None)
# #     parser.add_argument("--kinds", type=str, nargs="+", default=None)
# #     parser.add_argument("--fracs", type=int, nargs="+", default=None)
# #     parser.add_argument("--skip-existing", action="store_true")
# #     parser.add_argument("--csv", type=Path, default=None)
# #     parser.add_argument("--results-root", type=Path, default=Path("results"))
# #     parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
# #     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
# #     parser.add_argument("--run-tag", type=str, default="positions")
# #     parser.add_argument("--cycle-length", type=int, default=None, help="Length of one cycle (e.g., 100)")
# #     parser.add_argument("--warm-steps", type=int, default=50, help="Fallback warm steps if cycle_length not set")
# #     parser.add_argument("--no-align", action="store_true", help="Disable boundary alignment")
# #     args = parser.parse_args()
    
# #     cfg = SystemConfig(
# #         name=args.system,
# #         N_list=args.N or [100, 250, 500, 750, 1000, 1500],
# #         reservoir_scales=args.scales or [0.5, 1.0, 1.5, 2.0, 5.0],
# #         kinds=args.kinds or DEFAULT_KINDS,
# #         fractions=args.fracs or DEFAULT_FRACTIONS,
# #         results_root=args.results_root,
# #         forecast_root=args.output_root,
# #         lot_root=args.lot_root,
# #         run_tag=args.run_tag,
# #         cycle_length=args.cycle_length,
# #         align_to_boundary=not args.no_align,
# #         warm_steps_default=args.warm_steps,
# #     )
    
# #     run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


# # if __name__ == "__main__":
# #     main()