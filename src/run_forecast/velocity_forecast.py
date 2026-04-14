# #!/usr/bin/env python3
# """
# VELOCITY v2 Forecasting: Predict velocities FROM POSITIONS, then integrate.

# This is the corrected formulation that works for ALL deterministic dynamical systems.

# PARADIGM (v2 - CORRECTED):
#     1. Compute LOT embeddings: T_t = T_σ^{μ_t}
#     2. Compute velocities: v_t = T_{t+1} - T_t
#     3. Train reservoir: T_t → v_t (position to velocity prediction)  ← KEY CHANGE
#     4. Autonomous rollout: 
#        a. Predict v̂_t from T̂_t
#        b. Integrate: T̂_{t+1} = T̂_t + v̂_t
#        c. Feed T̂_{t+1} back as next input
#     5. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

# WHY THIS IS CORRECT:
#     For any deterministic system, velocity is a function of position: v = f(x)
#     The original v_t → v_{t+1} paradigm assumed velocity is autoregressive,
#     which only works for simple smooth dynamics (swirling cluster, geodesic).
    
#     T_t → v_t learns the actual physics and works for:
#     - Swirling cluster (position/angle determines tangential velocity)
#     - Geodesic transport (position determines geodesic direction)  
#     - Double gyre (position in flow field determines local velocity)
#     - Any other deterministic dynamical system

# Compare to POSITIONS (positions_forecast.py):
#     - POSITIONS predicts T_{t+1} directly from T_t
#     - VELOCITY v2 predicts v_t from T_t, then integrates
#     - Both learn position-based mappings, but VELOCITY v2 targets velocity
#       which may be more stationary

# Usage:
#     from velocity_forecast_v2 import run_grid, SystemConfig
    
#     cfg = SystemConfig(
#         name="double_gyre",
#         N_list=[500],
#         reservoir_scales=[1.0],
#     )
#     results = run_grid(cfg)
    
# CLI:
#     python velocity_forecast_v2.py double_gyre
#     python velocity_forecast_v2.py swirling_cluster --N 500 1000
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

# # ═══════════════════════════════════════════════════════════════
# # WARM-UP UTILITIES (self-contained)
# # ═══════════════════════════════════════════════════════════════

# def compute_warm_steps(T: int, system_name: str, cycle_length: int = None) -> int:
#     """
#     Compute warm-up steps for training.
    
#     For VELOCITY v2 (T_t → v_t), we need positions T_0...T_{warm-1} 
#     to predict velocities v_0...v_{warm-1}.
    
#     Args:
#         T: Total trajectory length
#         system_name: Name of dynamical system
#         cycle_length: Optional override for cycle length
        
#     Returns:
#         Number of warm-up steps for training
#     """
#     # System-specific defaults
#     SYSTEM_DEFAULTS = {
#         "swirling_cluster": 100,
#         "geodesic_transport": 100,
#         "double_gyre": 200,  # One full period
#     }
    
#     if cycle_length is not None:
#         target = cycle_length
#     elif system_name in SYSTEM_DEFAULTS:
#         target = SYSTEM_DEFAULTS[system_name]
#     else:
#         # Default: use ~40% of trajectory
#         target = int(0.4 * T)
    
#     # Ensure we leave room for forecasting
#     # Need at least 1 step for forecast, and warm_steps < T-1 for velocities
#     max_warm = T - 2
#     return max(1, min(target, max_warm))


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
#     """Configuration for VELOCITY v2 forecasting experiments."""
    
#     name: str  # "geodesic_transport", "swirling_cluster", or "double_gyre"
    
#     # Grid parameters
#     N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
#     reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
#     # LOT reference configuration
#     kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
#     fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
#     # Warm-up configuration
#     cycle_length: Optional[int] = None  # Override system default if set
    
#     # RC hyperparameters (tuned for position→velocity prediction)
#     spectral_radius: float = 0.9        # Can be higher since we're not doing v→v
#     input_scaling: float = 0.5          # Higher — positions have larger magnitude than velocities
#     leak_rate: float = 0.3              # Lower leak rate to retain position information
#     ridge_param: float = 1e-2           # Regularization to prevent overfitting
#     random_seed: int = 42
    
#     # Paths
#     results_root: Path = field(default_factory=lambda: Path("results"))
#     forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
#     lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
#     # Save options
#     save_dtype: type = np.float32
#     save_velocities: bool = True
#     save_maps: bool = True
    
#     run_tag: str = "velocity_v2"
    
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
#             f"velocity_resFromOrigN_{pct}pct{tag}"
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
#     pred_vel: np.ndarray,
#     true_vel: np.ndarray,
#     pred_maps: np.ndarray,
#     true_maps: np.ndarray,
#     pred_meas: np.ndarray,
#     true_meas: np.ndarray,
#     weights: np.ndarray,
#     cfg: SystemConfig
# ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
#     """Choose what to save based on available disk space."""
#     dtype = cfg.save_dtype
    
#     def estimate(save_vel: bool, save_maps: bool) -> int:
#         total = 0
#         if save_vel:
#             total += est_bytes(pred_vel, dtype) + est_bytes(true_vel, dtype)
#         if save_maps:
#             total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
#         total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
#         total += est_bytes(weights, np.float32)
#         total += 2 * est_bytes(pred_meas, dtype)
#         return total
    
#     ok, info = preflight_space(out_dir, estimate(True, True))
#     if ok:
#         return {"save_vel": True, "save_maps": True, "save_meas": True}, info
    
#     ok, info = preflight_space(out_dir, estimate(False, True))
#     if ok:
#         return {"save_vel": False, "save_maps": True, "save_meas": True}, info
    
#     ok, info = preflight_space(out_dir, estimate(False, False))
#     if ok:
#         return {"save_vel": False, "save_maps": False, "save_meas": True}, info
    
#     raise OSError(f"Insufficient disk space at {out_dir}")


# # ═══════════════════════════════════════════════════════════════
# # CORE PIPELINE (v2: T_t → v_t)
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
#     Run VELOCITY v2 forecast for single configuration.
    
#     VELOCITY v2 paradigm: predict velocities FROM positions T_t → v_t, then integrate.
    
#     This is the correct formulation for all deterministic dynamical systems.
    
#     Returns dict with timing/metrics, or None if skipped/missing.
#     """
#     if not (lot_dir / "lot_maps.npy").exists() or not (lot_dir / "velocities.npy").exists():
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
#     maps = np.load(lot_dir / "lot_maps.npy")    # (T, R, d)
#     vels = np.load(lot_dir / "velocities.npy")  # (T-1, R, d)
#     time_load = time.perf_counter() - t0_load
    
#     T, R, d = maps.shape
#     assert d == 2, f"Expected d=2, got d={d}"
#     assert vels.shape[0] == T - 1, f"Velocity shape mismatch: {vels.shape[0]} vs {T-1}"
    
#     # ── PREP ──
#     t0_prep = time.perf_counter()
    
#     # ═══════════════════════════════════════════════════════════════
#     # KEY CHANGE: T_t → v_t (position to velocity)
#     # ═══════════════════════════════════════════════════════════════
#     # This learns the actual physics: velocity is a function of position
#     # Works for ALL deterministic systems (swirling, geodesic, double gyre, etc.)
    
#     map_flat = maps.reshape(T, R * d)       # Positions: T_0, ..., T_{T-1}
#     vel_flat = vels.reshape(T - 1, R * d)   # Velocities: v_0, ..., v_{T-2}
    
#     # Input: positions T_0, ..., T_{T-2}
#     # Target: velocities v_0, ..., v_{T-2}
#     input_seq = map_flat[:-1]    # T_0, ..., T_{T-2}
#     target_seq = vel_flat        # v_0, ..., v_{T-2}
    
#     max_steps = input_seq.shape[0]
    
#     # Compute warm-up steps
#     warm_steps = compute_warm_steps(
#         T=T,
#         system_name=cfg.name,
#         cycle_length=cfg.cycle_length,
#     )
    
#     train_input = input_seq[:warm_steps]    # T_0, ..., T_{warm_steps-1}
#     train_target = target_seq[:warm_steps]  # v_0, ..., v_{warm_steps-1}
    
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
#     # ═══════════════════════════════════════════════════════════════
#     # KEY CHANGE: Custom autonomous rollout with integration
#     # ═══════════════════════════════════════════════════════════════
#     # 1. Predict v̂_t from T̂_t
#     # 2. Integrate: T̂_{t+1} = T̂_t + v̂_t
#     # 3. Feed T̂_{t+1} back as next input
    
#     t0_roll = time.perf_counter()
    
#     # Forecast horizon
#     forecast_steps = T - 1 - warm_steps
    
#     # Initialize with the last training position
#     # We start predicting from T_{warm_steps}
#     current_map = maps[warm_steps].reshape(R * d)  # T_{warm_steps}
    
#     # Storage for predictions
#     pred_vel = np.empty((forecast_steps, R, d), dtype=maps.dtype)
#     pred_maps = np.empty((forecast_steps + 1, R, d), dtype=maps.dtype)
#     pred_maps[0] = maps[warm_steps]  # Seed with true position
    
#     # Note: rc.state is already set from the training run (states[-1])
#     # The reservoir remembers its state after rc.run(train_input)
    
#     # Autonomous rollout with integration
#     for t in range(forecast_steps):
#         # 1. Update reservoir state with current position
#         #    _update() modifies rc.state internally and returns it
#         rc._update(current_map)
        
#         # 2. Predict velocity from current state
#         #    Need to get state as row vector with bias appended
#         state_row = rc.state.flatten().reshape(1, -1)
#         pred_vel_t = rc.predict(state_row)[0]
#         pred_vel[t] = pred_vel_t.reshape(R, d)
        
#         # 3. Integrate: T̂_{t+1} = T̂_t + v̂_t
#         next_map = current_map + pred_vel_t
#         pred_maps[t + 1] = next_map.reshape(R, d)
        
#         # 4. Update current position for next iteration
#         current_map = next_map
    
#     time_rollout = time.perf_counter() - t0_roll
    
#     # ── GROUND TRUTH ──
#     true_future_maps = maps[warm_steps:warm_steps + forecast_steps + 1]
#     true_future_vels = vels[warm_steps:warm_steps + forecast_steps]
    
#     # ── METRICS ──
#     forecast_vel_rmse = rmse(pred_vel, true_future_vels)
#     map_rmse = rmse(pred_maps[1:], true_future_maps[1:])
    
#     # ── PREPARE OUTPUT ──
#     pred_measures_X = pred_maps[1:]
#     true_measures_X = true_future_maps[1:]
#     weights = np.full(R, 1.0 / R, dtype=np.float32)
    
#     # Load actual reference σ (if available), otherwise fall back to maps[0]
#     ref_path = lot_dir / "reference.npy"
#     if ref_path.exists():
#         sigma_points = np.load(ref_path).astype(maps.dtype)
#         using_actual_reference = True
#     else:
#         sigma_points = maps[0].copy()
#         using_actual_reference = False
#         print(f"[WARN] {lot_dir}/reference.npy not found, using maps[0] as σ")
    
#     true_disp = true_measures_X - sigma_points
#     pred_disp = pred_measures_X - sigma_points
    
#     plan, space_info = choose_save_plan(
#         out_dir, pred_vel, true_future_vels, pred_maps, true_future_maps,
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
    
#     if plan["save_vel"]:
#         atomic_save_npy(out_dir / "predicted_velocities.npy", pred_vel.astype(cfg.save_dtype))
#         atomic_save_npy(out_dir / "true_future_vels.npy", true_future_vels.astype(cfg.save_dtype))
    
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
#         "method": "VELOCITY_v2",
#         "paradigm": "T_t -> v_t (position to velocity)",
#         "domain": "lot_velocity",
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
#         "forecast_vel_rmse": forecast_vel_rmse,
#         "forecast_map_rmse": map_rmse,
#         "using_actual_reference": using_actual_reference,
#         "pred_range_min": float(pred_measures_X.min()),
#         "pred_range_max": float(pred_measures_X.max()),
#         "time_load": time_load,
#         "time_prep": time_prep,
#         "time_train": time_train,
#         "time_rollout": time_rollout,
#         "time_save": time_save,
#         "time_total": time_total,
#         "saved_vel": plan["save_vel"],
#         "saved_maps": plan["save_maps"],
#         "output_dir": str(out_dir),
#     }
    
#     with open(out_dir / "metadata.json", "w") as f:
#         json.dump(result, f, indent=2)
    
#     print(f"[VELOCITY_v2] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
#           f"train={train_rmse:.6f} vel={forecast_vel_rmse:.6f} map={map_rmse:.6f} | "
#           f"range=[{pred_measures_X.min():.2f}, {pred_measures_X.max():.2f}] | {time_total:.2f}s")
    
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
    
#     print(f"\n[DONE] VELOCITY_v2 grid: {len(results)} runs")
#     return results


# # ═══════════════════════════════════════════════════════════════
# # CLI
# # ═══════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description="VELOCITY forecasting (predict velocities, integrate)")
#     parser.add_argument("system", type=str, help="System name (e.g., geodesic_transport, swirling_cluster, sst_ocean)")
#     parser.add_argument("--N", type=int, nargs="+", default=None)
#     parser.add_argument("--scales", type=float, nargs="+", default=None)
#     parser.add_argument("--kinds", type=str, nargs="+", default=None)
#     parser.add_argument("--fracs", type=int, nargs="+", default=None)
#     parser.add_argument("--skip-existing", action="store_true")
#     parser.add_argument("--csv", type=Path, default=None)
#     parser.add_argument("--results-root", type=Path, default=Path("results"))
#     parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
#     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
#     parser.add_argument("--run-tag", type=str, default="velocity_v2")
#     parser.add_argument("--cycle-length", type=int, default=None, 
#                         help="Override cycle length (default: auto-detect from system)")
#     # Hyperparameter overrides
#     parser.add_argument("--spectral-radius", type=float, default=None)
#     parser.add_argument("--input-scaling", type=float, default=None)
#     parser.add_argument("--leak-rate", type=float, default=None)
#     parser.add_argument("--ridge-param", type=float, default=None)
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
    
#     # Apply hyperparameter overrides
#     if args.spectral_radius is not None:
#         cfg.spectral_radius = args.spectral_radius
#     if args.input_scaling is not None:
#         cfg.input_scaling = args.input_scaling
#     if args.leak_rate is not None:
#         cfg.leak_rate = args.leak_rate
#     if args.ridge_param is not None:
#         cfg.ridge_param = args.ridge_param
    
#     run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


# if __name__ == "__main__":
#     main()













#!/usr/bin/env python3
"""
VELOCITY Forecasting: Predict LOT velocity fields, then integrate.

Predicts future LOT velocities (in L²(σ) space), then reconstructs 
transport maps by integrating from a seed map.

PARADIGM:
    1. Compute LOT embeddings: T_t = T_σ^{μ_t}
    2. Compute velocities: v_t = T_{t+1} - T_t
    3. Train reservoir: v_t → v_{t+1} (one-step velocity prediction)
    4. Autonomous rollout: feed predicted velocities back as input
    5. Integrate: T̂_{t+1} = T_t + v̂_t
    6. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

KEY CHARACTERISTICS:
    - Predicts velocity fields (differences), not positions
    - More stationary target: velocity magnitudes stay consistent over time
    - Errors accumulate gradually through integration (linear growth)
    - Superior to POSITIONS approach for long-horizon forecasting

Compare to POSITIONS (positions_forecast.py):
    - POSITIONS predicts T_{t+1} directly from T_t
    - Non-stationary target, errors compound exponentially

Supports both geodesic_transport and swirling_cluster systems.
Both systems use 7 reference types.

Usage:
    from velocity_forecast import run_grid, SystemConfig
    
    cfg = SystemConfig(
        name="geodesic_transport",
        N_list=[500, 1000],
        reservoir_scales=[1.0, 2.0],
    )
    results = run_grid(cfg)
    
CLI:
    python velocity_forecast.py geodesic_transport
    python velocity_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
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

# Import warm-up utilities (cycle detection, warm-up computation, validation)
try:
    from .warmup_utils import (
        compute_warm_steps_for_velocity,
        validate_warm_steps,
        analyze_and_configure,
        detect_cycle_from_maps,
        detect_cycle_from_velocities,
        validate_cyclicity,
        SYSTEM_DEFAULTS,
    )
except ImportError:
    from warmup_utils import (
        compute_warm_steps_for_velocity,
        validate_warm_steps,
        analyze_and_configure,
        detect_cycle_from_maps,
        detect_cycle_from_velocities,
        validate_cyclicity,
        SYSTEM_DEFAULTS,
    )

# Import velocity scaling utilities for unscaling predictions
try:
    from data_utils.simulation.generate_lot_embeddings import unscale_velocities
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data_utils.simulation.generate_lot_embeddings import unscale_velocities
    except ImportError:
        # Fallback: define unscaling inline if import fails
        def unscale_velocities(scaled_velocities, params):
            """Fallback unscaling function."""
            method = params.get("scaling_method", "none")
            if method == "none":
                return scaled_velocities
            elif method == "1/m":
                return scaled_velocities * params["m"]
            elif method == "minmax":
                eps = 1e-8
                if params.get("constant_field", False):
                    return scaled_velocities * params["mag_max"]
                mag_min = params["mag_min"]
                mag_max = params["mag_max"]
                min_floor = params["min_floor"]
                mag_range = mag_max - mag_min
                scaled_mags = np.linalg.norm(scaled_velocities, axis=-1, keepdims=True)
                directions = scaled_velocities / (scaled_mags + eps)
                normalized_mags = (scaled_mags - min_floor) / (1.0 - min_floor + eps)
                original_mags = mag_min + normalized_mags * mag_range
                return directions * original_mags
            else:
                return scaled_velocities

# Import RC utilities (hyperparameter suggestion)
try:
    from .rc_utils import suggest_hyperparameters, analyze_dynamics
except ImportError:
    try:
        from rc_utils import suggest_hyperparameters, analyze_dynamics
    except ImportError:
        # Fallback if rc_utils not available
        suggest_hyperparameters = None
        analyze_dynamics = None

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
    """Configuration for VELOCITY forecasting experiments."""

    name: str  # "geodesic_transport" or "swirling_cluster"

    # Forecasting paradigm (paper Section 5.3):
    #   "velocity_autoregressive" (DEFAULT): Train RC with v_t → v_{t+1}.
    #       Feeds predicted velocities back during rollout, then integrates
    #       to recover LOT maps. This is the method described in the paper.
    paradigm: str = "velocity_autoregressive"

    # Grid parameters
    N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
    reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])

    # LOT reference configuration
    kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
    fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())

    # Warm-up configuration (now uses system defaults from warmup_utils)
    cycle_length: Optional[int] = None  # Override system default if set
    auto_detect_cycle: bool = False  # Auto-detect cycle from velocities

    # RC hyperparameters — paper defaults (§5.2–5.3):
    #   spectral_radius=0.7, leak_rate=0.7, ridge=1e-6, input_scaling=0.1
    spectral_radius: float = 0.7
    input_scaling: float = 0.1
    leak_rate: float = 0.7
    ridge_param: float = 1e-6
    random_seed: int = 42

    # Hyperparameter sweep configuration
    hyperparameter_sweep: bool = False  # If True, sweep over lists below and pick best
    spectral_radius_list: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    leak_rate_list: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    ridge_param_list: List[float] = field(default_factory=lambda: [1e-6, 1e-4, 1e-2])

    # Adaptive configuration
    adaptive_hyperparameters: bool = False  # Auto-tune hyperparameters from velocity stats
    validate_cyclicity: bool = False  # Warn if system doesn't appear cyclic

    # Paths
    results_root: Path = field(default_factory=lambda: Path("results"))
    forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
    lot_root: Path = field(default_factory=lambda: Path("lot_maps"))

    # Save options
    save_dtype: type = np.float32
    save_velocities: bool = True
    save_maps: bool = True

    run_tag: str = "velocity"

    # Boundary handling for integration (prevents particles drifting outside domain)
    # "none": no boundary handling (particles can drift anywhere)
    # "periodic": wrap around [0, 1]^d (for closed systems like tidal, SST)
    # "clip": clamp to [0, 1]^d (for bounded domains)
    boundary_mode: str = "none"

    # Velocity damping: scale predicted velocity by damping * decay^step.
    # Pulls forecasts toward zero velocity (persistence) as confidence drops.
    # damping=1.0, decay=1.0 means no damping (standard autonomous rollout).
    damping: float = 1.0
    decay: float = 1.0

    # Reservoir state nudging: at every nudge_interval autonomous steps,
    # blend reservoir state toward observation-driven state.
    # nudge_interval=0 disables nudging (default).
    nudge_interval: int = 0
    nudge_strength: float = 0.5

    # Reservoir backend: "auto" uses PyTorch when installed (GPU if --device cuda/auto), else NumPy
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
            f"velocity_resFromOrigN_{pct}pct{tag}"
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


def integrate_velocities_to_maps(
    pred_vel: np.ndarray,
    seed_map: np.ndarray,
    forecast_steps: int,
    velocity_scaling_params: dict,
    boundary_mode: str,
) -> np.ndarray:
    """
    Roll out predicted LOT velocities to LOT maps (same convention as main VELOCITY path).
    pred_vel: (forecast_steps, R, d). Returns (forecast_steps + 1, R, d).
    """
    scaling_method = velocity_scaling_params.get("scaling_method", "none")
    if scaling_method != "none":
        pv = unscale_velocities(pred_vel, velocity_scaling_params)
    else:
        pv = pred_vel
    pred_maps = np.empty((forecast_steps + 1,) + seed_map.shape, dtype=seed_map.dtype)
    pred_maps[0] = seed_map
    for t in range(forecast_steps):
        next_pos = pred_maps[t] + pv[t]
        if boundary_mode == "periodic":
            next_pos = next_pos % 1.0
        elif boundary_mode == "clip":
            next_pos = np.clip(next_pos, 0.0, 1.0)
        pred_maps[t + 1] = next_pos
    return pred_maps


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
    pred_vel: np.ndarray,
    true_vel: np.ndarray,
    pred_maps: np.ndarray,
    true_maps: np.ndarray,
    pred_meas: np.ndarray,
    true_meas: np.ndarray,
    weights: np.ndarray,
    cfg: SystemConfig
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Choose what to save based on available disk space."""
    dtype = cfg.save_dtype
    
    def estimate(save_vel: bool, save_maps: bool) -> int:
        total = 0
        if save_vel:
            total += est_bytes(pred_vel, dtype) + est_bytes(true_vel, dtype)
        if save_maps:
            total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
        total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
        total += est_bytes(weights, np.float32)
        total += 2 * est_bytes(pred_meas, dtype)
        return total
    
    ok, info = preflight_space(out_dir, estimate(True, True))
    if ok:
        return {"save_vel": True, "save_maps": True, "save_meas": True}, info
    
    ok, info = preflight_space(out_dir, estimate(False, True))
    if ok:
        return {"save_vel": False, "save_maps": True, "save_meas": True}, info
    
    ok, info = preflight_space(out_dir, estimate(False, False))
    if ok:
        return {"save_vel": False, "save_maps": False, "save_meas": True}, info
    
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
    Run VELOCITY forecast for single configuration.
    
    VELOCITY paradigm: predict velocities v_t → v_{t+1}, then integrate.
    
    Returns dict with timing/metrics, or None if skipped/missing.
    """
    if not (lot_dir / "lot_maps.npy").exists() or not (lot_dir / "velocities.npy").exists():
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
    maps = np.load(lot_dir / "lot_maps.npy")    # (T, R, d)
    vels = np.load(lot_dir / "velocities.npy")  # (T-1, R, d) - may be scaled

    # Load velocity scaling parameters from metadata
    meta_path = lot_dir / "metadata.json"
    velocity_scaling_params = {"scaling_method": "none"}  # Default: no scaling
    if meta_path.exists():
        with open(meta_path) as f:
            lot_metadata = json.load(f)
        velocity_scaling_params = lot_metadata.get("velocity_scaling", {"scaling_method": "none"})

    time_load = time.perf_counter() - t0_load

    T, R, d = maps.shape
    # Support both 2D (d=2) and 3D (d=3) data
    assert d in (2, 3), f"Expected d=2 or d=3, got d={d}"
    assert vels.shape[0] == T - 1, f"Velocity shape mismatch: {vels.shape[0]} vs {T-1}"
    
    # ── PREP ──
    t0_prep = time.perf_counter()

    # VELOCITY paradigm: predict vel[t+1] from vel[t]
    # This is more stationary than predicting maps directly
    vel_flat = vels.reshape(T - 1, R * d)
    input_seq = vel_flat[:-1]   # v_0, ..., v_{T-3}
    target_seq = vel_flat[1:]   # v_1, ..., v_{T-2}
    max_steps = input_seq.shape[0]

    # ── CYCLE DETECTION & CONFIGURATION ──
    # Determine cycle length (auto-detect or use defaults)
    detected_cycle = None
    cycle_source = "default"

    if cfg.auto_detect_cycle or cfg.adaptive_hyperparameters or cfg.validate_cyclicity:
        # Run full analysis
        analysis = analyze_and_configure(
            velocities=vels,
            system_name=cfg.name,
            T=T,
            validate=cfg.validate_cyclicity,
            verbose=False,
            prefer_system_defaults=not cfg.auto_detect_cycle,
        )
        detected_cycle = analysis["cycle_length"]
        cycle_source = analysis["cycle_source"]

    # Determine effective cycle length
    if cfg.cycle_length is not None:
        effective_cycle = cfg.cycle_length
        cycle_source = "explicit"
    elif cfg.auto_detect_cycle and detected_cycle is not None:
        effective_cycle = detected_cycle
    elif cfg.name in SYSTEM_DEFAULTS:
        effective_cycle = SYSTEM_DEFAULTS[cfg.name]["cycle_length"]
        cycle_source = "system_default"
    else:
        effective_cycle = None

    # Use warmup_utils for correct warm-up calculation
    warm_steps = compute_warm_steps_for_velocity(
        T=T,
        system_name=cfg.name,
        cycle_length=effective_cycle,
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
    true_future_maps = maps[warm_steps:warm_steps + forecast_steps + 1]
    true_future_vels = vels[warm_steps:warm_steps + forecast_steps]
    seed_map = maps[warm_steps]

    # ── ROLLOUT METHOD SELECTION ──
    use_damping = cfg.damping != 1.0 or cfg.decay != 1.0
    use_nudging = cfg.nudge_interval > 0
    # True future velocities in flat form, used as observations for nudging
    nudge_obs = vel_flat[warm_steps:warm_steps + forecast_steps] if use_nudging else None
    rollout_tag = "standard"
    if use_nudging:
        rollout_tag = f"nudged(K={cfg.nudge_interval}, s={cfg.nudge_strength})"
    elif use_damping:
        rollout_tag = f"damped(d={cfg.damping}, decay={cfg.decay})"

    def _do_rollout(rc_obj, warmup, n_fsteps):
        """Select and run the appropriate autonomous rollout method."""
        if use_nudging:
            obs = nudge_obs[:n_fsteps] if nudge_obs is not None else None
            return rc_obj.run_autonomous_nudged(
                warmup, n_fsteps,
                observations=obs,
                nudge_interval=cfg.nudge_interval,
                nudge_strength=cfg.nudge_strength,
            )
        elif use_damping:
            return rc_obj.run_autonomous_damped(
                warmup, n_fsteps,
                damping=cfg.damping,
                decay=cfg.decay,
            )
        else:
            return rc_obj.run_autonomous(warmup, n_fsteps)

    # ── HYPERPARAMETER SWEEP ──
    # Sweep over hyperparameter combinations like the tornado analysis
    if cfg.hyperparameter_sweep:
        # Match POSITIONS sweep: select hyperparameters by integrated LOT map RMSE so
        # ``forecast_map_rmse`` is what we optimize (fair comparison vs positions_forecast).
        hp_source = "sweep"
        best_map_rmse = float("inf")
        best_hp = None
        best_rc = None
        best_pred_vel = None
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

                    _, pred_vel_flat = _do_rollout(rc, train_input, forecast_steps)
                    pred_vel_candidate = pred_vel_flat.reshape(forecast_steps, R, d)

                    pred_maps_c = integrate_velocities_to_maps(
                        pred_vel_candidate,
                        seed_map,
                        forecast_steps,
                        velocity_scaling_params,
                        cfg.boundary_mode,
                    )
                    m_rmse = rmse(pred_maps_c[1:], true_future_maps[1:])
                    vel_rmse = rmse(pred_vel_candidate, true_future_vels)
                    sweep_results.append((sr, leak, ridge, m_rmse, vel_rmse, tr_rmse))

                    if m_rmse < best_map_rmse:
                        best_map_rmse = m_rmse
                        best_hp = (sr, leak, ridge)
                        best_rc = rc
                        best_pred_vel = pred_vel_candidate
                        best_train_rmse = tr_rmse
                        best_backend_used = backend_used
                        best_dev_str = dev_str

        spectral_radius, leak_rate, ridge_param = best_hp
        input_scaling = cfg.input_scaling
        rc = best_rc
        pred_vel = best_pred_vel
        train_rmse = best_train_rmse
        backend_used, dev_str = best_backend_used, best_dev_str

        sweep_results.sort(key=lambda x: x[3])  # by map RMSE
        print(
            f"  [SWEEP] Tested {len(sweep_results)} configs (select on map RMSE), "
            f"best: sr={spectral_radius}, leak={leak_rate}, ridge={ridge_param}"
        )

    # ── ADAPTIVE HYPERPARAMETERS ──
    # Use rc_utils.suggest_hyperparameters if enabled, otherwise use config defaults
    elif cfg.adaptive_hyperparameters and suggest_hyperparameters is not None:
        suggested = suggest_hyperparameters(vels, method="velocity", analyze=True)
        spectral_radius = suggested["spectral_radius"]
        input_scaling = suggested["input_scaling"]
        leak_rate = suggested["leak_rate"]
        ridge_param = suggested["ridge_param"]
        hp_source = "rc_utils"

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

        train_pred = rc.predict(states)
        train_rmse = rmse(train_pred, train_target)

        _, pred_vel_flat = _do_rollout(rc, train_input, forecast_steps)
        pred_vel = pred_vel_flat.reshape(forecast_steps, R, d)
    else:
        spectral_radius = cfg.spectral_radius
        input_scaling = cfg.input_scaling
        leak_rate = cfg.leak_rate
        ridge_param = cfg.ridge_param
        hp_source = "config"

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

        train_pred = rc.predict(states)
        train_rmse = rmse(train_pred, train_target)

        _, pred_vel_flat = _do_rollout(rc, train_input, forecast_steps)
        pred_vel = pred_vel_flat.reshape(forecast_steps, R, d)

    time_train = time.perf_counter() - t0_train
    time_rollout = 0.0  # Rollout is included in time_train for VELOCITY

    # ── UNSCALE VELOCITIES ──
    # If velocities were scaled during LOT embedding generation,
    # we need to unscale predictions before integration
    scaling_method = velocity_scaling_params.get("scaling_method", "none")
    if scaling_method != "none":
        # Unscale predicted velocities (they're in scaled space)
        pred_vel_unscaled = unscale_velocities(pred_vel, velocity_scaling_params)
        # Unscale true velocities for fair metric comparison
        true_future_vels_unscaled = unscale_velocities(true_future_vels, velocity_scaling_params)
    else:
        pred_vel_unscaled = pred_vel
        true_future_vels_unscaled = true_future_vels

    # ── MAP RECONSTRUCTION (Integration) ──
    # KEY INSIGHT: We integrate predicted velocities to get maps
    # This gives linear error growth instead of exponential
    t0_maps = time.perf_counter()

    # Integrate: map[t+1] = map[t] + vel[t] (using UNSCALED velocities)
    pred_maps = np.empty((forecast_steps + 1, R, d), dtype=maps.dtype)
    pred_maps[0] = seed_map
    for t in range(forecast_steps):
        next_pos = pred_maps[t] + pred_vel_unscaled[t]
        # Apply boundary handling to keep particles in domain
        if cfg.boundary_mode == "periodic":
            next_pos = next_pos % 1.0  # Wrap to [0, 1)^d
        elif cfg.boundary_mode == "clip":
            next_pos = np.clip(next_pos, 0.0, 1.0)  # Clamp to [0, 1]^d
        # "none": no boundary handling (default, allows particles to drift)
        pred_maps[t + 1] = next_pos

    time_maps = time.perf_counter() - t0_maps

    # ── METRICS ──
    # Compare unscaled velocities for interpretable metrics
    forecast_vel_rmse = rmse(pred_vel_unscaled, true_future_vels_unscaled)
    map_rmse = rmse(pred_maps[1:], true_future_maps[1:])
    
    # ── PREPARE OUTPUT ──
    pred_measures_X = pred_maps[1:]
    true_measures_X = true_future_maps[1:]
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
        out_dir, pred_vel, true_future_vels, pred_maps, true_future_maps,
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
    
    if plan["save_vel"]:
        # Save unscaled velocities (physical units) for interpretability
        atomic_save_npy(out_dir / "predicted_velocities.npy", pred_vel_unscaled.astype(cfg.save_dtype))
        atomic_save_npy(out_dir / "true_future_vels.npy", true_future_vels_unscaled.astype(cfg.save_dtype))
        # Also save scaled velocities if scaling was applied (for debugging)
        if scaling_method != "none":
            atomic_save_npy(out_dir / "predicted_velocities_scaled.npy", pred_vel.astype(cfg.save_dtype))
            atomic_save_npy(out_dir / "true_future_vels_scaled.npy", true_future_vels.astype(cfg.save_dtype))
    
    if plan["save_maps"]:
        atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
        atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
    atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
    try:
        rc.save_configuration(out_dir)
    except Exception:
        pass
    
    time_save = time.perf_counter() - t0_save
    time_total = time_load + time_prep + time_train + time_rollout + time_maps + time_save
    
    # ── RESULT ──
    result = {
        "system": cfg.name,
        "method": "VELOCITY",
        "domain": "lot_velocity",
        "N": N,
        "R": R,
        "kind": kind,
        "frac_pct": frac,
        "reservoir_scale": reservoir_scale,
        "reservoir_size": reservoir_size,
        "T": T,
        "d": d,
        # Cycle configuration
        "cycle_length": effective_cycle,
        "cycle_source": cycle_source,
        "detected_cycle": detected_cycle,
        "warm_steps": warm_steps,
        "forecast_steps": forecast_steps,
        # Hyperparameters (actual values used)
        "spectral_radius": spectral_radius,
        "input_scaling": input_scaling,
        "leak_rate": leak_rate,
        "ridge_param": ridge_param,
        "hp_source": hp_source,
        "sweep_optimizes": "forecast_map_rmse" if cfg.hyperparameter_sweep else None,
        # Velocity scaling (from LOT embedding)
        "velocity_scaling_method": scaling_method,
        "velocity_scaling_params": velocity_scaling_params,
        # Metrics
        "train_rmse": train_rmse,
        "forecast_vel_rmse": forecast_vel_rmse,
        "forecast_map_rmse": map_rmse,
        "using_actual_reference": using_actual_reference,
        # Timing
        "time_load": time_load,
        "time_prep": time_prep,
        "time_train": time_train,
        "time_rollout": time_rollout,
        "time_maps_recon": time_maps,
        "time_save": time_save,
        "time_total": time_total,
        # Save info
        "saved_vel": plan["save_vel"],
        "saved_maps": plan["save_maps"],
        "output_dir": str(out_dir),
        "rc_backend": backend_used,
        "rc_device": dev_str,
        # Rollout method
        "rollout_method": rollout_tag,
        "damping": cfg.damping,
        "decay": cfg.decay,
        "nudge_interval": cfg.nudge_interval,
        "nudge_strength": cfg.nudge_strength,
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(result, f, indent=2)

    # Enhanced logging with cycle, HP, and scaling info
    cycle_info = f"cycle={effective_cycle}({cycle_source})" if effective_cycle else "cycle=auto"
    hp_info = f"hp={hp_source}" if cfg.adaptive_hyperparameters else ""
    scale_info = f"scale={scaling_method}" if scaling_method != "none" else ""
    print(f"[VELOCITY] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} {cycle_info} {hp_info} {scale_info} | "
          f"train={train_rmse:.6f} vel={forecast_vel_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
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
    
    print(f"\n[DONE] VELOCITY grid: {len(results)} runs")
    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VELOCITY forecasting (predict velocities, integrate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run with system defaults
  python velocity_forecast.py geodesic_transport --N 500

  # Auto-detect cycle and use adaptive hyperparameters
  python velocity_forecast.py swirling_cluster --adaptive --auto-detect-cycle

  # Validate cyclicity (warns if system doesn't appear cyclic)
  python velocity_forecast.py era5_wind --validate-cyclicity
        """
    )
    # System - now accepts any string for flexibility
    parser.add_argument("system", type=str,
                        help="System name (e.g., geodesic_transport, swirling_cluster, era5_wind)")

    # Grid parameters
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=None)
    parser.add_argument("--kinds", type=str, nargs="+", default=None)
    parser.add_argument("--fracs", type=int, nargs="+", default=None)

    # Paths
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
    parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
    parser.add_argument("--run-tag", type=str, default="velocity")

    # Cycle configuration
    parser.add_argument("--cycle-length", type=int, default=None,
                        help="Override cycle length (default: use system defaults)")
    parser.add_argument("--auto-detect-cycle", action="store_true",
                        help="Auto-detect cycle length from velocity autocorrelation")

    # Adaptive features
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive hyperparameters based on velocity statistics")
    parser.add_argument("--validate-cyclicity", action="store_true",
                        help="Warn if system doesn't appear cyclic (optional)")

    # Manual hyperparameter overrides (take precedence over adaptive)
    parser.add_argument("--spectral-radius", type=float, default=None)
    parser.add_argument("--input-scaling", type=float, default=None)
    parser.add_argument("--leak-rate", type=float, default=None)
    parser.add_argument("--ridge-param", type=float, default=None)

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
        auto_detect_cycle=args.auto_detect_cycle,
        adaptive_hyperparameters=args.adaptive,
        validate_cyclicity=args.validate_cyclicity,
        rc_backend=args.rc_backend,
        rc_device=args.rc_device,
    )

    # Apply manual hyperparameter overrides
    if args.spectral_radius is not None:
        cfg.spectral_radius = args.spectral_radius
    if args.input_scaling is not None:
        cfg.input_scaling = args.input_scaling
    if args.leak_rate is not None:
        cfg.leak_rate = args.leak_rate
    if args.ridge_param is not None:
        cfg.ridge_param = args.ridge_param

    run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


if __name__ == "__main__":
    main()





# # #!/usr/bin/env python3
# # """
# # VELOCITY Forecasting: Predict LOT velocity fields, then integrate.

# # Predicts future LOT velocities (in L²(σ) space), then reconstructs 
# # transport maps by integrating from a seed map.

# # PARADIGM:
# #     1. Compute LOT embeddings: T_t = T_σ^{μ_t}
# #     2. Compute velocities: v_t = T_{t+1} - T_t
# #     3. Train reservoir: v_t → v_{t+1} (one-step velocity prediction)
# #     4. Autonomous rollout: feed predicted velocities back as input
# #     5. Integrate: T̂_{t+1} = T_t + v̂_t
# #     6. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

# # KEY CHARACTERISTICS:
# #     - Predicts velocity fields (differences), not positions
# #     - More stationary target: velocity magnitudes stay consistent over time
# #     - Errors accumulate gradually through integration (linear growth)
# #     - Superior to POSITIONS approach for long-horizon forecasting

# # Compare to POSITIONS (positions_forecast.py):
# #     - POSITIONS predicts T_{t+1} directly from T_t
# #     - Non-stationary target, errors compound exponentially

# # Supports both geodesic_transport and swirling_cluster systems.
# # Both systems use 7 reference types.

# # Usage:
# #     from velocity_forecast import run_grid, SystemConfig
    
# #     cfg = SystemConfig(
# #         name="geodesic_transport",
# #         N_list=[500, 1000],
# #         reservoir_scales=[1.0, 2.0],
# #     )
# #     results = run_grid(cfg)
    
# # CLI:
# #     python velocity_forecast.py geodesic_transport
# #     python velocity_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
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

# # # Import warm-up utilities
# # from .warmup_utils import compute_warm_steps_for_velocity, validate_warm_steps


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
# #     """Configuration for VELOCITY forecasting experiments."""
    
# #     name: str  # "geodesic_transport" or "swirling_cluster"
    
# #     # Grid parameters
# #     N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
# #     reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
# #     # LOT reference configuration
# #     kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
# #     fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
# #     # Warm-up configuration (now uses system defaults from warmup_utils)
# #     cycle_length: Optional[int] = None  # Override system default if set
    
# #     # RC hyperparameters (tuned for velocity prediction stability)
# #     spectral_radius: float = 0.7        # Lower for autonomous stability
# #     input_scaling: float = 0.1
# #     leak_rate: float = 0.7              # Leaky integration smooths dynamics
# #     ridge_param: float = 1e-6
# #     random_seed: int = 42
    
# #     # Paths
# #     results_root: Path = field(default_factory=lambda: Path("results"))
# #     forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
# #     lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
# #     # Save options
# #     save_dtype: type = np.float32
# #     save_velocities: bool = True
# #     save_maps: bool = True
    
# #     run_tag: str = "velocity"
    
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
# #             f"velocity_resFromOrigN_{pct}pct{tag}"
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
# #     pred_vel: np.ndarray,
# #     true_vel: np.ndarray,
# #     pred_maps: np.ndarray,
# #     true_maps: np.ndarray,
# #     pred_meas: np.ndarray,
# #     true_meas: np.ndarray,
# #     weights: np.ndarray,
# #     cfg: SystemConfig
# # ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
# #     """Choose what to save based on available disk space."""
# #     dtype = cfg.save_dtype
    
# #     def estimate(save_vel: bool, save_maps: bool) -> int:
# #         total = 0
# #         if save_vel:
# #             total += est_bytes(pred_vel, dtype) + est_bytes(true_vel, dtype)
# #         if save_maps:
# #             total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
# #         total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
# #         total += est_bytes(weights, np.float32)
# #         total += 2 * est_bytes(pred_meas, dtype)
# #         return total
    
# #     ok, info = preflight_space(out_dir, estimate(True, True))
# #     if ok:
# #         return {"save_vel": True, "save_maps": True, "save_meas": True}, info
    
# #     ok, info = preflight_space(out_dir, estimate(False, True))
# #     if ok:
# #         return {"save_vel": False, "save_maps": True, "save_meas": True}, info
    
# #     ok, info = preflight_space(out_dir, estimate(False, False))
# #     if ok:
# #         return {"save_vel": False, "save_maps": False, "save_meas": True}, info
    
# #     raise OSError(f"Insufficient disk space at {out_dir}")


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
# #     Run VELOCITY forecast for single configuration.
    
# #     VELOCITY paradigm: predict velocities v_t → v_{t+1}, then integrate.
    
# #     Returns dict with timing/metrics, or None if skipped/missing.
# #     """
# #     if not (lot_dir / "lot_maps.npy").exists() or not (lot_dir / "velocities.npy").exists():
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
# #     maps = np.load(lot_dir / "lot_maps.npy")    # (T, R, d)
# #     vels = np.load(lot_dir / "velocities.npy")  # (T-1, R, d)
# #     time_load = time.perf_counter() - t0_load
    
# #     T, R, d = maps.shape
# #     assert d == 2, f"Expected d=2, got d={d}"
# #     assert vels.shape[0] == T - 1, f"Velocity shape mismatch: {vels.shape[0]} vs {T-1}"
    
# #     # ── PREP ──
# #     t0_prep = time.perf_counter()
    
# #     # VELOCITY paradigm: predict vel[t+1] from vel[t]
# #     # This is more stationary than predicting maps directly
# #     vel_flat = vels.reshape(T - 1, R * d)
# #     input_seq = vel_flat[:-1]   # v_0, ..., v_{T-3}
# #     target_seq = vel_flat[1:]   # v_1, ..., v_{T-2}
# #     max_steps = input_seq.shape[0]
    
# #     # Use warmup_utils for correct warm-up calculation
# #     warm_steps = compute_warm_steps_for_velocity(
# #         T=T,
# #         system_name=cfg.name,
# #         cycle_length=cfg.cycle_length,
# #         lot_dir=lot_dir,
# #     )
    
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
    
# #     # Forecast horizon: predict velocities vel[warm_steps], ..., vel[T-2]
# #     # This reconstructs maps map[warm_steps+1], ..., map[T-1]
# #     forecast_steps = T - 1 - warm_steps
    
# #     _, pred_vel_flat = rc.run_autonomous(train_input, forecast_steps)
# #     pred_vel = pred_vel_flat.reshape(forecast_steps, R, d)
    
# #     time_rollout = time.perf_counter() - t0_roll
    
# #     # ── MAP RECONSTRUCTION (Integration) ──
# #     # KEY INSIGHT: We integrate predicted velocities to get maps
# #     # This gives linear error growth instead of exponential
# #     t0_maps = time.perf_counter()
    
# #     seed_map = maps[warm_steps]
    
# #     # Integrate: map[t+1] = map[t] + vel[t]
# #     pred_maps = np.empty((forecast_steps + 1, R, d), dtype=maps.dtype)
# #     pred_maps[0] = seed_map
# #     for t in range(forecast_steps):
# #         pred_maps[t + 1] = pred_maps[t] + pred_vel[t]
    
# #     true_future_maps = maps[warm_steps:warm_steps + forecast_steps + 1]
# #     true_future_vels = vels[warm_steps:warm_steps + forecast_steps]
    
# #     time_maps = time.perf_counter() - t0_maps
    
# #     # ── METRICS ──
# #     forecast_vel_rmse = rmse(pred_vel, true_future_vels)
# #     map_rmse = rmse(pred_maps[1:], true_future_maps[1:])
    
# #     # ── PREPARE OUTPUT ──
# #     pred_measures_X = pred_maps[1:]
# #     true_measures_X = true_future_maps[1:]
# #     weights = np.full(R, 1.0 / R, dtype=np.float32)
    
# #     sigma_points = maps[0].copy()
# #     true_disp = true_measures_X - sigma_points
# #     pred_disp = pred_measures_X - sigma_points
    
# #     plan, space_info = choose_save_plan(
# #         out_dir, pred_vel, true_future_vels, pred_maps, true_future_maps,
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
    
# #     if plan["save_vel"]:
# #         atomic_save_npy(out_dir / "predicted_velocities.npy", pred_vel.astype(cfg.save_dtype))
# #         atomic_save_npy(out_dir / "true_future_vels.npy", true_future_vels.astype(cfg.save_dtype))
    
# #     if plan["save_maps"]:
# #         atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
# #         atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
# #     atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
# #     try:
# #         rc.save_configuration(out_dir)
# #     except Exception:
# #         pass
    
# #     time_save = time.perf_counter() - t0_save
# #     time_total = time_load + time_prep + time_train + time_rollout + time_maps + time_save
    
# #     # ── RESULT ──
# #     result = {
# #         "system": cfg.name,
# #         "method": "VELOCITY",
# #         "domain": "lot_velocity",
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
# #         "forecast_vel_rmse": forecast_vel_rmse,
# #         "forecast_map_rmse": map_rmse,
# #         "time_load": time_load,
# #         "time_prep": time_prep,
# #         "time_train": time_train,
# #         "time_rollout": time_rollout,
# #         "time_maps_recon": time_maps,
# #         "time_save": time_save,
# #         "time_total": time_total,
# #         "saved_vel": plan["save_vel"],
# #         "saved_maps": plan["save_maps"],
# #         "output_dir": str(out_dir),
# #     }
    
# #     with open(out_dir / "metadata.json", "w") as f:
# #         json.dump(result, f, indent=2)
    
# #     print(f"[VELOCITY] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
# #           f"train={train_rmse:.6f} vel={forecast_vel_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
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
    
# #     print(f"\n[DONE] VELOCITY grid: {len(results)} runs")
# #     return results


# # # ═══════════════════════════════════════════════════════════════
# # # CLI
# # # ═══════════════════════════════════════════════════════════════

# # def main():
# #     parser = argparse.ArgumentParser(description="VELOCITY forecasting (predict velocities, integrate)")
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
# #     parser.add_argument("--run-tag", type=str, default="velocity")
# #     parser.add_argument("--cycle-length", type=int, default=None, 
# #                         help="Override cycle length (default: auto-detect from system)")
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
# #     )
    
# #     run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


# # if __name__ == "__main__":
# #     main()




# # # #!/usr/bin/env python3
# # # """
# # # VELOCITY Forecasting: Predict LOT velocity fields, then integrate.

# # # Predicts future LOT velocities (in L²(σ) space), then reconstructs 
# # # transport maps by integrating from a seed map.

# # # PARADIGM:
# # #     1. Compute LOT embeddings: T_t = T_σ^{μ_t}
# # #     2. Compute velocities: v_t = T_{t+1} - T_t
# # #     3. Train reservoir: v_t → v_{t+1} (one-step velocity prediction)
# # #     4. Autonomous rollout: feed predicted velocities back as input
# # #     5. Integrate: T̂_{t+1} = T_t + v̂_t
# # #     6. Reconstruct particles: μ̂_{t+1} = (T̂_{t+1})_# σ

# # # KEY CHARACTERISTICS:
# # #     - Predicts velocity fields (differences), not positions
# # #     - More stationary target: velocity magnitudes stay consistent over time
# # #     - Errors accumulate gradually through integration (linear growth)
# # #     - Superior to POSITIONS approach for long-horizon forecasting

# # # Compare to POSITIONS (positions_forecast.py):
# # #     - POSITIONS predicts T_{t+1} directly from T_t
# # #     - Non-stationary target, errors compound exponentially

# # # Supports both geodesic_transport and swirling_cluster systems.
# # # Both systems use 7 reference types.

# # # Usage:
# # #     from velocity_forecast import run_grid, SystemConfig
    
# # #     cfg = SystemConfig(
# # #         name="geodesic_transport",
# # #         N_list=[500, 1000],
# # #         reservoir_scales=[1.0, 2.0],
# # #         cycle_length=100,
# # #     )
# # #     results = run_grid(cfg)
    
# # # CLI:
# # #     python velocity_forecast.py geodesic_transport
# # #     python velocity_forecast.py swirling_cluster --N 500 1000 --kinds clean_circle uniform_square
# # # """

# # # from __future__ import annotations
# # # import os
# # # import time
# # # import json
# # # import csv
# # # import shutil
# # # import platform
# # # import argparse
# # # from pathlib import Path
# # # from dataclasses import dataclass, field
# # # from typing import List, Optional, Dict, Any, Tuple
# # # import numpy as np


# # # # ═══════════════════════════════════════════════════════════════
# # # # CONFIGURATION
# # # # ═══════════════════════════════════════════════════════════════

# # # DEFAULT_KINDS = [
# # #     "clean_circle_small",
# # #     "clean_circle_large",
# # #     "uniform_square",
# # #     "gaussian_iso",
# # #     "snapshot_begin",
# # #     "snapshot_middle",
# # #     "snapshot_end",
# # # ]

# # # DEFAULT_FRACTIONS = [10, 25, 50, 75, 100]


# # # @dataclass
# # # class SystemConfig:
# # #     """Configuration for VELOCITY forecasting experiments."""
    
# # #     name: str  # "geodesic_transport" or "swirling_cluster"
    
# # #     # Grid parameters
# # #     N_list: List[int] = field(default_factory=lambda: [100, 250, 500, 750, 1000, 1500])
# # #     reservoir_scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 5.0])
    
# # #     # LOT reference configuration
# # #     kinds: List[str] = field(default_factory=lambda: DEFAULT_KINDS.copy())
# # #     fractions: List[int] = field(default_factory=lambda: DEFAULT_FRACTIONS.copy())
    
# # #     # Warm-up configuration
# # #     cycle_length: Optional[int] = None  # L: length of one cycle (e.g., 100 for one forward sweep)
# # #     align_to_boundary: bool = True      # align autonomous start to cycle boundary
# # #     warm_steps_default: int = 50        # fallback if cycle_length not set
    
# # #     # RC hyperparameters (tuned for velocity prediction stability)
# # #     spectral_radius: float = 0.7        # Lower for autonomous stability
# # #     input_scaling: float = 0.1
# # #     leak_rate: float = 0.7              # Leaky integration smooths dynamics
# # #     ridge_param: float = 1e-6
# # #     random_seed: int = 42
    
# # #     # Paths
# # #     results_root: Path = field(default_factory=lambda: Path("results"))
# # #     forecast_root: Path = field(default_factory=lambda: Path("forecast_output"))
# # #     lot_root: Path = field(default_factory=lambda: Path("lot_maps"))
    
# # #     # Save options
# # #     save_dtype: type = np.float32
# # #     save_velocities: bool = True
# # #     save_maps: bool = True
    
# # #     run_tag: str = "velocity"
    
# # #     def __post_init__(self):
# # #         self.results_root = Path(self.results_root)
# # #         self.forecast_root = Path(self.forecast_root)
# # #         self.lot_root = Path(self.lot_root)
    
# # #     def lot_dir(self, N: int, kind: str, frac: int) -> Path:
# # #         return self.lot_root / f"{self.name}_N_{N}" / kind / f"frac{frac}"
    
# # #     def output_dir(self, N: int, kind: str, frac: int, reservoir_scale: float) -> Path:
# # #         pct = int(reservoir_scale * 100)
# # #         tag = f"_{self.run_tag}" if self.run_tag else ""
# # #         return (
# # #             self.forecast_root / f"{self.name}_N_{N}" / kind / f"frac{frac}" /
# # #             f"velocity_resFromOrigN_{pct}pct{tag}"
# # #         )


# # # # ═══════════════════════════════════════════════════════════════
# # # # UTILITIES
# # # # ═══════════════════════════════════════════════════════════════

# # # def env_banner() -> None:
# # #     print(f"[env] Python {platform.python_version()} | NumPy {np.__version__} | "
# # #           f"OMP={os.environ.get('OMP_NUM_THREADS', '?')} MKL={os.environ.get('MKL_NUM_THREADS', '?')}")


# # # def rmse(a: np.ndarray, b: np.ndarray) -> float:
# # #     return float(np.sqrt(np.mean((a - b) ** 2)))


# # # def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
# # #     """Atomic save to avoid corrupt partial files."""
# # #     path = Path(path)
# # #     path.parent.mkdir(parents=True, exist_ok=True)
# # #     tmp = path.parent / f".tmp_{path.name}"
# # #     np.save(tmp, arr)
# # #     tmp_actual = tmp if tmp.exists() else tmp.parent / (tmp.name + ".npy")
# # #     os.replace(tmp_actual, path)


# # # def minimal_run_exists(out_dir: Path) -> bool:
# # #     """Check if essential outputs exist."""
# # #     return all((out_dir / f).exists() for f in [
# # #         "predicted_measures_X.npy", "true_measures_X.npy", "measures_w.npy"
# # #     ])


# # # def clean_incomplete_run(out_dir: Path) -> None:
# # #     if out_dir.exists() and not minimal_run_exists(out_dir):
# # #         print(f"[CLEAN] Removing incomplete: {out_dir}")
# # #         shutil.rmtree(out_dir, ignore_errors=True)


# # # def append_csv_row(path: Path, header: List[str], row: List[Any]) -> None:
# # #     path.parent.mkdir(parents=True, exist_ok=True)
# # #     write_header = not path.exists()
# # #     with open(path, "a", newline="") as f:
# # #         w = csv.writer(f)
# # #         if write_header:
# # #             w.writerow(header)
# # #         w.writerow(row)


# # # def est_bytes(arr: np.ndarray, dtype=None) -> int:
# # #     dt = np.dtype(dtype) if dtype is not None else arr.dtype
# # #     return int(arr.size * dt.itemsize)


# # # def preflight_space(dest_dir: Path, bytes_needed: int, margin: float = 1.15) -> Tuple[bool, Dict]:
# # #     dest_dir = Path(dest_dir)
# # #     dest_dir.mkdir(parents=True, exist_ok=True)
# # #     total, used, free = shutil.disk_usage(dest_dir)
# # #     need = int(bytes_needed * margin)
# # #     return free >= need, {"free": free, "need": need, "where": str(dest_dir)}


# # # def choose_save_plan(
# # #     out_dir: Path,
# # #     pred_vel: np.ndarray,
# # #     true_vel: np.ndarray,
# # #     pred_maps: np.ndarray,
# # #     true_maps: np.ndarray,
# # #     pred_meas: np.ndarray,
# # #     true_meas: np.ndarray,
# # #     weights: np.ndarray,
# # #     cfg: SystemConfig
# # # ) -> Tuple[Dict[str, bool], Dict[str, Any]]:
# # #     """Choose what to save based on available disk space."""
# # #     dtype = cfg.save_dtype
    
# # #     def estimate(save_vel: bool, save_maps: bool) -> int:
# # #         total = 0
# # #         if save_vel:
# # #             total += est_bytes(pred_vel, dtype) + est_bytes(true_vel, dtype)
# # #         if save_maps:
# # #             total += est_bytes(pred_maps, dtype) + est_bytes(true_maps, dtype)
# # #         total += est_bytes(pred_meas, dtype) + est_bytes(true_meas, dtype)
# # #         total += est_bytes(weights, np.float32)
# # #         total += 2 * est_bytes(pred_meas, dtype)
# # #         return total
    
# # #     ok, info = preflight_space(out_dir, estimate(True, True))
# # #     if ok:
# # #         return {"save_vel": True, "save_maps": True, "save_meas": True}, info
    
# # #     ok, info = preflight_space(out_dir, estimate(False, True))
# # #     if ok:
# # #         return {"save_vel": False, "save_maps": True, "save_meas": True}, info
    
# # #     ok, info = preflight_space(out_dir, estimate(False, False))
# # #     if ok:
# # #         return {"save_vel": False, "save_maps": False, "save_meas": True}, info
    
# # #     raise OSError(f"Insufficient disk space at {out_dir}")


# # # def compute_warm_steps(max_steps: int, cfg: SystemConfig) -> int:
# # #     """
# # #     Compute warm_steps based on cycle configuration.
    
# # #     For VELOCITY (LOT velocities), we use cycle_length - 1 since:
# # #     - velocity[t] corresponds to map[t] -> map[t+1]
# # #     - One cycle of L maps has L-1 velocity transitions
    
# # #     If cycle_length is set and align_to_boundary is True:
# # #         warm_steps = cycle_length - 1 (train on one full cycle of velocities)
# # #     Otherwise:
# # #         warm_steps = warm_steps_default (fallback, typically 50)
# # #     """
# # #     if cfg.cycle_length is not None and cfg.align_to_boundary:
# # #         return max(1, min(cfg.cycle_length - 1, max_steps - 1))
# # #     return max(1, min(cfg.warm_steps_default, max_steps - 1))


# # # # ═══════════════════════════════════════════════════════════════
# # # # CORE PIPELINE
# # # # ═══════════════════════════════════════════════════════════════

# # # def run_single(
# # #     lot_dir: Path,
# # #     out_dir: Path,
# # #     N: int,
# # #     kind: str,
# # #     frac: int,
# # #     reservoir_scale: float,
# # #     cfg: SystemConfig,
# # #     skip_existing: bool = False,
# # # ) -> Optional[Dict[str, Any]]:
# # #     """
# # #     Run VELOCITY forecast for single configuration.
    
# # #     VELOCITY paradigm: predict velocities v_t → v_{t+1}, then integrate.
    
# # #     Returns dict with timing/metrics, or None if skipped/missing.
# # #     """
# # #     if not (lot_dir / "lot_maps.npy").exists() or not (lot_dir / "velocities.npy").exists():
# # #         print(f"[MISS] {lot_dir}")
# # #         return None
    
# # #     if skip_existing and minimal_run_exists(out_dir):
# # #         print(f"[SKIP] {out_dir}")
# # #         return None
    
# # #     clean_incomplete_run(out_dir)
# # #     out_dir.mkdir(parents=True, exist_ok=True)
    
# # #     # Import RC with correct API
# # #     from rc_computer import ReservoirComputer, ReservoirConfig, InitScheme
    
# # #     # ── LOAD ──
# # #     t0_load = time.perf_counter()
# # #     maps = np.load(lot_dir / "lot_maps.npy")    # (T, R, d)
# # #     vels = np.load(lot_dir / "velocities.npy")  # (T-1, R, d)
# # #     time_load = time.perf_counter() - t0_load
    
# # #     T, R, d = maps.shape
# # #     assert d == 2, f"Expected d=2, got d={d}"
# # #     assert vels.shape[0] == T - 1, f"Velocity shape mismatch: {vels.shape[0]} vs {T-1}"
    
# # #     # ── PREP ──
# # #     t0_prep = time.perf_counter()
    
# # #     # VELOCITY paradigm: predict vel[t+1] from vel[t]
# # #     # This is more stationary than predicting maps directly
# # #     vel_flat = vels.reshape(T - 1, R * d)
# # #     input_seq = vel_flat[:-1]   # v_0, ..., v_{T-3}
# # #     target_seq = vel_flat[1:]   # v_1, ..., v_{T-2}
# # #     max_steps = input_seq.shape[0]
    
# # #     warm_steps = compute_warm_steps(max_steps, cfg)
    
# # #     train_input = input_seq[:warm_steps]
# # #     train_target = target_seq[:warm_steps]
    
# # #     time_prep = time.perf_counter() - t0_prep
    
# # #     # ── TRAIN ──
# # #     t0_train = time.perf_counter()
# # #     reservoir_size = max(1, int(round(reservoir_scale * N)))
    
# # #     rc_config = ReservoirConfig(
# # #         input_size=R * d,
# # #         reservoir_size=reservoir_size,
# # #         output_size=R * d,
# # #         spectral_radius=cfg.spectral_radius,
# # #         input_scaling=cfg.input_scaling,
# # #         leak_rate=cfg.leak_rate,
# # #         ridge_param=cfg.ridge_param,
# # #         activation="tanh",
# # #         init_scheme=InitScheme.SPARSE,
# # #         random_seed=cfg.random_seed,
# # #     )
# # #     rc = ReservoirComputer(rc_config)
    
# # #     states = rc.run(train_input)
# # #     rc.train(states, train_target)
    
# # #     # Training RMSE
# # #     train_pred = rc.predict(states)
# # #     train_rmse = rmse(train_pred, train_target)
    
# # #     time_train = time.perf_counter() - t0_train
    
# # #     # ── ROLLOUT ──
# # #     t0_roll = time.perf_counter()
    
# # #     # Forecast horizon: predict velocities vel[warm_steps], ..., vel[T-2]
# # #     # This reconstructs maps map[warm_steps+1], ..., map[T-1]
# # #     forecast_steps = T - 1 - warm_steps
    
# # #     _, pred_vel_flat = rc.run_autonomous(train_input, forecast_steps)
# # #     pred_vel = pred_vel_flat.reshape(forecast_steps, R, d)
    
# # #     time_rollout = time.perf_counter() - t0_roll
    
# # #     # ── MAP RECONSTRUCTION (Integration) ──
# # #     # KEY INSIGHT: We integrate predicted velocities to get maps
# # #     # This gives linear error growth instead of exponential
# # #     t0_maps = time.perf_counter()
    
# # #     seed_map = maps[warm_steps]
    
# # #     # Integrate: map[t+1] = map[t] + vel[t]
# # #     pred_maps = np.empty((forecast_steps + 1, R, d), dtype=maps.dtype)
# # #     pred_maps[0] = seed_map
# # #     for t in range(forecast_steps):
# # #         pred_maps[t + 1] = pred_maps[t] + pred_vel[t]
    
# # #     true_future_maps = maps[warm_steps:warm_steps + forecast_steps + 1]
# # #     true_future_vels = vels[warm_steps:warm_steps + forecast_steps]
    
# # #     time_maps = time.perf_counter() - t0_maps
    
# # #     # ── METRICS ──
# # #     forecast_vel_rmse = rmse(pred_vel, true_future_vels)
# # #     map_rmse = rmse(pred_maps[1:], true_future_maps[1:])
    
# # #     # ── PREPARE OUTPUT ──
# # #     pred_measures_X = pred_maps[1:]
# # #     true_measures_X = true_future_maps[1:]
# # #     weights = np.full(R, 1.0 / R, dtype=np.float32)
    
# # #     sigma_points = maps[0].copy()
# # #     true_disp = true_measures_X - sigma_points
# # #     pred_disp = pred_measures_X - sigma_points
    
# # #     plan, space_info = choose_save_plan(
# # #         out_dir, pred_vel, true_future_vels, pred_maps, true_future_maps,
# # #         pred_measures_X, true_measures_X, weights, cfg
# # #     )
    
# # #     # ── SAVE ──
# # #     t0_save = time.perf_counter()
    
# # #     atomic_save_npy(out_dir / "predicted_measures_X.npy", pred_measures_X.astype(cfg.save_dtype))
# # #     atomic_save_npy(out_dir / "true_measures_X.npy", true_measures_X.astype(cfg.save_dtype))
# # #     atomic_save_npy(out_dir / "measures_w.npy", weights)
# # #     atomic_save_npy(out_dir / "reference_sigma_X.npy", sigma_points.astype(cfg.save_dtype))
# # #     atomic_save_npy(out_dir / "true_displacements.npy", true_disp.astype(cfg.save_dtype))
# # #     atomic_save_npy(out_dir / "pred_displacements.npy", pred_disp.astype(cfg.save_dtype))
    
# # #     if plan["save_vel"]:
# # #         atomic_save_npy(out_dir / "predicted_velocities.npy", pred_vel.astype(cfg.save_dtype))
# # #         atomic_save_npy(out_dir / "true_future_vels.npy", true_future_vels.astype(cfg.save_dtype))
    
# # #     if plan["save_maps"]:
# # #         atomic_save_npy(out_dir / "predicted_maps.npy", pred_maps.astype(cfg.save_dtype))
# # #         atomic_save_npy(out_dir / "true_future_maps.npy", true_future_maps.astype(cfg.save_dtype))
    
# # #     atomic_save_npy(out_dir / "warm_start_info.npy", np.array([warm_steps, forecast_steps], dtype=np.int32))
    
# # #     try:
# # #         rc.save_configuration(out_dir)
# # #     except Exception:
# # #         pass
    
# # #     time_save = time.perf_counter() - t0_save
# # #     time_total = time_load + time_prep + time_train + time_rollout + time_maps + time_save
    
# # #     # ── RESULT ──
# # #     result = {
# # #         "system": cfg.name,
# # #         "method": "VELOCITY",
# # #         "domain": "lot_velocity",
# # #         "N": N,
# # #         "R": R,
# # #         "kind": kind,
# # #         "frac_pct": frac,
# # #         "reservoir_scale": reservoir_scale,
# # #         "reservoir_size": reservoir_size,
# # #         "T": T,
# # #         "d": d,
# # #         "cycle_length": cfg.cycle_length,
# # #         "warm_steps": warm_steps,
# # #         "forecast_steps": forecast_steps,
# # #         "train_rmse": train_rmse,
# # #         "forecast_vel_rmse": forecast_vel_rmse,
# # #         "forecast_map_rmse": map_rmse,
# # #         "time_load": time_load,
# # #         "time_prep": time_prep,
# # #         "time_train": time_train,
# # #         "time_rollout": time_rollout,
# # #         "time_maps_recon": time_maps,
# # #         "time_save": time_save,
# # #         "time_total": time_total,
# # #         "saved_vel": plan["save_vel"],
# # #         "saved_maps": plan["save_maps"],
# # #         "output_dir": str(out_dir),
# # #     }
    
# # #     with open(out_dir / "metadata.json", "w") as f:
# # #         json.dump(result, f, indent=2)
    
# # #     print(f"[VELOCITY] N={N} {kind}/{frac}% res={reservoir_size} warm={warm_steps} | "
# # #           f"train={train_rmse:.6f} vel={forecast_vel_rmse:.6f} map={map_rmse:.6f} | {time_total:.2f}s")
    
# # #     return result


# # # def run_grid(
# # #     cfg: SystemConfig,
# # #     skip_existing: bool = False,
# # #     csv_path: Optional[Path] = None,
# # # ) -> List[Dict[str, Any]]:
# # #     """Run full parameter grid: N × kinds × fractions × reservoir_scales."""
# # #     env_banner()
# # #     np.random.seed(cfg.random_seed)
    
# # #     results = []
    
# # #     for N in cfg.N_list:
# # #         for kind in cfg.kinds:
# # #             for frac in cfg.fractions:
# # #                 lot_dir = cfg.lot_dir(N, kind, frac)
                
# # #                 for scale in cfg.reservoir_scales:
# # #                     out_dir = cfg.output_dir(N, kind, frac, scale)
                    
# # #                     result = run_single(
# # #                         lot_dir, out_dir, N, kind, frac, scale, cfg, skip_existing
# # #                     )
                    
# # #                     if result is not None:
# # #                         results.append(result)
# # #                         if csv_path:
# # #                             append_csv_row(csv_path, list(result.keys()), list(result.values()))
    
# # #     print(f"\n[DONE] VELOCITY grid: {len(results)} runs")
# # #     return results


# # # # ═══════════════════════════════════════════════════════════════
# # # # CLI
# # # # ═══════════════════════════════════════════════════════════════

# # # def main():
# # #     parser = argparse.ArgumentParser(description="VELOCITY forecasting (predict velocities, integrate)")
# # #     parser.add_argument("system", choices=["geodesic_transport", "swirling_cluster"])
# # #     parser.add_argument("--N", type=int, nargs="+", default=None)
# # #     parser.add_argument("--scales", type=float, nargs="+", default=None)
# # #     parser.add_argument("--kinds", type=str, nargs="+", default=None)
# # #     parser.add_argument("--fracs", type=int, nargs="+", default=None)
# # #     parser.add_argument("--skip-existing", action="store_true")
# # #     parser.add_argument("--csv", type=Path, default=None)
# # #     parser.add_argument("--results-root", type=Path, default=Path("results"))
# # #     parser.add_argument("--output-root", type=Path, default=Path("forecast_output"))
# # #     parser.add_argument("--lot-root", type=Path, default=Path("lot_maps"))
# # #     parser.add_argument("--run-tag", type=str, default="velocity")
# # #     parser.add_argument("--cycle-length", type=int, default=None, help="Length of one cycle (e.g., 100)")
# # #     parser.add_argument("--warm-steps", type=int, default=50, help="Fallback warm steps if cycle_length not set")
# # #     parser.add_argument("--no-align", action="store_true", help="Disable boundary alignment")
# # #     args = parser.parse_args()
    
# # #     cfg = SystemConfig(
# # #         name=args.system,
# # #         N_list=args.N or [100, 250, 500, 750, 1000, 1500],
# # #         reservoir_scales=args.scales or [0.5, 1.0, 1.5, 2.0, 5.0],
# # #         kinds=args.kinds or DEFAULT_KINDS,
# # #         fractions=args.fracs or DEFAULT_FRACTIONS,
# # #         results_root=args.results_root,
# # #         forecast_root=args.output_root,
# # #         lot_root=args.lot_root,
# # #         run_tag=args.run_tag,
# # #         cycle_length=args.cycle_length,
# # #         align_to_boundary=not args.no_align,
# # #         warm_steps_default=args.warm_steps,
# # #     )
    
# # #     run_grid(cfg, skip_existing=args.skip_existing, csv_path=args.csv)


# # # if __name__ == "__main__":
# # #     main()