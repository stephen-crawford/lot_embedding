#!/usr/bin/env python3
"""
End-to-end pipeline: trajectory → LOT embeddings → VELOCITY vs POSITIONS forecasts.

Default: synthetic breathing-spiral cloud (SwirlingClusterSystem), normalized to
[0, 1]^2 — no network required. Mirrors the real-world path (particles → LOT → RC).

Usage (from repository root, venv activated):
    python scripts/realworld_experiment_pipeline.py
    python scripts/realworld_experiment_pipeline.py --quick

Real GOES (writes under repo ``goes_data/``; fetch if missing):
    python scripts/realworld_experiment_pipeline.py --real-goes --fetch-goes --N 500

Or use an existing export:
    python scripts/realworld_experiment_pipeline.py --goes-dir /path/to/goes_data

Expects PYTHONPATH to include ``src`` or run with repo root as cwd (script adds src).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _normalize01(traj: np.ndarray, margin: float = 0.02) -> np.ndarray:
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def build_synthetic_trajectory(n_particles: int, n_steps: int, n_cycles: int) -> np.ndarray:
    from data_utils.simulation.measure_dynamical_systems import SwirlingClusterSystem

    sys = SwirlingClusterSystem(N=n_particles, noise_scale=0.02, seed=42)
    raw = sys.get_cyclical_trajectory(n_cycles=n_cycles, n_steps=n_steps, max_radius=3.5)
    return _normalize01(raw.astype(np.float32))


def load_goes_trajectory(goes_dir: Path) -> np.ndarray:
    p = goes_dir / "particles.npy"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run: python scripts/download_realworld_data.py --source goes "
            f"--output-goes {goes_dir}   or pass --fetch-goes on this script."
        )
    traj = np.load(p)
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected (T, N, 2), got {traj.shape}")
    return _normalize01(traj.astype(np.float32))


def run_pipeline(
    *,
    system: str,
    n_particles: int,
    n_steps: int,
    n_cycles: int,
    goes_dir: Path | None,
    quick: bool,
    ot_method: str,
    ot_device: str,
    rc_backend: str,
    lot_kind: str = "gaussian_iso",
    lot_frac: int = 100,
    assignment: str = "fixed",
    anchor_frame: int = 0,
    reservoir_scale: float = 1.0,
    run_tag_velocity: str = "pipeline_velocity",
    run_tag_positions: str = "pipeline_positions",
    spectral_radius_vel: float | None = None,
    leak_rate_vel: float | None = None,
    ridge_vel: float | None = None,
    spectral_radius_pos: float | None = None,
    leak_rate_pos: float | None = None,
    ridge_pos: float | None = None,
    fetch_goes_if_missing: bool = False,
    goes_hours: int = 24,
    goes_date: str = "2024-07-15",
    goes_hour_utc: int = 18,
    damping: float = 1.0,
    decay: float = 1.0,
    nudge_interval: int = 0,
    nudge_strength: float = 0.5,
) -> dict:
    results_root = REPO_ROOT / "results"
    lot_root = REPO_ROOT / "lot_maps"
    forecast_root = REPO_ROOT / "forecast_output"

    if quick:
        n_particles = min(n_particles, 60)
        n_steps = min(n_steps, 80)
        n_cycles = 2

    if goes_dir is not None:
        goes_dir = Path(goes_dir)
        particles_path = goes_dir / "particles.npy"
        if not particles_path.is_file():
            if fetch_goes_if_missing:
                from data_utils.goes_cloud_data import fetch_goes_particles_bundle

                fetch_goes_particles_bundle(
                    goes_dir,
                    hours=goes_hours,
                    n_particles=n_particles,
                    date=goes_date,
                    hour_utc=goes_hour_utc,
                    verbose=True,
                )
            else:
                raise FileNotFoundError(
                    f"Missing GOES particles at {particles_path}.\n"
                    f"Install real-world deps (pip install -r requirements-realworld.txt), then either:\n"
                    f"  python scripts/download_realworld_data.py --source goes --output-goes {goes_dir} "
                    f"--hours {goes_hours} --n-particles {n_particles} --date {goes_date} --hour-utc {goes_hour_utc}\n"
                    "or re-run with --fetch-goes (same effect, in-process)."
                )
        traj = load_goes_trajectory(goes_dir)
        t_go, n_go, _d = traj.shape
        n_steps = min(n_steps, t_go)
        traj = traj[:n_steps]
        n_actual = n_go
    else:
        traj = build_synthetic_trajectory(n_particles, n_steps, n_cycles)
        t_actual, n_actual, _ = traj.shape

    out_traj = results_root / f"{system}_N_{n_actual}"
    out_traj.mkdir(parents=True, exist_ok=True)
    traj_path = out_traj / "trajectory.npy"
    np.save(traj_path, traj)
    t_actual, n_actual, _ = traj.shape
    cycle_length = max(1, t_actual // max(1, n_cycles))

    from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings

    generate_lot_embeddings(
        system,
        N_list=[n_actual],
        kinds=[lot_kind],
        fractions=[lot_frac],
        assignment=assignment,
        anchor_frame=anchor_frame,
        results_root=results_root,
        lot_root=lot_root,
        verbose=False,
        ot_method=ot_method,
        ot_device=ot_device,
        sinkhorn_reg=0.01,
        sinkhorn_iters=120 if ot_method == "sinkhorn" else 150,
        sinkhorn_batch_frames=32,
    )

    lot_dir = lot_root / f"{system}_N_{n_actual}" / lot_kind / f"frac{lot_frac}"

    from run_forecast.positions_forecast import SystemConfig as PosCfg, run_single as pos_run
    from run_forecast.velocity_forecast import SystemConfig as VelCfg, run_single as vel_run

    vel_kw = {}
    if spectral_radius_vel is not None:
        vel_kw["spectral_radius"] = spectral_radius_vel
    if leak_rate_vel is not None:
        vel_kw["leak_rate"] = leak_rate_vel
    if ridge_vel is not None:
        vel_kw["ridge_param"] = ridge_vel

    pos_kw = {}
    if spectral_radius_pos is not None:
        pos_kw["spectral_radius"] = spectral_radius_pos
    if leak_rate_pos is not None:
        pos_kw["leak_rate"] = leak_rate_pos
    if ridge_pos is not None:
        pos_kw["ridge_param"] = ridge_pos

    vel_cfg = VelCfg(
        name=system,
        N_list=[n_actual],
        kinds=[lot_kind],
        fractions=[lot_frac],
        reservoir_scales=[reservoir_scale],
        results_root=results_root,
        forecast_root=forecast_root,
        lot_root=lot_root,
        cycle_length=cycle_length,
        boundary_mode="clip",
        rc_backend=rc_backend,
        rc_device="cpu",
        run_tag=run_tag_velocity,
        damping=damping,
        decay=decay,
        nudge_interval=nudge_interval,
        nudge_strength=nudge_strength,
        **vel_kw,
    )
    pos_cfg = PosCfg(
        name=system,
        N_list=[n_actual],
        kinds=[lot_kind],
        fractions=[lot_frac],
        reservoir_scales=[reservoir_scale],
        results_root=results_root,
        forecast_root=forecast_root,
        lot_root=lot_root,
        cycle_length=cycle_length,
        boundary_mode="clip",
        rc_backend=rc_backend,
        rc_device="cpu",
        run_tag=run_tag_positions,
        **pos_kw,
    )

    vel_out = vel_cfg.output_dir(n_actual, lot_kind, lot_frac, reservoir_scale)
    pos_out = pos_cfg.output_dir(n_actual, lot_kind, lot_frac, reservoir_scale)

    rv = vel_run(
        lot_dir,
        vel_out,
        n_actual,
        lot_kind,
        lot_frac,
        reservoir_scale,
        vel_cfg,
        skip_existing=False,
    )
    rp = pos_run(
        lot_dir,
        pos_out,
        n_actual,
        lot_kind,
        lot_frac,
        reservoir_scale,
        pos_cfg,
        skip_existing=False,
    )

    if rv is None or rp is None:
        raise RuntimeError("Forecast run failed (missing LOT data or error); check stdout.")

    with open(vel_out / "metadata.json") as f:
        mv = json.load(f)
    with open(pos_out / "metadata.json") as f:
        mp = json.load(f)

    # ── LOT-space RMSE (existing metric) ──
    v_rmse = float(mv["forecast_map_rmse"])
    p_rmse = float(mp["forecast_map_rmse"])
    ratio_rmse = p_rmse / max(v_rmse, 1e-12)
    vel_wins_rmse = v_rmse < p_rmse

    # ── Debiased Sinkhorn divergence (paper Section 6.1, Eq. 12) ──
    # Measures W_2^2 approximation directly on reconstructed particle clouds,
    # independent of the LOT embedding geometry.
    vel_pred_X = np.load(vel_out / "predicted_measures_X.npy")
    vel_true_X = np.load(vel_out / "true_measures_X.npy")
    pos_pred_X = np.load(pos_out / "predicted_measures_X.npy")
    pos_true_X = np.load(pos_out / "true_measures_X.npy")

    # Adaptive epsilon: 0.05 * median pairwise squared distance in reference
    sigma_path = vel_out / "reference_sigma_X.npy"
    if sigma_path.exists():
        sigma_pts = np.load(sigma_path).astype(np.float64)
        dists = np.sum(
            (sigma_pts[:, None, :] - sigma_pts[None, :, :]) ** 2, axis=-1
        )
        sink_eps = float(np.median(dists)) * 0.05
    else:
        sink_eps = 0.01  # fallback

    sink_eps = max(sink_eps, 1e-4)  # floor to avoid numerical issues

    try:
        import ot as pot  # noqa: F811

        def _sinkhorn_div_per_step(pred_X, true_X, eps):
            """Debiased Sinkhorn divergence at each forecast timestep."""
            T_h, N_h, d_h = pred_X.shape
            w = np.full(N_h, 1.0 / N_h, dtype=np.float64)
            divs = np.empty(T_h, dtype=np.float64)
            for t in range(T_h):
                Xa = pred_X[t].astype(np.float64)
                Xb = true_X[t].astype(np.float64)
                M_ab = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)
                M_aa = np.sum((Xa[:, None, :] - Xa[None, :, :]) ** 2, axis=-1)
                M_bb = np.sum((Xb[:, None, :] - Xb[None, :, :]) ** 2, axis=-1)
                W_ab = float(pot.sinkhorn2(w, w, M_ab, reg=eps, numItermax=5000))
                W_aa = float(pot.sinkhorn2(w, w, M_aa, reg=eps, numItermax=5000))
                W_bb = float(pot.sinkhorn2(w, w, M_bb, reg=eps, numItermax=5000))
                divs[t] = max(W_ab - 0.5 * (W_aa + W_bb), 0.0)
            return divs

        vel_sink_divs = _sinkhorn_div_per_step(vel_pred_X, vel_true_X, sink_eps)
        pos_sink_divs = _sinkhorn_div_per_step(pos_pred_X, pos_true_X, sink_eps)

        # Mean Sinkhorn divergence (approximates time-averaged W_2^2)
        vel_sink_mean = float(np.mean(vel_sink_divs))
        pos_sink_mean = float(np.mean(pos_sink_divs))

        # sqrt(mean Sinkhorn) as W_2 surrogate (paper Eq. 11)
        vel_sink_w2 = float(np.sqrt(vel_sink_mean))
        pos_sink_w2 = float(np.sqrt(pos_sink_mean))

        ratio_sink = pos_sink_w2 / max(vel_sink_w2, 1e-12)
        vel_wins_sinkhorn = vel_sink_w2 < pos_sink_w2

        sinkhorn_results = {
            "sinkhorn_epsilon": sink_eps,
            "sinkhorn_mean_velocity": vel_sink_mean,
            "sinkhorn_mean_positions": pos_sink_mean,
            "sinkhorn_w2_velocity": vel_sink_w2,
            "sinkhorn_w2_positions": pos_sink_w2,
            "sinkhorn_positions_over_velocity_ratio": ratio_sink,
            "velocity_wins_sinkhorn": vel_wins_sinkhorn,
            "sinkhorn_per_step_velocity": vel_sink_divs.tolist(),
            "sinkhorn_per_step_positions": pos_sink_divs.tolist(),
        }
        print(
            f"[SINKHORN] eps={sink_eps:.6f} | "
            f"velocity W2={vel_sink_w2:.6f}  positions W2={pos_sink_w2:.6f}  "
            f"ratio={ratio_sink:.2f}  winner={'VELOCITY' if vel_wins_sinkhorn else 'POSITIONS'}"
        )

    except ImportError:
        print("[WARN] POT not installed — skipping Sinkhorn evaluation (pip install POT)")
        sinkhorn_results = {"sinkhorn_error": "POT library not installed"}
        vel_wins_sinkhorn = None

    # ── Combined winner ──
    vel_wins = vel_wins_sinkhorn if vel_wins_sinkhorn is not None else vel_wins_rmse

    summary = {
        "system": system,
        "trajectory_shape": list(traj.shape),
        "trajectory_path": str(traj_path),
        "lot_dir": str(lot_dir),
        "lot_kind": lot_kind,
        "lot_frac": lot_frac,
        "lot_assignment": assignment,
        "reservoir_scale": reservoir_scale,
        "velocity_forecast_dir": str(vel_out),
        "positions_forecast_dir": str(pos_out),
        # LOT-space RMSE
        "forecast_map_rmse_velocity": v_rmse,
        "forecast_map_rmse_positions": p_rmse,
        "positions_over_velocity_ratio_rmse": ratio_rmse,
        "velocity_wins_map_rmse": vel_wins_rmse,
        # Sinkhorn W2
        **sinkhorn_results,
        # Overall
        "velocity_wins": vel_wins,
        "comparison_metric": "sinkhorn_w2" if vel_wins_sinkhorn is not None else "lot_embedding_map_rmse",
        "cycle_length": cycle_length,
        "ot_method": ot_method,
    }
    summary_path = REPO_ROOT / "pipeline_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="LOT+RC real-world-style experiment pipeline")
    ap.add_argument("--system", default="pipeline_cloud_demo", help="Experiment name / folder prefix")
    ap.add_argument("--N", type=int, default=100, dest="n_particles", help="Particle count")
    ap.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Time steps (synthetic); ~200+ with 2 cycles helps VELOCITY beat POSITIONS on the demo",
    )
    ap.add_argument("--cycles", type=int, default=2, help="Breathing cycles (synthetic)")
    ap.add_argument(
        "--real-goes",
        action="store_true",
        help="Use repository goes_data/ as trajectory (real GOES export layout)",
    )
    ap.add_argument("--goes-dir", type=Path, default=None, help="Folder with particles.npy from GOES export")
    ap.add_argument(
        "--fetch-goes",
        action="store_true",
        help="If particles.npy is missing, download GOES into --goes-dir (needs goes2go + network)",
    )
    ap.add_argument("--goes-hours", type=int, default=24, help="With --fetch-goes: hours of GOES to pull")
    ap.add_argument("--goes-date", default="2024-07-15", help="With --fetch-goes: UTC date YYYY-MM-DD")
    ap.add_argument("--goes-hour-utc", type=int, default=18, help="With --fetch-goes: start hour UTC")
    ap.add_argument("--quick", action="store_true", help="Smaller N/steps for fast CI")
    ap.add_argument("--ot-method", choices=["emd", "sinkhorn"], default="emd")
    ap.add_argument("--ot-device", default="cpu", help="cpu or cuda (sinkhorn only)")
    ap.add_argument("--rc-backend", choices=["auto", "numpy", "torch"], default="numpy")
    ap.add_argument("--lot-kind", default="gaussian_iso")
    ap.add_argument("--lot-frac", type=int, default=100)
    ap.add_argument("--assignment", choices=["fixed", "per_frame"], default="per_frame")
    ap.add_argument("--reservoir-scale", type=float, default=1.0)
    ap.add_argument("--run-tag-velocity", default="pipeline_velocity")
    ap.add_argument("--run-tag-positions", default="pipeline_positions")
    # Velocity damping (pulls forecast toward zero velocity over horizon)
    ap.add_argument("--damping", type=float, default=1.0,
                    help="Initial velocity damping factor (1.0 = no damping)")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="Per-step decay for damping (alpha_t = damping * decay^t)")
    # Reservoir state nudging (blend toward observation-driven state)
    ap.add_argument("--nudge-interval", type=int, default=0,
                    help="Nudge every K forecast steps (0 = disabled)")
    ap.add_argument("--nudge-strength", type=float, default=0.5,
                    help="Blend factor for nudging (0=autonomous, 1=full observation)")
    args = ap.parse_args()

    goes_dir = REPO_ROOT / "goes_data" if args.real_goes else args.goes_dir

    summary = run_pipeline(
        system=args.system,
        n_particles=args.n_particles,
        n_steps=args.steps,
        n_cycles=args.cycles,
        goes_dir=goes_dir,
        quick=args.quick,
        ot_method=args.ot_method,
        ot_device=args.ot_device,
        rc_backend=args.rc_backend,
        lot_kind=args.lot_kind,
        lot_frac=args.lot_frac,
        assignment=args.assignment,
        reservoir_scale=args.reservoir_scale,
        run_tag_velocity=args.run_tag_velocity,
        run_tag_positions=args.run_tag_positions,
        fetch_goes_if_missing=args.fetch_goes,
        goes_hours=args.goes_hours,
        goes_date=args.goes_date,
        goes_hour_utc=args.goes_hour_utc,
        damping=args.damping,
        decay=args.decay,
        nudge_interval=args.nudge_interval,
        nudge_strength=args.nudge_strength,
    )

    print(json.dumps(summary, indent=2))
    v_win = summary["velocity_wins"]
    metric = summary.get("comparison_metric", "lot_embedding_map_rmse")

    print(f"\n{'='*60}")
    print(f"  LOT-RMSE    velocity={summary['forecast_map_rmse_velocity']:.6f}  "
          f"positions={summary['forecast_map_rmse_positions']:.6f}  "
          f"winner={'VELOCITY' if summary['velocity_wins_map_rmse'] else 'POSITIONS'}")
    if "sinkhorn_w2_velocity" in summary:
        print(f"  Sinkhorn-W2 velocity={summary['sinkhorn_w2_velocity']:.6f}  "
              f"positions={summary['sinkhorn_w2_positions']:.6f}  "
              f"winner={'VELOCITY' if summary.get('velocity_wins_sinkhorn') else 'POSITIONS'}")
    print(f"  Primary metric: {metric} -> {'VELOCITY' if v_win else 'POSITIONS'} wins")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
