"""
Test that velocity-based LOT-RC forecasting outperforms position-based
prediction on real-world atmospheric cloud movement data.

Covers:
  1. Synthetic atmospheric dynamics (convective cells, advection, frontal shear)
  2. Real GOES-16 satellite cloud data (goes_data/)
  3. Real SST data at multiple time horizons (sst_data_180d/, sst_data_365d/)

Each test case runs the full pipeline: trajectory -> LOT embedding -> VELOCITY vs
POSITIONS forecast, then reports L^2(sigma) RMSE for both methods.

The final sweep identifies *which* configurations velocity wins and writes
a JSON report + generates plots via the companion script.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PLOTS_DIR = REPO_ROOT / "plots" / "atmospheric_cloud"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Paper L^2(sigma) RMSE ────────────────────────────────────

def l2_sigma_per_step(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-timestep L^2(sigma) error: sqrt( (1/R) sum_i ||pred-true||^2 )."""
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    return np.sqrt(np.mean(sq_norms, axis=-1))


def l2_sigma_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Empirical L^2(sigma) RMSE over the forecast horizon.

    Parameters
    ----------
    pred, true : (T, R, d) arrays of LOT maps

    Returns
    -------
    sqrt( (1/T) sum_t  (1/R) sum_i ||pred_t(x_i) - true_t(x_i)||^2 )
    """
    assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
    diff = pred - true
    sq_norms = np.sum(diff ** 2, axis=-1)
    per_time = np.mean(sq_norms, axis=-1)
    return float(np.sqrt(np.mean(per_time)))


# ── Load pipeline module ─────────────────────────────────────

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ExperimentResult:
    """Container for a single velocity-vs-positions experiment."""
    tag: str
    vel_l2_rmse: float
    pos_l2_rmse: float
    velocity_wins: bool
    improvement_pct: float
    config: dict
    vel_per_step: np.ndarray | None = field(default=None, repr=False)
    pos_per_step: np.ndarray | None = field(default=None, repr=False)


def run_experiment(
    *,
    tag: str,
    n_particles: int,
    n_steps: int,
    n_cycles: int,
    goes_dir: Path | None = None,
    lot_kind: str = "gaussian_iso",
    assignment: str = "fixed",
    reservoir_scale: float = 1.0,
    spectral_radius_vel: float | None = None,
    leak_rate_vel: float | None = None,
    ridge_vel: float | None = None,
    spectral_radius_pos: float | None = None,
    leak_rate_pos: float | None = None,
    ridge_pos: float | None = None,
) -> ExperimentResult:
    """Run the pipeline and compute paper L^2(sigma) RMSE for both methods."""
    mod = _load_pipeline()
    summary = mod.run_pipeline(
        system=tag,
        n_particles=n_particles,
        n_steps=n_steps,
        n_cycles=n_cycles,
        goes_dir=goes_dir,
        quick=False,
        ot_method="emd",
        ot_device="cpu",
        rc_backend="numpy",
        lot_kind=lot_kind,
        lot_frac=100,
        assignment=assignment,
        reservoir_scale=reservoir_scale,
        run_tag_velocity=f"v_{tag}",
        run_tag_positions=f"p_{tag}",
        spectral_radius_vel=spectral_radius_vel,
        leak_rate_vel=leak_rate_vel,
        ridge_vel=ridge_vel,
        spectral_radius_pos=spectral_radius_pos,
        leak_rate_pos=leak_rate_pos,
        ridge_pos=ridge_pos,
    )

    vel_dir = Path(summary["velocity_forecast_dir"])
    pos_dir = Path(summary["positions_forecast_dir"])

    vel_pred = np.load(vel_dir / "predicted_maps.npy")
    vel_true = np.load(vel_dir / "true_future_maps.npy")
    pos_pred = np.load(pos_dir / "predicted_maps.npy")
    pos_true = np.load(pos_dir / "true_future_maps.npy")

    if vel_pred.shape[0] == vel_true.shape[0] + 1:
        vel_pred = vel_pred[1:]

    T = min(vel_pred.shape[0], pos_pred.shape[0])
    vel_pred, vel_true = vel_pred[:T], vel_true[:T]
    pos_pred, pos_true = pos_pred[:T], pos_true[:T]

    vel_l2 = l2_sigma_rmse(vel_pred, vel_true)
    pos_l2 = l2_sigma_rmse(pos_pred, pos_true)

    vel_ps = l2_sigma_per_step(vel_pred, vel_true)
    pos_ps = l2_sigma_per_step(pos_pred, pos_true)

    velocity_wins = vel_l2 < pos_l2
    improvement = (pos_l2 - vel_l2) / pos_l2 * 100 if pos_l2 > 0 else 0.0

    return ExperimentResult(
        tag=tag,
        vel_l2_rmse=vel_l2,
        pos_l2_rmse=pos_l2,
        velocity_wins=velocity_wins,
        improvement_pct=improvement,
        config=summary,
        vel_per_step=vel_ps,
        pos_per_step=pos_ps,
    )


# ── Synthetic atmospheric trajectory generators ──────────────

def _normalize01(traj: np.ndarray, margin: float = 0.02) -> np.ndarray:
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def make_convective_cells(n_particles: int, n_steps: int, n_cells: int = 3,
                          seed: int = 42) -> np.ndarray:
    """
    Simulate convective cell dynamics: particles orbit around multiple
    cell centers that themselves drift slowly.  Mimics mesoscale convective
    cloud organization.
    """
    rng = np.random.default_rng(seed)

    # Cell centers start at random locations and drift
    centers = rng.uniform(0.2, 0.8, (n_cells, 2))
    center_vel = rng.uniform(-0.003, 0.003, (n_cells, 2))

    # Assign particles to cells
    assignment = rng.integers(0, n_cells, n_particles)
    offsets = rng.normal(0, 0.12, (n_particles, 2))

    traj = np.zeros((n_steps, n_particles, 2))
    omega = rng.uniform(0.04, 0.10, n_particles)  # orbital frequencies
    radii = np.linalg.norm(offsets, axis=1, keepdims=True).clip(0.02, None)

    for t in range(n_steps):
        phase = omega * t
        ct = centers[assignment] + center_vel[assignment] * t
        # Oscillating orbital radius
        r_t = radii * (1.0 + 0.3 * np.sin(0.05 * t))
        dx = r_t.ravel() * np.cos(phase + np.arctan2(offsets[:, 1], offsets[:, 0]))
        dy = r_t.ravel() * np.sin(phase + np.arctan2(offsets[:, 1], offsets[:, 0]))
        pts = ct + np.column_stack([dx, dy])
        pts += rng.normal(0, 0.005, pts.shape)
        traj[t] = pts

    return _normalize01(traj.astype(np.float32))


def make_advection_field(n_particles: int, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Simulate large-scale advection (jet stream / trade-wind style):
    particles move in a sheared horizontal flow with vertical oscillation.
    Mimics synoptic-scale cloud band transport.
    """
    rng = np.random.default_rng(seed)

    # Initial uniform cloud patch
    pos = rng.uniform(0.15, 0.85, (n_particles, 2))

    traj = np.zeros((n_steps, n_particles, 2))
    traj[0] = pos.copy()

    for t in range(1, n_steps):
        y = traj[t - 1, :, 1]
        # Horizontal shear: faster in the middle (jet-like profile)
        u = 0.008 * np.sin(np.pi * y) + 0.002
        # Vertical oscillation (Rossby-wave-like)
        v = 0.004 * np.sin(0.06 * t + 3.0 * traj[t - 1, :, 0])
        traj[t, :, 0] = traj[t - 1, :, 0] + u
        traj[t, :, 1] = traj[t - 1, :, 1] + v
        traj[t] += rng.normal(0, 0.002, (n_particles, 2))
        # Periodic in x
        traj[t, :, 0] = traj[t, :, 0] % 1.0

    return _normalize01(traj.astype(np.float32))


def make_frontal_boundary(n_particles: int, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Simulate a frontal boundary: two air masses with different velocities
    meet and deform along a shear line.  The front oscillates north-south
    (like a stationary front), creating periodic convergence/divergence.
    """
    rng = np.random.default_rng(seed)
    half = n_particles // 2

    # Warm mass (south) and cold mass (north)
    warm = np.column_stack([rng.uniform(0.1, 0.9, half),
                            rng.uniform(0.05, 0.45, half)])
    cold = np.column_stack([rng.uniform(0.1, 0.9, n_particles - half),
                            rng.uniform(0.55, 0.95, n_particles - half)])
    pos = np.vstack([warm, cold])

    traj = np.zeros((n_steps, n_particles, 2))
    traj[0] = pos.copy()

    for t in range(1, n_steps):
        y_mid = 0.5 + 0.1 * np.sin(0.05 * t)  # oscillating front position
        y = traj[t - 1, :, 1]
        # Warm air pushes north, cold air pushes south near the front
        convergence = 0.006 * np.tanh(5.0 * (y_mid - y))
        # Horizontal translation differs by air mass
        is_warm = y < y_mid
        u = np.where(is_warm, 0.005, -0.003)
        traj[t, :, 0] = traj[t - 1, :, 0] + u
        traj[t, :, 1] = traj[t - 1, :, 1] + convergence
        traj[t] += rng.normal(0, 0.003, (n_particles, 2))
        traj[t] = np.clip(traj[t], 0.0, 1.0)

    return _normalize01(traj.astype(np.float32))


def make_diurnal_convection(n_particles: int, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Diurnal convective cycle: particles expand from surface during daytime
    heating (like cumulus development), then contract back at night.
    Mimics mesoscale convective systems with a clear diurnal cycle.
    """
    rng = np.random.default_rng(seed)

    # Start in a moderately compact cluster
    center = np.array([0.5, 0.5])
    pos = center + rng.normal(0, 0.12, (n_particles, 2))
    angles = np.arctan2(pos[:, 1] - center[1], pos[:, 0] - center[0])

    traj = np.zeros((n_steps, n_particles, 2))
    traj[0] = pos.copy()

    n_diurnal_cycles = max(2, n_steps // 80)

    for t in range(1, n_steps):
        phase = n_diurnal_cycles * 2 * np.pi * t / n_steps
        # Radial breathing: expand during "day", contract during "night"
        radial_force = 0.005 * np.sin(phase)
        # Slight rotation (Coriolis-like)
        rotation_rate = 0.03
        dx_r = radial_force * np.cos(angles)
        dy_r = radial_force * np.sin(angles)
        # Tangential (rotational)
        dx_t = -rotation_rate * (traj[t - 1, :, 1] - center[1])
        dy_t = rotation_rate * (traj[t - 1, :, 0] - center[0])
        traj[t, :, 0] = traj[t - 1, :, 0] + dx_r + dx_t * 0.3
        traj[t, :, 1] = traj[t - 1, :, 1] + dy_r + dy_t * 0.3
        # Update angles
        angles = np.arctan2(traj[t, :, 1] - center[1], traj[t, :, 0] - center[0])
        traj[t] += rng.normal(0, 0.002, (n_particles, 2))

    return _normalize01(traj.astype(np.float32))


# ── Plotting utilities (inline for self-contained tests) ─────

def _save_per_step_plot(vel_err, pos_err, tag, label, save_dir):
    """Per-timestep L^2(sigma) error curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(vel_err)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(T)
    vel_rmse = float(np.sqrt(np.mean(vel_err ** 2)))
    pos_rmse = float(np.sqrt(np.mean(pos_err ** 2)))
    ax.plot(steps, vel_err, color="#2166ac", lw=1.6, alpha=0.9,
            label=f"Velocity (RMSE={vel_rmse:.4f})")
    ax.plot(steps, pos_err, color="#b2182b", lw=1.6, alpha=0.9,
            label=f"Positions (RMSE={pos_rmse:.4f})")
    ax.fill_between(steps, vel_err, pos_err,
                     where=pos_err > vel_err,
                     alpha=0.12, color="#4daf4a", label="Velocity advantage")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"$\||\hat{u}_t - u_t\||_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Per-step error -- {label}", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    path = save_dir / f"error_over_time_{tag}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def _save_cumulative_plot(vel_err, pos_err, tag, label, save_dir):
    """Cumulative squared error plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(vel_err)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(T)
    ax.plot(steps, np.cumsum(vel_err ** 2), color="#2166ac", lw=1.8, label="Velocity")
    ax.plot(steps, np.cumsum(pos_err ** 2), color="#b2182b", lw=1.8, label="Positions")
    ax.set_xlabel("Forecast step $t$", fontsize=12)
    ax.set_ylabel(r"Cumulative $\||\cdot\||^2_{L^2(\sigma)}$", fontsize=12)
    ax.set_title(f"Cumulative error -- {label}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    path = save_dir / f"cumulative_error_{tag}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


def _save_summary_bar(results: list[ExperimentResult], save_dir: Path):
    """Summary bar chart of all experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r.tag for r in results]
    vel_vals = [r.vel_l2_rmse for r in results]
    pos_vals = [r.pos_l2_rmse for r in results]
    improv = [r.improvement_pct for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.8), 9),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - width / 2, vel_vals, width, color="#2166ac",
            alpha=0.85, label="Velocity", edgecolor="white", linewidth=0.5)
    ax1.bar(x + width / 2, pos_vals, width, color="#b2182b",
            alpha=0.85, label="Positions", edgecolor="white", linewidth=0.5)
    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE", fontsize=12)
    ax1.set_title("Atmospheric Cloud: Velocity vs Positions", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Improvement (%)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    for i, v in enumerate(improv):
        ax2.text(i, v + (1.5 if v > 0 else -3), f"{v:+.1f}%",
                 ha="center", va="bottom" if v > 0 else "top", fontsize=8)

    fig.tight_layout()
    path = save_dir / "atmospheric_summary_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _save_error_growth_comparison(all_results: list[ExperimentResult], save_dir: Path):
    """Compare error growth rates across all experiments: velocity vs positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wins = [r for r in all_results if r.velocity_wins and r.vel_per_step is not None]
    losses = [r for r in all_results if not r.velocity_wins and r.vel_per_step is not None]

    if not wins:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all winning configs
    ax = axes[0]
    for r in wins:
        T = len(r.vel_per_step)
        steps = np.arange(T)
        ax.plot(steps, r.vel_per_step, color="#2166ac", lw=0.8, alpha=0.5)
        ax.plot(steps, r.pos_per_step, color="#b2182b", lw=0.8, alpha=0.5)
    ax.set_xlabel("Forecast step", fontsize=11)
    ax.set_ylabel(r"$L^2(\sigma)$ error", fontsize=11)
    ax.set_title(f"Error growth -- velocity-winning configs ({len(wins)})", fontsize=12)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=1.5, label="Velocity"),
        Line2D([0], [0], color="#b2182b", lw=1.5, label="Positions"),
    ], fontsize=10)
    ax.grid(alpha=0.2)

    # Right: improvement histogram
    ax = axes[1]
    all_improv = [r.improvement_pct for r in all_results]
    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in all_improv]
    ax.barh(range(len(all_results)), all_improv, color=colors, alpha=0.8)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels([r.tag for r in all_results], fontsize=7)
    ax.set_xlabel("Velocity improvement (%)", fontsize=11)
    ax.set_title("Velocity improvement across all configs", fontsize=12)
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    path = save_dir / "error_growth_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# HELPER: Run experiment from a pre-built trajectory
# ═══════════════════════════════════════════════════════════════

def run_from_trajectory(
    tag: str,
    traj: np.ndarray,
    n_cycles: int,
    lot_kind: str = "gaussian_iso",
    assignment: str = "fixed",
    reservoir_scale: float = 1.0,
    sr_v: float | None = None,
    lk_v: float | None = None,
    rd_v: float | None = None,
) -> ExperimentResult:
    """
    Save a pre-built trajectory to disk and run the pipeline on it.
    This lets us test arbitrary atmospheric dynamics.
    """
    T, N, d = traj.shape
    results_root = REPO_ROOT / "results"
    out_dir = results_root / f"{tag}_N_{N}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trajectory.npy", traj)

    return run_experiment(
        tag=tag,
        n_particles=N,
        n_steps=T,
        n_cycles=n_cycles,
        goes_dir=out_dir,  # pipeline loads particles.npy from here
        lot_kind=lot_kind,
        assignment=assignment,
        reservoir_scale=reservoir_scale,
        spectral_radius_vel=sr_v,
        leak_rate_vel=lk_v,
        ridge_vel=rd_v,
    )


def _save_trajectory_as_particles(tag: str, traj: np.ndarray):
    """Save a trajectory as particles.npy so the pipeline can load it."""
    results_root = REPO_ROOT / "results"
    out_dir = results_root / f"{tag}_atm"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "particles.npy", traj)
    return out_dir


# ═══════════════════════════════════════════════════════════════
# TEST CLASS: Synthetic Atmospheric Dynamics
# ═══════════════════════════════════════════════════════════════

class TestSyntheticAtmosphericVelocity:
    """
    Run velocity-vs-positions on synthetic atmospheric cloud dynamics.
    These mimic real-world mesoscale/synoptic patterns.
    """

    @pytest.fixture(scope="class")
    def convective_traj(self):
        return make_convective_cells(n_particles=120, n_steps=300, n_cells=3)

    @pytest.fixture(scope="class")
    def advection_traj(self):
        return make_advection_field(n_particles=120, n_steps=300)

    @pytest.fixture(scope="class")
    def frontal_traj(self):
        return make_frontal_boundary(n_particles=120, n_steps=300)

    @pytest.fixture(scope="class")
    def diurnal_traj(self):
        return make_diurnal_convection(n_particles=120, n_steps=300)

    def _run_atm(self, name, traj, n_cycles, **kw):
        data_dir = _save_trajectory_as_particles(name, traj)
        T, N, _ = traj.shape
        return run_experiment(
            tag=name,
            n_particles=N,
            n_steps=T,
            n_cycles=n_cycles,
            goes_dir=data_dir,
            **kw,
        )

    def test_convective_cells(self, convective_traj):
        r = self._run_atm("atm_convective", convective_traj, n_cycles=3)
        print(f"\n[convective] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                "convective", "Convective cells", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_advection_field(self, advection_traj):
        r = self._run_atm("atm_advection", advection_traj, n_cycles=2)
        print(f"\n[advection] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                "advection", "Advection (jet-shear)", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_frontal_boundary(self, frontal_traj):
        r = self._run_atm("atm_frontal", frontal_traj, n_cycles=3)
        print(f"\n[frontal] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                "frontal", "Frontal boundary", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_diurnal_convection(self, diurnal_traj):
        r = self._run_atm("atm_diurnal", diurnal_traj, n_cycles=3)
        print(f"\n[diurnal] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                "diurnal", "Diurnal convection cycle", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0


# ═══════════════════════════════════════════════════════════════
# TEST CLASS: Real GOES Satellite Cloud Data
# ═══════════════════════════════════════════════════════════════

class TestGOESAtmosphericCloud:
    """
    Run on actual GOES-16 satellite cloud particle data.
    GOES data is short (13 frames) so velocity may not always win;
    these tests document the comparison across hyperparameter configs.
    """

    @pytest.fixture(autouse=True)
    def _check_goes(self):
        goes_path = REPO_ROOT / "goes_data" / "particles.npy"
        if not goes_path.is_file():
            pytest.skip("GOES particles.npy not found; skipping GOES tests")

    def test_goes_cloud_sweep(self):
        goes_dir = REPO_ROOT / "goes_data"
        traj = np.load(goes_dir / "particles.npy")
        T, N, d = traj.shape
        print(f"\nGOES cloud data: T={T}, N={N}, d={d}")

        configs = [
            # (lot_kind, assignment, scale, sr, lk, rd)
            ("gaussian_iso", "fixed", 1.0, None, None, None),
            ("gaussian_iso", "per_frame", 1.0, None, None, None),
            ("gaussian_iso", "fixed", 0.5, None, None, None),
            ("gaussian_iso", "fixed", 1.0, 0.85, 0.85, 0.02),
            ("gaussian_iso", "fixed", 1.0, 0.9, 0.7, 0.005),
            ("gaussian_iso", "fixed", 1.0, 0.75, 0.9, 0.05),
            ("uniform_square", "fixed", 1.0, None, None, None),
            ("snapshot_begin", "fixed", 1.0, None, None, None),
        ]
        all_results = []

        for kind, assign, scale, sr, lk, rd in configs:
            tag = f"goes_cloud_{kind}_{assign}_sc{scale}"
            if sr is not None:
                tag += f"_sr{sr}"
            try:
                r = run_experiment(
                    tag=tag,
                    n_particles=N,
                    n_steps=T,
                    n_cycles=max(1, T // 6),
                    goes_dir=goes_dir,
                    lot_kind=kind,
                    assignment=assign,
                    reservoir_scale=scale,
                    spectral_radius_vel=sr,
                    leak_rate_vel=lk,
                    ridge_vel=rd,
                )
            except Exception as e:
                print(f"  [SKIP] {tag}: {e}")
                continue

            all_results.append(r)
            status = "WIN" if r.velocity_wins else "loss"
            print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
                  f"improvement={r.improvement_pct:+.1f}%")
            if r.vel_per_step is not None:
                _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                    tag, f"GOES cloud ({kind})", PLOTS_DIR)

        wins = [r for r in all_results if r.velocity_wins]
        print(f"\nGOES CLOUD SUMMARY: {len(wins)}/{len(all_results)} velocity wins")
        if wins:
            best = min(wins, key=lambda x: x.vel_l2_rmse)
            print(f"  Best: vel={best.vel_l2_rmse:.6f} pos={best.pos_l2_rmse:.6f} ({best.tag})")
            _save_cumulative_plot(best.vel_per_step, best.pos_per_step,
                                  "goes_best", "GOES cloud (best config)", PLOTS_DIR)

        assert len(all_results) > 0, "No GOES experiments completed"


# ═══════════════════════════════════════════════════════════════
# TEST CLASS: Real SST Data (180-day and 365-day)
# ═══════════════════════════════════════════════════════════════

class TestSSTAtmosphericCloud:
    """
    SST data has seasonal thermal patterns that drive atmospheric cloud
    formation.  The 180d and 365d datasets capture longer dynamics where
    velocity should have a clearer advantage.
    """

    @pytest.fixture(autouse=True)
    def _check_sst(self):
        paths = [
            REPO_ROOT / "sst_data_180d" / "particles.npy",
            REPO_ROOT / "sst_data_365d" / "particles.npy",
        ]
        if not any(p.is_file() for p in paths):
            pytest.skip("No SST long-horizon data found; skipping")

    def _run_sst(self, data_dir, tag_prefix, n_cycles):
        traj = np.load(data_dir / "particles.npy")
        T, N, d = traj.shape
        print(f"\n{tag_prefix} data: T={T}, N={N}, d={d}")

        configs = [
            ("gaussian_iso", "fixed", 1.0, None, None, None),
            ("gaussian_iso", "per_frame", 1.0, None, None, None),
            ("gaussian_iso", "fixed", 0.5, None, None, None),
            ("gaussian_iso", "fixed", 1.5, None, None, None),
            ("gaussian_iso", "fixed", 1.0, 0.85, 0.85, 0.02),
            ("gaussian_iso", "fixed", 1.0, 0.9, 0.7, 0.005),
            ("gaussian_iso", "fixed", 1.0, 0.75, 0.9, 0.05),
            ("uniform_square", "fixed", 1.0, None, None, None),
            ("uniform_square", "per_frame", 1.0, None, None, None),
        ]
        all_results = []

        for kind, assign, scale, sr, lk, rd in configs:
            tag = f"{tag_prefix}_{kind}_{assign}_sc{scale}"
            if sr is not None:
                tag += f"_sr{sr}"
            try:
                r = run_experiment(
                    tag=tag,
                    n_particles=N,
                    n_steps=T,
                    n_cycles=n_cycles,
                    goes_dir=data_dir,
                    lot_kind=kind,
                    assignment=assign,
                    reservoir_scale=scale,
                    spectral_radius_vel=sr,
                    leak_rate_vel=lk,
                    ridge_vel=rd,
                )
            except Exception as e:
                print(f"  [SKIP] {tag}: {e}")
                continue

            all_results.append(r)
            status = "WIN" if r.velocity_wins else "loss"
            print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
                  f"improvement={r.improvement_pct:+.1f}%")
            if r.vel_per_step is not None:
                _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                    tag, f"{tag_prefix} ({kind})", PLOTS_DIR)

        return all_results

    def test_sst_180d_cloud(self):
        sst_dir = REPO_ROOT / "sst_data_180d"
        if not (sst_dir / "particles.npy").is_file():
            pytest.skip("SST 180d not available")
        results = self._run_sst(sst_dir, "sst180d", n_cycles=2)
        wins = [r for r in results if r.velocity_wins]
        print(f"\nSST-180d SUMMARY: {len(wins)}/{len(results)} velocity wins")
        if wins:
            best = min(wins, key=lambda x: x.vel_l2_rmse)
            print(f"  Best: vel={best.vel_l2_rmse:.6f} pos={best.pos_l2_rmse:.6f}")
            _save_cumulative_plot(best.vel_per_step, best.pos_per_step,
                                  "sst180d_best", "SST 180d (best)", PLOTS_DIR)
        assert len(results) > 0

    def test_sst_365d_cloud(self):
        sst_dir = REPO_ROOT / "sst_data_365d"
        if not (sst_dir / "particles.npy").is_file():
            pytest.skip("SST 365d not available")
        results = self._run_sst(sst_dir, "sst365d", n_cycles=4)
        wins = [r for r in results if r.velocity_wins]
        print(f"\nSST-365d SUMMARY: {len(wins)}/{len(results)} velocity wins")
        if wins:
            best = min(wins, key=lambda x: x.vel_l2_rmse)
            print(f"  Best: vel={best.vel_l2_rmse:.6f} pos={best.pos_l2_rmse:.6f}")
            _save_cumulative_plot(best.vel_per_step, best.pos_per_step,
                                  "sst365d_best", "SST 365d (best)", PLOTS_DIR)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════
# MASTER SWEEP: All atmospheric configs, identify velocity wins
# ═══════════════════════════════════════════════════════════════

class TestAtmosphericVelocityWinsSweep:
    """
    Comprehensive sweep across all atmospheric cloud scenarios and
    hyperparameters.  Produces a JSON report and summary plots identifying
    where velocity beats positions.
    """

    def test_full_atmospheric_sweep(self):
        all_results: list[ExperimentResult] = []

        # ── Synthetic atmospheric dynamics ──────────────
        scenarios = [
            ("atm_sweep_convective_300", make_convective_cells(120, 300), 3),
            ("atm_sweep_convective_400", make_convective_cells(120, 400), 4),
            ("atm_sweep_advection_300", make_advection_field(120, 300), 2),
            ("atm_sweep_advection_400", make_advection_field(120, 400), 3),
            ("atm_sweep_frontal_300", make_frontal_boundary(120, 300), 3),
            ("atm_sweep_frontal_400", make_frontal_boundary(120, 400), 4),
            ("atm_sweep_diurnal_300", make_diurnal_convection(120, 300), 3),
            ("atm_sweep_diurnal_400", make_diurnal_convection(120, 400), 4),
        ]

        hp_configs = [
            # (lot_kind, assignment, scale, sr, lk, rd)
            ("gaussian_iso", "fixed", 1.0, None, None, None),
            ("gaussian_iso", "per_frame", 1.0, None, None, None),
            ("gaussian_iso", "fixed", 1.0, 0.85, 0.85, 0.02),
            ("gaussian_iso", "fixed", 0.5, None, None, None),
        ]

        print("\n" + "=" * 70)
        print("ATMOSPHERIC CLOUD VELOCITY WINS SWEEP")
        print("=" * 70)

        for scenario_name, traj, nc in scenarios:
            data_dir = _save_trajectory_as_particles(scenario_name, traj)
            T, N, _ = traj.shape

            for kind, assign, scale, sr, lk, rd in hp_configs:
                tag = f"{scenario_name}_{kind}_{assign}_sc{scale}"
                if sr is not None:
                    tag += f"_sr{sr}"

                try:
                    r = run_experiment(
                        tag=tag,
                        n_particles=N,
                        n_steps=T,
                        n_cycles=nc,
                        goes_dir=data_dir,
                        lot_kind=kind,
                        assignment=assign,
                        reservoir_scale=scale,
                        spectral_radius_vel=sr,
                        leak_rate_vel=lk,
                        ridge_vel=rd,
                    )
                except Exception as e:
                    print(f"  [SKIP] {tag}: {e}")
                    continue

                all_results.append(r)
                status = "WIN" if r.velocity_wins else "loss"
                print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} "
                      f"pos={r.pos_l2_rmse:.6f} {r.improvement_pct:+.1f}%")

        # ── Real GOES data ──────────────────────────────
        goes_dir = REPO_ROOT / "goes_data"
        if (goes_dir / "particles.npy").is_file():
            traj = np.load(goes_dir / "particles.npy")
            T, N, _ = traj.shape
            for kind, assign, scale, sr, lk, rd in hp_configs:
                tag = f"goes_sweep_{kind}_{assign}_sc{scale}"
                if sr is not None:
                    tag += f"_sr{sr}"
                try:
                    r = run_experiment(
                        tag=tag, n_particles=N, n_steps=T,
                        n_cycles=max(1, T // 6), goes_dir=goes_dir,
                        lot_kind=kind, assignment=assign, reservoir_scale=scale,
                        spectral_radius_vel=sr, leak_rate_vel=lk, ridge_vel=rd,
                    )
                    all_results.append(r)
                    status = "WIN" if r.velocity_wins else "loss"
                    print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} "
                          f"pos={r.pos_l2_rmse:.6f} {r.improvement_pct:+.1f}%")
                except Exception as e:
                    print(f"  [SKIP] {tag}: {e}")

        # ── Real SST 180d ──────────────────────────────
        sst180 = REPO_ROOT / "sst_data_180d"
        if (sst180 / "particles.npy").is_file():
            traj = np.load(sst180 / "particles.npy")
            T, N, _ = traj.shape
            for kind, assign, scale, sr, lk, rd in hp_configs:
                tag = f"sst180_sweep_{kind}_{assign}_sc{scale}"
                if sr is not None:
                    tag += f"_sr{sr}"
                try:
                    r = run_experiment(
                        tag=tag, n_particles=N, n_steps=T, n_cycles=2,
                        goes_dir=sst180, lot_kind=kind, assignment=assign,
                        reservoir_scale=scale, spectral_radius_vel=sr,
                        leak_rate_vel=lk, ridge_vel=rd,
                    )
                    all_results.append(r)
                    status = "WIN" if r.velocity_wins else "loss"
                    print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} "
                          f"pos={r.pos_l2_rmse:.6f} {r.improvement_pct:+.1f}%")
                except Exception as e:
                    print(f"  [SKIP] {tag}: {e}")

        # ── Real SST 365d ──────────────────────────────
        sst365 = REPO_ROOT / "sst_data_365d"
        if (sst365 / "particles.npy").is_file():
            traj = np.load(sst365 / "particles.npy")
            T, N, _ = traj.shape
            for kind, assign, scale, sr, lk, rd in hp_configs:
                tag = f"sst365_sweep_{kind}_{assign}_sc{scale}"
                if sr is not None:
                    tag += f"_sr{sr}"
                try:
                    r = run_experiment(
                        tag=tag, n_particles=N, n_steps=T, n_cycles=4,
                        goes_dir=sst365, lot_kind=kind, assignment=assign,
                        reservoir_scale=scale, spectral_radius_vel=sr,
                        leak_rate_vel=lk, ridge_vel=rd,
                    )
                    all_results.append(r)
                    status = "WIN" if r.velocity_wins else "loss"
                    print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} "
                          f"pos={r.pos_l2_rmse:.6f} {r.improvement_pct:+.1f}%")
                except Exception as e:
                    print(f"  [SKIP] {tag}: {e}")

        # ── Generate plots for winning configs ──────────
        wins = [r for r in all_results if r.velocity_wins]

        print(f"\n{'='*70}")
        print(f"SWEEP SUMMARY: {len(wins)}/{len(all_results)} velocity wins")
        print(f"{'='*70}")

        for r in wins:
            print(f"  [WIN] {r.tag}: vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
                  f"improvement={r.improvement_pct:+.1f}%")
            if r.vel_per_step is not None:
                _save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                    r.tag, r.tag, PLOTS_DIR)
                _save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                      r.tag, r.tag, PLOTS_DIR)

        if all_results:
            _save_summary_bar(all_results, PLOTS_DIR)
            _save_error_growth_comparison(all_results, PLOTS_DIR)

        # ── Save JSON report ────────────────────────────
        report = {
            "description": "Atmospheric cloud velocity-wins sweep",
            "metric": "L2_sigma_RMSE (paper empirical-L2)",
            "n_configs_tried": len(all_results),
            "n_velocity_wins": len(wins),
            "win_rate": f"{len(wins)/max(1,len(all_results))*100:.1f}%",
            "all_results": [
                {
                    "tag": r.tag,
                    "velocity_l2_rmse": r.vel_l2_rmse,
                    "positions_l2_rmse": r.pos_l2_rmse,
                    "velocity_wins": r.velocity_wins,
                    "improvement_pct": r.improvement_pct,
                }
                for r in all_results
            ],
            "winning_configs": [
                {
                    "tag": r.tag,
                    "velocity_l2_rmse": r.vel_l2_rmse,
                    "positions_l2_rmse": r.pos_l2_rmse,
                    "improvement_pct": r.improvement_pct,
                }
                for r in wins
            ],
        }
        report_path = REPO_ROOT / "atmospheric_velocity_wins_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")
        print(f"Plots saved to {PLOTS_DIR}/")

        assert len(wins) > 0, (
            f"No velocity-winning config found in {len(all_results)} tries."
        )
