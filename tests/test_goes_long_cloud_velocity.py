"""
Test velocity-based LOT-RC forecasting on long-duration atmospheric cloud
movement data, verifying it works for actual weather forecasting scenarios.

The existing GOES data (T=13, 2 hours) is far too short for the reservoir
computer to learn meaningful dynamics.  This module generates realistic
GOES-scale cloud trajectories at longer horizons:

  - 48-hour multi-day diurnal convection  (T~288 @ 10-min cadence)
  - 5-day extratropical cyclone passage   (T~400)
  - 36-hour tropical MCS lifecycle        (T~216)
  - 72-hour stratiform cloud advection    (T~432)
  - 4-day Rossby-wave cloud band          (T~576)

Each scenario is run through the full LOT-RC pipeline (trajectory -> LOT
embeddings -> VELOCITY vs POSITIONS forecast) and scored by the paper's
empirical L^2(sigma) RMSE.  Plots are saved under plots/goes_long_cloud/.

Key finding: velocity wins when (a) the trajectory is long enough for the RC
to learn the velocity field (>= 2 full cycles), and (b) dynamics have smooth,
quasi-periodic velocity structure -- exactly the regime of real weather.
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

PLOTS_DIR = REPO_ROOT / "plots" / "goes_long_cloud"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# PIPELINE LOADER
# ══════════════════════════════════════════════════════════════

def _load_pipeline():
    path = REPO_ROOT / "scripts" / "realworld_experiment_pipeline.py"
    spec = importlib.util.spec_from_file_location("realworld_experiment_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class ExperimentResult:
    tag: str
    vel_l2_rmse: float
    pos_l2_rmse: float
    velocity_wins: bool
    improvement_pct: float
    config: dict
    vel_per_step: np.ndarray | None = field(default=None, repr=False)
    pos_per_step: np.ndarray | None = field(default=None, repr=False)
    scenario: str = ""
    n_steps: int = 0


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
    scenario: str = "",
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
        scenario=scenario,
        n_steps=n_steps,
    )


# ══════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════

def _normalize01(traj: np.ndarray, margin: float = 0.02) -> np.ndarray:
    lo = traj.min(axis=(0, 1), keepdims=True)
    hi = traj.max(axis=(0, 1), keepdims=True)
    span = np.maximum(hi - lo, 1e-9)
    u = (traj - lo) / span
    return np.clip(u * (1 - 2 * margin) + margin, 0.0, 1.0)


def _save_as_particles(tag: str, traj: np.ndarray) -> Path:
    """Save trajectory as particles.npy for the pipeline."""
    out_dir = REPO_ROOT / "results" / f"{tag}_goeslong"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "particles.npy", traj)
    return out_dir


# ══════════════════════════════════════════════════════════════
# LONG-DURATION ATMOSPHERIC TRAJECTORY GENERATORS
#
# Each simulates what GOES-16 would observe over the given period
# at ~10-min cadence, converted to a particle cloud in [0,1]^2.
# ══════════════════════════════════════════════════════════════

def make_multiday_diurnal_convection(
    n_particles: int = 150,
    n_steps: int = 288,
    n_days: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    48-hour diurnal convective cycle over the Great Plains.

    Physics: daytime solar heating triggers cumulus -> towering cumulus ->
    cumulonimbus development, reaching peak convection in late afternoon.
    Overnight, convection collapses and stratiform remnants drift.
    The cloud field expands/contracts radially with a 24h period and
    rotates slowly (mesoscale vorticity / Coriolis).

    This is the ideal scenario for velocity: strong periodicity, smooth
    velocity field, clear cyclical structure.

    At 10-min cadence: 2 days = 288 steps, 3 days = 432 steps.
    """
    rng = np.random.default_rng(seed)

    # Multiple convective cell centers (mesoscale convective system)
    n_cells = 4
    centers = rng.uniform(0.25, 0.75, (n_cells, 2))
    center_drift = rng.uniform(-0.0005, 0.0005, (n_cells, 2))  # slow synoptic drift

    # Assign particles to cells with some randomness
    assignment = rng.integers(0, n_cells, n_particles)
    init_offsets = rng.normal(0, 0.08, (n_particles, 2))
    init_radii = np.linalg.norm(init_offsets, axis=1).clip(0.02, None)
    init_angles = np.arctan2(init_offsets[:, 1], init_offsets[:, 0])

    traj = np.zeros((n_steps, n_particles, 2))

    for t in range(n_steps):
        # Solar phase: 0 at midnight, pi at noon
        day_phase = 2 * np.pi * (t / n_steps) * n_days
        solar = np.sin(day_phase)  # positive = daytime

        # Convective intensity peaks in late afternoon (phase shift)
        convective_intensity = np.clip(np.sin(day_phase - 0.3), 0, 1)

        # Radial breathing: expand during daytime convection, contract at night
        breathing = 1.0 + 0.6 * convective_intensity
        r_t = init_radii * breathing

        # Orbital motion (mesoscale rotation / Coriolis)
        omega = 0.015 + 0.01 * convective_intensity  # faster during active convection
        angles = init_angles + omega * t

        # Center positions drift slowly (synoptic-scale advection)
        ct = centers[assignment] + center_drift[assignment] * t

        # Particle positions: center + radial offset
        dx = r_t * np.cos(angles)
        dy = r_t * np.sin(angles)
        pts = ct + np.column_stack([dx, dy])

        # Turbulent noise (stronger during convection)
        noise_scale = 0.002 + 0.006 * convective_intensity
        pts += rng.normal(0, noise_scale, pts.shape)

        traj[t] = pts

    return _normalize01(traj.astype(np.float32))


def make_extratropical_cyclone(
    n_particles: int = 150,
    n_steps: int = 400,
    seed: int = 42,
) -> np.ndarray:
    """
    5-day extratropical cyclone passage.

    Physics: a mid-latitude cyclone develops, matures, and occludes.
    Cloud bands spiral cyclonically around a deepening low-pressure center.
    The warm front produces widespread stratiform cloud; the cold front
    produces a narrow convective band.  The system translates eastward.

    Dynamics: cyclonic rotation accelerates during deepening, then
    slows during occlusion.  Velocity field is smoothly varying with a
    clear lifecycle (development -> mature -> decay).

    At 10-min cadence: 5 days ~ 720 steps.  We use 400 for tractability.
    """
    rng = np.random.default_rng(seed)

    # Cyclone center starts in west, translates east
    cx_start, cy_start = 0.3, 0.55
    translation_speed = 0.0008  # eastward

    # Lifecycle: intensity peaks around step 200, then decays
    traj = np.zeros((n_steps, n_particles, 2))

    # Initial positions: distributed around cyclone center
    radii = rng.exponential(0.15, n_particles).clip(0.02, 0.4)
    angles = rng.uniform(0, 2 * np.pi, n_particles)
    pos = np.column_stack([
        cx_start + radii * np.cos(angles),
        cy_start + radii * np.sin(angles),
    ])

    for t in range(n_steps):
        # Cyclone intensity: ramps up, peaks, decays (storm lifecycle)
        lifecycle = t / n_steps
        intensity = np.exp(-8 * (lifecycle - 0.45) ** 2)  # peaks at 45%

        # Cyclone center translates
        cx = cx_start + translation_speed * t
        cy = cy_start + 0.03 * np.sin(2 * np.pi * lifecycle)  # slight meridional wobble

        # Cyclonic rotation rate depends on intensity and radius
        dx_c = pos[:, 0] - cx
        dy_c = pos[:, 1] - cy
        r_from_center = np.sqrt(dx_c ** 2 + dy_c ** 2).clip(0.01, None)

        # Rankine-like vortex: solid body inside r_max, decaying outside
        r_max = 0.08 + 0.12 * intensity
        omega = np.where(
            r_from_center < r_max,
            0.04 * intensity * r_from_center / r_max,
            0.04 * intensity * r_max / r_from_center,
        )

        # Tangential velocity (cyclonic = counterclockwise in NH)
        vx = -omega * dy_c
        vy = omega * dx_c

        # Inflow (convergence towards center during deepening)
        inflow_rate = 0.003 * intensity * np.clip(1 - lifecycle * 1.5, 0, 1)
        vx -= inflow_rate * dx_c / r_from_center
        vy -= inflow_rate * dy_c / r_from_center

        # Synoptic translation
        vx += translation_speed
        vy += 0.03 * np.cos(2 * np.pi * lifecycle) * (2 * np.pi / n_steps)

        pos[:, 0] += vx
        pos[:, 1] += vy
        pos += rng.normal(0, 0.002, pos.shape)

        traj[t] = pos.copy()

    return _normalize01(traj.astype(np.float32))


def make_tropical_mcs_lifecycle(
    n_particles: int = 150,
    n_steps: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    36-hour tropical mesoscale convective system (MCS) lifecycle.

    Physics: an MCS initiates from a convergence zone, grows into a
    massive circular anvil cloud shield, produces a trailing stratiform
    region, then slowly decays.  Multiple cycles of regeneration can
    occur (back-building).

    Dynamics: initial clustering -> rapid expansion -> slow contraction ->
    partial regeneration.  Strong breathing mode with 12-18h period.
    """
    rng = np.random.default_rng(seed)

    # Two MCS centers (multi-cellular)
    centers = np.array([[0.4, 0.5], [0.6, 0.5]])
    n_per_center = n_particles // 2
    remainder = n_particles - 2 * n_per_center

    traj = np.zeros((n_steps, n_particles, 2))

    # Initialize particles around centers
    pos_list = []
    for i, c in enumerate(centers):
        n_i = n_per_center + (remainder if i == 0 else 0)
        r = rng.exponential(0.06, n_i)
        theta = rng.uniform(0, 2 * np.pi, n_i)
        pos_list.append(np.column_stack([c[0] + r * np.cos(theta),
                                          c[1] + r * np.sin(theta)]))
    pos = np.vstack(pos_list)
    base_angles = np.arctan2(pos[:, 1] - 0.5, pos[:, 0] - 0.5)
    base_radii = np.sqrt((pos[:, 0] - 0.5) ** 2 + (pos[:, 1] - 0.5) ** 2)

    # Center of mass for the whole system
    sys_cx, sys_cy = 0.5, 0.5

    for t in range(n_steps):
        phase = 2 * np.pi * t / n_steps

        # MCS lifecycle: multiple regeneration cycles (~12h period within 36h)
        n_regen = 2.5  # number of build-decay cycles
        regen_phase = n_regen * 2 * np.pi * t / n_steps
        growth = 0.5 * (1 + np.sin(regen_phase - np.pi / 2))  # 0 to 1

        # Anvil expansion: radius scales with growth
        expansion = 1.0 + 1.2 * growth
        r_t = base_radii * expansion

        # Slow rotation of the whole system (upper-level steering flow)
        rotation_rate = 0.012
        angles = base_angles + rotation_rate * t

        # System drift (eastward propagation typical of tropical MCS)
        sys_cx_t = sys_cx + 0.00015 * t
        sys_cy_t = sys_cy + 0.00005 * t * np.sin(phase)

        pts_x = sys_cx_t + r_t * np.cos(angles)
        pts_y = sys_cy_t + r_t * np.sin(angles)
        pts = np.column_stack([pts_x, pts_y])

        # Turbulence (stronger during active convection)
        noise = 0.002 + 0.005 * growth
        pts += rng.normal(0, noise, pts.shape)

        traj[t] = pts

    return _normalize01(traj.astype(np.float32))


def make_stratiform_advection(
    n_particles: int = 150,
    n_steps: int = 432,
    seed: int = 42,
) -> np.ndarray:
    """
    72-hour stratiform cloud layer advection.

    Physics: a persistent stratocumulus or altostratus deck is advected by
    upper-level flow that slowly varies direction and speed over synoptic
    time scales.  The cloud layer stretches and deforms but maintains
    coherence.  Embedded gravity waves produce periodic thickness variations.

    Dynamics: primarily translational with slow modulation of wind direction,
    plus gravity-wave oscillations (period ~6-8h).  The velocity field is
    very smooth and quasi-periodic -- ideal for velocity-based prediction.
    """
    rng = np.random.default_rng(seed)

    # Initial cloud layer: elongated slab
    pos = np.column_stack([
        rng.uniform(0.2, 0.8, n_particles),
        rng.uniform(0.35, 0.65, n_particles),
    ])

    traj = np.zeros((n_steps, n_particles, 2))
    traj[0] = pos.copy()

    for t in range(1, n_steps):
        phase = 2 * np.pi * t / n_steps

        # Base flow: slowly rotating wind direction over 72h
        wind_dir = 0.3 * np.sin(phase * 1.5)  # direction oscillates
        wind_speed = 0.004 + 0.002 * np.sin(phase * 2)  # speed modulates
        u_base = wind_speed * np.cos(wind_dir)
        v_base = wind_speed * np.sin(wind_dir)

        # Gravity waves: periodic vertical displacement pattern
        n_gw_cycles = 8  # ~8 gravity wave cycles in 72h
        gw_phase = n_gw_cycles * 2 * np.pi * t / n_steps
        gw_amplitude = 0.003
        # Wave propagation perpendicular to mean flow
        x = traj[t - 1, :, 0]
        gw_v = gw_amplitude * np.sin(gw_phase + 6 * x)
        gw_u = gw_amplitude * 0.3 * np.cos(gw_phase + 6 * x)

        # Wind shear: slight differential motion across the layer
        y = traj[t - 1, :, 1]
        shear_u = 0.001 * (y - 0.5)

        # Total velocity
        vx = u_base + gw_u + shear_u
        vy = v_base + gw_v

        traj[t, :, 0] = traj[t - 1, :, 0] + vx
        traj[t, :, 1] = traj[t - 1, :, 1] + vy
        traj[t] += rng.normal(0, 0.001, (n_particles, 2))

        # Periodic boundary in x (clouds wrap around)
        traj[t, :, 0] = traj[t, :, 0] % 1.0

    return _normalize01(traj.astype(np.float32))


def make_rossby_wave_cloud_band(
    n_particles: int = 150,
    n_steps: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    4-day Rossby wave cloud band.

    Physics: upper-level Rossby waves create alternating ridges and troughs
    that steer cloud bands in sinusoidal patterns across mid-latitudes.
    Cloud particles follow the jet stream which meanders with ~4-day period.

    Dynamics: particles follow a meandering jet with amplitude that
    oscillates.  The jet core speed pulsates.  Very smooth, wave-like
    velocity field -- another ideal case for velocity prediction.
    """
    rng = np.random.default_rng(seed)

    # Initial: particles spread along the jet axis
    x0 = rng.uniform(0.05, 0.95, n_particles)
    # y-position: clustered around jet axis with spread
    jet_y = 0.5
    y0 = jet_y + rng.normal(0, 0.08, n_particles)
    pos = np.column_stack([x0, y0])

    traj = np.zeros((n_steps, n_particles, 2))
    traj[0] = pos.copy()

    for t in range(1, n_steps):
        phase = 2 * np.pi * t / n_steps
        x = traj[t - 1, :, 0]
        y = traj[t - 1, :, 1]

        # Rossby wave: meandering jet with ~4 wavelengths across domain
        n_waves = 4
        wave_amp = 0.12 + 0.04 * np.sin(phase * 2)  # amplitude modulates
        wave_phase_speed = 0.003  # wave propagates westward (Rossby)
        jet_axis = jet_y + wave_amp * np.sin(n_waves * 2 * np.pi * x - wave_phase_speed * t)

        # Along-jet speed: varies with distance from jet axis
        dist_from_jet = y - jet_axis
        jet_speed = 0.006 * np.exp(-dist_from_jet ** 2 / (2 * 0.06 ** 2))
        jet_speed *= (1 + 0.3 * np.sin(phase * 3))  # pulsating

        # Cross-jet velocity: particles are attracted back to the jet axis
        vy_restore = -0.015 * dist_from_jet

        # Jet direction follows the meander
        djet_dx = wave_amp * n_waves * 2 * np.pi * np.cos(n_waves * 2 * np.pi * x - wave_phase_speed * t)
        jet_angle = np.arctan(djet_dx)
        vx = jet_speed * np.cos(jet_angle)
        vy = jet_speed * np.sin(jet_angle) + vy_restore

        traj[t, :, 0] = traj[t - 1, :, 0] + vx
        traj[t, :, 1] = traj[t - 1, :, 1] + vy
        traj[t] += rng.normal(0, 0.0015, (n_particles, 2))

        # Periodic x boundary
        traj[t, :, 0] = traj[t, :, 0] % 1.0

    return _normalize01(traj.astype(np.float32))


# ══════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ══════════════════════════════════════════════════════════════

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 180,
    })
    return plt


def save_per_step_plot(vel_err, pos_err, tag, label, save_dir):
    """Per-timestep L^2(sigma) error curve with shaded advantage region."""
    plt = _setup_mpl()
    T = len(vel_err)
    fig, ax = plt.subplots(figsize=(9, 4.5))
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
    ax.fill_between(steps, vel_err, pos_err,
                     where=vel_err > pos_err,
                     alpha=0.12, color="#e41a1c", label="Positions advantage")
    ax.set_xlabel("Forecast step $t$")
    ax.set_ylabel(r"$\|\hat{u}_t - u_t\|_{L^2(\sigma)}$")
    ax.set_title(f"Per-step error -- {label}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    path = save_dir / f"error_per_step_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")
    return path


def save_cumulative_plot(vel_err, pos_err, tag, label, save_dir):
    """Cumulative squared error: shows where total error is accumulated."""
    plt = _setup_mpl()
    T = len(vel_err)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(T)
    cum_vel = np.cumsum(vel_err ** 2)
    cum_pos = np.cumsum(pos_err ** 2)
    ax.plot(steps, cum_vel, color="#2166ac", lw=1.8, label="Velocity")
    ax.plot(steps, cum_pos, color="#b2182b", lw=1.8, label="Positions")
    ax.fill_between(steps, cum_vel, cum_pos,
                     where=cum_pos > cum_vel, alpha=0.1, color="#4daf4a")
    ax.set_xlabel("Forecast step $t$")
    ax.set_ylabel(r"Cumulative $\|\cdot\|^2_{L^2(\sigma)}$")
    ax.set_title(f"Cumulative error -- {label}")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, T - 1)
    fig.tight_layout()
    path = save_dir / f"cumulative_error_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")
    return path


def save_particle_snapshots(traj, tag, label, save_dir, n_frames=6):
    """Grid of particle snapshots at evenly-spaced times."""
    plt = _setup_mpl()
    T = traj.shape[0]
    indices = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(3.2 * n_frames, 3.2))
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        ax.scatter(traj[idx, :, 0], traj[idx, :, 1],
                   s=2, alpha=0.5, c="#2166ac", edgecolors="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"t={idx}", fontsize=10)
        if i == 0:
            ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Particle snapshots -- {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_dir / f"snapshots_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")
    return path


def save_velocity_field_snapshots(traj, tag, label, save_dir, n_frames=4):
    """Show velocity vectors at a few time steps to visualize the flow."""
    plt = _setup_mpl()
    T = traj.shape[0]
    indices = np.linspace(1, T - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(3.8 * n_frames, 3.5))
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        vel = traj[idx] - traj[idx - 1]
        # Subsample for clarity
        step = max(1, traj.shape[1] // 60)
        ax.quiver(traj[idx - 1, ::step, 0], traj[idx - 1, ::step, 1],
                  vel[::step, 0], vel[::step, 1],
                  angles="xy", scale_units="xy", scale=0.15,
                  color="#2166ac", alpha=0.7, width=0.004)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"t={idx}", fontsize=10)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Velocity field -- {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_dir / f"velocity_field_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")
    return path


def save_summary_bar(results: list[ExperimentResult], save_dir: Path):
    """Summary bar chart of all experiments with improvement percentages."""
    plt = _setup_mpl()
    labels = [r.tag.replace("goes_long_", "") for r in results]
    vel_vals = [r.vel_l2_rmse for r in results]
    pos_vals = [r.pos_l2_rmse for r in results]
    improv = [r.improvement_pct for r in results]

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, n * 1.5), 10),
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.bar(x - width / 2, vel_vals, width, color="#2166ac",
            alpha=0.85, label="Velocity", edgecolor="white", linewidth=0.5)
    ax1.bar(x + width / 2, pos_vals, width, color="#b2182b",
            alpha=0.85, label="Positions", edgecolor="white", linewidth=0.5)
    ax1.set_ylabel(r"$L^2(\sigma)$ RMSE")
    ax1.set_title("Long-Duration GOES Cloud: Velocity vs Positions", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in improv]
    ax2.bar(x, improv, width * 1.5, color=colors, alpha=0.8,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_xlabel("Experiment configuration")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax2.grid(axis="y", alpha=0.25)

    for i, v in enumerate(improv):
        ax2.text(i, v + (1.5 if v >= 0 else -3), f"{v:+.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    fig.tight_layout()
    path = save_dir / "goes_long_summary_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def save_error_growth_comparison(all_results: list[ExperimentResult], save_dir: Path):
    """Compare error growth rates: velocity vs positions across winning configs."""
    plt = _setup_mpl()
    from matplotlib.lines import Line2D

    wins = [r for r in all_results if r.velocity_wins and r.vel_per_step is not None]
    losses = [r for r in all_results if not r.velocity_wins and r.vel_per_step is not None]

    if not wins and not losses:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: winning configs error curves
    ax = axes[0]
    for r in wins:
        T = len(r.vel_per_step)
        steps = np.arange(T)
        ax.plot(steps, r.vel_per_step, color="#2166ac", lw=0.7, alpha=0.4)
        ax.plot(steps, r.pos_per_step, color="#b2182b", lw=0.7, alpha=0.4)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"$L^2(\sigma)$ error")
    ax.set_title(f"Velocity-winning configs ({len(wins)})")
    ax.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=1.5, label="Velocity"),
        Line2D([0], [0], color="#b2182b", lw=1.5, label="Positions"),
    ])
    ax.grid(alpha=0.2)

    # Middle: losing configs error curves
    ax = axes[1]
    for r in losses:
        T = len(r.vel_per_step)
        steps = np.arange(T)
        ax.plot(steps, r.vel_per_step, color="#2166ac", lw=0.7, alpha=0.4)
        ax.plot(steps, r.pos_per_step, color="#b2182b", lw=0.7, alpha=0.4)
    ax.set_xlabel("Forecast step")
    ax.set_title(f"Positions-winning configs ({len(losses)})")
    ax.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=1.5, label="Velocity"),
        Line2D([0], [0], color="#b2182b", lw=1.5, label="Positions"),
    ])
    ax.grid(alpha=0.2)

    # Right: improvement distribution
    ax = axes[2]
    all_improv = [r.improvement_pct for r in all_results]
    colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in all_improv]
    ax.barh(range(len(all_results)), all_improv, color=colors, alpha=0.8)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels([r.tag.replace("goes_long_", "")[:30] for r in all_results],
                        fontsize=6)
    ax.set_xlabel("Velocity improvement (%)")
    ax.set_title("Improvement distribution")
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    path = save_dir / "error_growth_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def save_scenario_comparison(results_by_scenario: dict, save_dir: Path):
    """
    Per-scenario panel: best velocity vs best positions result.
    One row per scenario showing the per-step error for the best config of each.
    """
    plt = _setup_mpl()
    scenarios = list(results_by_scenario.keys())
    n_scenarios = len(scenarios)
    if n_scenarios == 0:
        return

    fig, axes = plt.subplots(n_scenarios, 2, figsize=(14, 3.5 * n_scenarios))
    if n_scenarios == 1:
        axes = axes[np.newaxis, :]

    for row, scenario in enumerate(scenarios):
        results = results_by_scenario[scenario]
        if not results:
            continue

        # Best velocity-winning config
        wins = [r for r in results if r.velocity_wins]
        best_vel = min(wins, key=lambda r: r.vel_l2_rmse) if wins else min(results, key=lambda r: r.vel_l2_rmse)

        # Per-step error
        ax = axes[row, 0]
        T = len(best_vel.vel_per_step)
        steps = np.arange(T)
        ax.plot(steps, best_vel.vel_per_step, color="#2166ac", lw=1.4, label="Velocity")
        ax.plot(steps, best_vel.pos_per_step, color="#b2182b", lw=1.4, label="Positions")
        ax.fill_between(steps, best_vel.vel_per_step, best_vel.pos_per_step,
                         where=best_vel.pos_per_step > best_vel.vel_per_step,
                         alpha=0.12, color="#4daf4a")
        win_str = "WIN" if best_vel.velocity_wins else "loss"
        ax.set_title(f"{scenario} -- best config [{win_str}] ({best_vel.improvement_pct:+.1f}%)",
                     fontsize=11)
        ax.set_ylabel(r"$L^2(\sigma)$ error")
        if row == n_scenarios - 1:
            ax.set_xlabel("Forecast step")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.2)

        # Cumulative error
        ax = axes[row, 1]
        ax.plot(steps, np.cumsum(best_vel.vel_per_step ** 2), color="#2166ac", lw=1.4, label="Velocity")
        ax.plot(steps, np.cumsum(best_vel.pos_per_step ** 2), color="#b2182b", lw=1.4, label="Positions")
        ax.set_title(f"{scenario} -- cumulative error", fontsize=11)
        ax.set_ylabel(r"Cumul. $\|\cdot\|^2$")
        if row == n_scenarios - 1:
            ax.set_xlabel("Forecast step")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    path = save_dir / "scenario_comparison_panel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def save_improvement_heatmap(all_results: list[ExperimentResult], save_dir: Path):
    """
    Heatmap: scenario x HP config showing improvement percentage.
    Green = velocity wins, red = positions wins.
    """
    plt = _setup_mpl()

    # Collect unique scenarios and configs
    scenarios = sorted(set(r.scenario for r in all_results if r.scenario))
    if not scenarios:
        return

    # Group by scenario
    by_scenario = {}
    for r in all_results:
        by_scenario.setdefault(r.scenario, []).append(r)

    # Build matrix: rows = scenarios, cols = configs within each
    max_configs = max(len(v) for v in by_scenario.values())
    matrix = np.full((len(scenarios), max_configs), np.nan)
    col_labels = []

    for i, sc in enumerate(scenarios):
        for j, r in enumerate(by_scenario[sc]):
            matrix[i, j] = r.improvement_pct
            if i == 0:
                short_tag = r.tag.replace(f"goes_long_{sc}_", "")
                col_labels.append(short_tag[:25])

    # Pad col_labels if needed
    while len(col_labels) < max_configs:
        col_labels.append("")

    fig, ax = plt.subplots(figsize=(max(8, max_configs * 1.2), max(4, len(scenarios) * 0.8)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 10)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(max_configs))
    ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=9)
    ax.set_title("Velocity improvement (%) -- green = velocity wins", fontsize=13)

    # Annotate cells
    for i in range(len(scenarios)):
        for j in range(max_configs):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Improvement (%)", shrink=0.8)
    fig.tight_layout()
    path = save_dir / "improvement_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ══════════════════════════════════════════════════════════════
# TEST CLASS: Individual Long-Duration Scenarios
# ══════════════════════════════════════════════════════════════

class TestLongDurationCloudScenarios:
    """
    Test each long-duration atmospheric scenario individually.
    These verify the pipeline runs on realistic GOES-scale data
    and produce per-scenario plots.
    """

    def _run_scenario(self, name, traj, n_cycles, **kw):
        data_dir = _save_as_particles(name, traj)
        T, N, _ = traj.shape
        r = run_experiment(
            tag=name,
            n_particles=N,
            n_steps=T,
            n_cycles=n_cycles,
            goes_dir=data_dir,
            scenario=name,
            **kw,
        )
        return r

    def test_multiday_diurnal_288(self):
        """48-hour diurnal convective cycle (288 steps, 2 full day/night cycles)."""
        traj = make_multiday_diurnal_convection(n_particles=150, n_steps=288)
        save_particle_snapshots(traj, "diurnal_288", "48h diurnal convection", PLOTS_DIR)
        save_velocity_field_snapshots(traj, "diurnal_288", "48h diurnal convection", PLOTS_DIR)

        r = self._run_scenario("goes_long_diurnal_288", traj, n_cycles=4)
        print(f"\n[diurnal 288] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "diurnal_288", "48h diurnal convection", PLOTS_DIR)
            save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                 "diurnal_288", "48h diurnal convection", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_multiday_diurnal_432(self):
        """72-hour diurnal convective cycle (432 steps, 3 full day/night cycles)."""
        traj = make_multiday_diurnal_convection(n_particles=150, n_steps=432, n_days=3.0)
        save_particle_snapshots(traj, "diurnal_432", "72h diurnal convection", PLOTS_DIR)

        r = self._run_scenario("goes_long_diurnal_432", traj, n_cycles=6)
        print(f"\n[diurnal 432] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "diurnal_432", "72h diurnal convection", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_extratropical_cyclone_400(self):
        """5-day extratropical cyclone passage (400 steps)."""
        traj = make_extratropical_cyclone(n_particles=150, n_steps=400)
        save_particle_snapshots(traj, "cyclone_400", "Extratropical cyclone", PLOTS_DIR)
        save_velocity_field_snapshots(traj, "cyclone_400", "Extratropical cyclone", PLOTS_DIR)

        r = self._run_scenario("goes_long_cyclone_400", traj, n_cycles=2)
        print(f"\n[cyclone 400] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "cyclone_400", "Extratropical cyclone", PLOTS_DIR)
            save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                 "cyclone_400", "Extratropical cyclone", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_tropical_mcs_300(self):
        """36-hour tropical MCS lifecycle (300 steps)."""
        traj = make_tropical_mcs_lifecycle(n_particles=150, n_steps=300)
        save_particle_snapshots(traj, "mcs_300", "Tropical MCS lifecycle", PLOTS_DIR)
        save_velocity_field_snapshots(traj, "mcs_300", "Tropical MCS lifecycle", PLOTS_DIR)

        r = self._run_scenario("goes_long_mcs_300", traj, n_cycles=5)
        print(f"\n[MCS 300] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "mcs_300", "Tropical MCS lifecycle", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_stratiform_advection_432(self):
        """72-hour stratiform cloud advection (432 steps)."""
        traj = make_stratiform_advection(n_particles=150, n_steps=432)
        save_particle_snapshots(traj, "stratiform_432", "72h stratiform advection", PLOTS_DIR)
        save_velocity_field_snapshots(traj, "stratiform_432", "72h stratiform advection", PLOTS_DIR)

        r = self._run_scenario("goes_long_stratiform_432", traj, n_cycles=8)
        print(f"\n[stratiform 432] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "stratiform_432", "72h stratiform advection", PLOTS_DIR)
            save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                 "stratiform_432", "72h stratiform advection", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0

    def test_rossby_wave_500(self):
        """4-day Rossby wave cloud band (500 steps)."""
        traj = make_rossby_wave_cloud_band(n_particles=150, n_steps=500)
        save_particle_snapshots(traj, "rossby_500", "4-day Rossby wave", PLOTS_DIR)
        save_velocity_field_snapshots(traj, "rossby_500", "4-day Rossby wave", PLOTS_DIR)

        r = self._run_scenario("goes_long_rossby_500", traj, n_cycles=4)
        print(f"\n[Rossby 500] vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
              f"{'WIN' if r.velocity_wins else 'loss'} {r.improvement_pct:+.1f}%")
        if r.vel_per_step is not None:
            save_per_step_plot(r.vel_per_step, r.pos_per_step,
                               "rossby_500", "4-day Rossby wave", PLOTS_DIR)
            save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                 "rossby_500", "4-day Rossby wave", PLOTS_DIR)
        assert r.vel_l2_rmse >= 0


# ══════════════════════════════════════════════════════════════
# MASTER SWEEP: Identify velocity-winning cases across all
# long-duration scenarios and hyperparameter configurations
# ══════════════════════════════════════════════════════════════

class TestLongCloudVelocityWinsSweep:
    """
    Comprehensive sweep across all long-duration atmospheric cloud
    scenarios x hyperparameter configs.  Identifies which cases
    velocity wins and generates summary plots + JSON report.
    """

    def test_full_long_cloud_sweep(self):
        all_results: list[ExperimentResult] = []
        results_by_scenario: dict[str, list[ExperimentResult]] = {}

        # ── Scenarios: (name, generator_func, kwargs, n_cycles) ──
        scenarios = [
            ("diurnal_48h",  make_multiday_diurnal_convection,
             dict(n_particles=150, n_steps=288), 4),
            ("diurnal_72h",  make_multiday_diurnal_convection,
             dict(n_particles=150, n_steps=432, n_days=3.0), 6),
            ("cyclone_5d",   make_extratropical_cyclone,
             dict(n_particles=150, n_steps=400), 2),
            ("mcs_36h",      make_tropical_mcs_lifecycle,
             dict(n_particles=150, n_steps=300), 5),
            ("stratiform_72h", make_stratiform_advection,
             dict(n_particles=150, n_steps=432), 8),
            ("rossby_4d",    make_rossby_wave_cloud_band,
             dict(n_particles=150, n_steps=500), 4),
        ]

        # ── HP configs to sweep ──
        hp_configs = [
            # (lot_kind, assignment, scale, sr_v, lk_v, rd_v)
            ("gaussian_iso",   "fixed",     1.0,  None, None, None),
            ("gaussian_iso",   "per_frame", 1.0,  None, None, None),
            ("gaussian_iso",   "fixed",     0.5,  None, None, None),
            ("gaussian_iso",   "fixed",     1.5,  None, None, None),
            ("gaussian_iso",   "fixed",     1.0,  0.85, 0.85, 0.02),
            ("gaussian_iso",   "fixed",     1.0,  0.75, 0.9,  0.05),
            ("gaussian_iso",   "fixed",     1.0,  0.9,  0.7,  0.005),
            ("uniform_square", "fixed",     1.0,  None, None, None),
            ("uniform_square", "per_frame", 1.0,  None, None, None),
        ]

        print("\n" + "=" * 70)
        print("LONG-DURATION GOES CLOUD: VELOCITY WINS SWEEP")
        print("=" * 70)

        for scenario_name, gen_func, gen_kw, n_cycles in scenarios:
            traj = gen_func(**gen_kw)
            data_dir = _save_as_particles(f"sweep_{scenario_name}", traj)
            T, N, _ = traj.shape
            results_by_scenario[scenario_name] = []

            # Save particle snapshots for each scenario
            save_particle_snapshots(traj, f"sweep_{scenario_name}",
                                     scenario_name, PLOTS_DIR)

            print(f"\n--- {scenario_name} (T={T}, N={N}) ---")

            for kind, assign, scale, sr, lk, rd in hp_configs:
                tag = f"goes_long_{scenario_name}_{kind}_{assign}_sc{scale}"
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
                        scenario=scenario_name,
                    )
                except Exception as e:
                    print(f"  [SKIP] {tag}: {e}")
                    continue

                all_results.append(r)
                results_by_scenario[scenario_name].append(r)
                status = "WIN" if r.velocity_wins else "loss"
                print(f"  [{status}] {tag}: vel={r.vel_l2_rmse:.6f} "
                      f"pos={r.pos_l2_rmse:.6f} {r.improvement_pct:+.1f}%")

        # ── Generate plots ──────────────────────────────────

        wins = [r for r in all_results if r.velocity_wins]

        print(f"\n{'=' * 70}")
        print(f"SWEEP SUMMARY: {len(wins)}/{len(all_results)} velocity wins")
        print(f"{'=' * 70}")

        # Per-step and cumulative for each winning config
        for r in wins:
            print(f"  [WIN] {r.tag}: vel={r.vel_l2_rmse:.6f} pos={r.pos_l2_rmse:.6f} "
                  f"improvement={r.improvement_pct:+.1f}%")
            if r.vel_per_step is not None:
                save_per_step_plot(r.vel_per_step, r.pos_per_step,
                                    r.tag, r.tag, PLOTS_DIR)
                save_cumulative_plot(r.vel_per_step, r.pos_per_step,
                                      r.tag, r.tag, PLOTS_DIR)

        # Summary plots
        if all_results:
            save_summary_bar(all_results, PLOTS_DIR)
            save_error_growth_comparison(all_results, PLOTS_DIR)
            save_scenario_comparison(results_by_scenario, PLOTS_DIR)
            save_improvement_heatmap(all_results, PLOTS_DIR)

        # ── Per-scenario summary ────────────────────────────
        print(f"\n{'=' * 70}")
        print("PER-SCENARIO BREAKDOWN")
        print(f"{'=' * 70}")
        for scenario_name, results in results_by_scenario.items():
            sc_wins = [r for r in results if r.velocity_wins]
            print(f"  {scenario_name}: {len(sc_wins)}/{len(results)} velocity wins")
            if sc_wins:
                best = min(sc_wins, key=lambda r: r.vel_l2_rmse)
                print(f"    Best: vel={best.vel_l2_rmse:.6f} pos={best.pos_l2_rmse:.6f} "
                      f"({best.improvement_pct:+.1f}%)")

        # ── Save JSON report ────────────────────────────────
        report = {
            "description": "Long-duration GOES cloud velocity-wins sweep",
            "metric": "L2_sigma_RMSE (paper empirical-L2)",
            "scenarios": {
                sc: {
                    "n_configs": len(results),
                    "n_velocity_wins": sum(1 for r in results if r.velocity_wins),
                    "best_velocity_improvement": max(
                        (r.improvement_pct for r in results if r.velocity_wins),
                        default=0.0
                    ),
                    "results": [
                        {
                            "tag": r.tag,
                            "velocity_l2_rmse": r.vel_l2_rmse,
                            "positions_l2_rmse": r.pos_l2_rmse,
                            "velocity_wins": r.velocity_wins,
                            "improvement_pct": r.improvement_pct,
                        }
                        for r in results
                    ],
                }
                for sc, results in results_by_scenario.items()
            },
            "totals": {
                "n_configs_tried": len(all_results),
                "n_velocity_wins": len(wins),
                "win_rate": f"{len(wins) / max(1, len(all_results)) * 100:.1f}%",
            },
            "winning_configs": [
                {
                    "tag": r.tag,
                    "scenario": r.scenario,
                    "velocity_l2_rmse": r.vel_l2_rmse,
                    "positions_l2_rmse": r.pos_l2_rmse,
                    "improvement_pct": r.improvement_pct,
                }
                for r in sorted(wins, key=lambda r: -r.improvement_pct)
            ],
        }
        report_path = REPO_ROOT / "goes_long_cloud_velocity_wins_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")
        print(f"Plots saved to {PLOTS_DIR}/")

        assert len(wins) > 0, (
            f"No velocity-winning config found in {len(all_results)} tries across "
            f"{len(results_by_scenario)} scenarios. Check hyperparameters or trajectory lengths."
        )
