#!/usr/bin/env python3
"""
GOES-16 satellite cloud forecasting visualization.

Properly handles geostationary projection so particles align
with cloud features in the satellite imagery.

Produces:
  1. (A)-(B)-(C) extraction figure matching paper style
  2. 3-row × N-col forecast comparison on satellite imagery
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_DIR = REPO / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GOES config ──
REGION = [-105, -95, 33, 43]  # [west, east, south, north]
BAND = "CMI_C13"
N_PARTICLES = 500
START = datetime(2024, 7, 15, 18, 0)
N_FRAMES = 25
CADENCE_MIN = 10
DATA_DIR = REPO / "goes_data_500"


# ══════════════════════════════════════════════════════════════
# Geostationary projection utilities
# ══════════════════════════════════════════════════════════════

def goes_to_latlon(ds):
    """Convert GOES geostationary x/y (radians) to lat/lon arrays."""
    proj = ds.goes_imager_projection
    H_sat = proj.attrs["perspective_point_height"]
    lon0 = np.radians(proj.attrs["longitude_of_projection_origin"])
    r_eq = proj.attrs["semi_major_axis"]
    r_pol = proj.attrs["semi_minor_axis"]

    x = ds.x.values
    y = ds.y.values
    xx, yy = np.meshgrid(x, y)

    a = np.sin(xx)**2 + np.cos(xx)**2 * (
        np.cos(yy)**2 + (r_eq / r_pol)**2 * np.sin(yy)**2
    )
    b = -2 * H_sat * np.cos(xx) * np.cos(yy)
    c = H_sat**2 - r_eq**2
    det = b**2 - 4 * a * c

    valid = det >= 0
    rs = np.full_like(det, np.nan)
    rs[valid] = (-b[valid] - np.sqrt(det[valid])) / (2 * a[valid])

    sx = rs * np.cos(xx) * np.cos(yy)
    sy = -rs * np.sin(xx)
    sz = rs * np.cos(xx) * np.sin(yy)

    lons = np.degrees(lon0 - np.arctan2(sy, H_sat - sx))
    lats = np.degrees(
        np.arctan((r_eq / r_pol)**2 * sz / np.sqrt((H_sat - sx)**2 + sy**2))
    )
    return lats, lons, valid


def subset_to_region(field, lats, lons, valid, region):
    """Crop field/coords to geographic bounding box. Returns cropped arrays + extent."""
    west, east, south, north = region
    mask = valid & (lons >= west) & (lons <= east) & (lats >= south) & (lats <= north)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return field, lats, lons, [west, east, south, north]
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return (
        field[r0:r1, c0:c1],
        lats[r0:r1, c0:c1],
        lons[r0:r1, c0:c1],
        [lons[r0:r1, c0:c1].min(), lons[r0:r1, c0:c1].max(),
         lats[r0:r1, c0:c1].min(), lats[r0:r1, c0:c1].max()],
    )


# ══════════════════════════════════════════════════════════════
# Download & particle extraction
# ══════════════════════════════════════════════════════════════

def download_frames():
    """Download GOES frames, subset to region, return fields + geo info."""
    from goes2go import GOES

    G = GOES(satellite=16, product="ABI-L2-MCMIPC", domain="C")
    frames = []
    current = START

    # Get projection info from first frame
    ds0 = G.nearesttime(current)
    lats_full, lons_full, valid_full = goes_to_latlon(ds0)

    for i in range(N_FRAMES):
        try:
            ds = G.nearesttime(current)
            raw = ds[BAND].values
            field, lats, lons, extent = subset_to_region(
                raw, lats_full, lons_full, valid_full, REGION
            )
            frames.append({
                "time": current,
                "field": field,
                "lats": lats,
                "lons": lons,
                "extent": extent,  # [west, east, south, north]
            })
            print(f"  Frame {i:2d}: {current.strftime('%H:%M')} UTC  "
                  f"field={field.shape}", flush=True)
        except Exception as e:
            print(f"  Frame {i:2d}: FAILED {e}")
        current += timedelta(minutes=CADENCE_MIN)

    return frames


def extract_particles(field, n_particles):
    """
    Sample particles from IR cloud field using quantile-based sampling.
    Returns normalised [0,1]² coordinates where (0,0)=top-left.
    Particles concentrate where clouds are (cold = bright in IR).
    """
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W]
    values = field.flatten().astype(np.float64)
    valid = np.isfinite(values)
    xx_f = xx.flatten()[valid].astype(np.float64)
    yy_f = yy.flatten()[valid].astype(np.float64)
    vals = values[valid]

    # Adaptive cloud weighting: coldest 30% of pixels in this frame
    T_cut = np.percentile(vals, 30)
    T_ceil = np.percentile(vals, 90)
    steepness = max((T_ceil - T_cut) * 0.1, 1.0)
    taper = 1.0 / (1.0 + np.exp((vals - T_cut) / steepness))
    weights = taper * np.maximum(T_ceil - vals, 0)
    s = weights.sum()
    if s > 0:
        weights /= s
    else:
        weights = np.ones_like(weights) / len(weights)

    # Quantile-based deterministic sampling
    cdf = np.cumsum(weights)
    qs = np.linspace(0.5 / n_particles, 1.0 - 0.5 / n_particles, n_particles)
    indices = np.clip(np.searchsorted(cdf, qs, side="right"), 0, len(weights) - 1)

    norm_x = xx_f[indices] / max(W - 1, 1)
    norm_y = yy_f[indices] / max(H - 1, 1)
    return np.column_stack([norm_x, norm_y]).astype(np.float32)


def particles_to_geo(particles_norm, extent):
    """Map [0,1]² normalised particles to geographic lon/lat for overlay."""
    west, east, south, north = extent
    lon = west + particles_norm[:, 0] * (east - west)
    lat = north - particles_norm[:, 1] * (north - south)  # y=0 is top (north)
    return lon, lat


# ══════════════════════════════════════════════════════════════
# Figure 1: (A)-(B)-(C) extraction panel
# ══════════════════════════════════════════════════════════════

def plot_extraction(frame, particles, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white",
                              gridspec_kw={"width_ratios": [1, 1, 0.85],
                                           "wspace": 0.12})

    field = frame["field"]
    ext = frame["extent"]
    img_extent = [ext[0], ext[1], ext[2], ext[3]]  # [left, right, bottom, top]
    lon, lat = particles_to_geo(particles, ext)

    # (A) Satellite imagery
    ax = axes[0]
    ax.imshow(field, cmap="gray", extent=img_extent, aspect="auto",
              origin="upper", interpolation="bilinear")
    ax.set_xlabel("Longitude (°W)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.set_title("(A)  Satellite Imagery", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)

    # (B) Particles on satellite
    ax = axes[1]
    ax.imshow(field, cmap="gray", extent=img_extent, aspect="auto",
              origin="upper", interpolation="bilinear")
    ax.scatter(lon, lat, s=6, c="#FF6D00", alpha=0.85,
               edgecolors="white", linewidths=0.15, zorder=3)
    ax.set_xlabel("Longitude (°W)", fontsize=10)
    ax.set_title("(B)  Particle Extraction", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    ax.set_yticklabels([])
    ax.text(0.02, 0.97, f"Particles (N={len(lon)})", transform=ax.transAxes,
            fontsize=8, va="top", color="#FF6D00", fontweight="bold",
            bbox=dict(fc="white", alpha=0.85, ec="#FF6D00", lw=0.5,
                      boxstyle="round,pad=0.2"))

    # Arrow A → B
    fig.text(0.365, 0.50, "→", fontsize=30, ha="center", va="center",
             color="#1565C0", fontweight="bold")

    # (C) Measure representation
    ax = axes[2]
    ax.set_facecolor("#fafafa")
    ax.scatter(particles[:, 0], particles[:, 1], s=8, c="#FF6D00",
               alpha=0.7, edgecolors="none")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(1.03, -0.03)
    ax.set_aspect("equal")
    ax.set_title("(C)  Measure Representation", fontsize=12,
                 fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    ax.set_yticklabels([])
    ax.text(0.95, 0.08, r"$\mu_t \in \mathcal{P}_2(\mathbb{R}^2)$",
            transform=ax.transAxes, fontsize=14, ha="right", va="bottom",
            style="italic", color="#333")

    # Arrow B → C
    fig.text(0.665, 0.50, "→", fontsize=30, ha="center", va="center",
             color="#1565C0", fontweight="bold")

    time_str = frame["time"].strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"GOES-16 ABI Band 13 (IR 10.3 μm)  —  {time_str}\n"
        f"Region: {REGION[2]}°–{REGION[3]}°N, {abs(REGION[0])}°–{abs(REGION[1])}°W",
        fontsize=11, y=1.02, color="#333")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Figure 2: 3-row forecast on satellite imagery
# ══════════════════════════════════════════════════════════════

def plot_forecast(frames, all_particles_norm, out_path):
    """
    Row 1: Ground truth particles on satellite
    Row 2: Velocity forecast particles on satellite
    Row 3: Position forecast particles on satellite
    """
    vel_dir = REPO / "forecast_output/goes_cloud_500_N_500/snapshot_begin/frac100/velocity_resFromOrigN_100pct_pipeline_velocity"
    pos_dir = REPO / "forecast_output/goes_cloud_500_N_500/snapshot_begin/frac100/positions_resFromOrigN_100pct_pipeline_positions"

    vel_pred = np.load(vel_dir / "predicted_measures_X.npy")
    pos_pred = np.load(pos_dir / "predicted_measures_X.npy")
    vel_warm = np.load(vel_dir / "warm_start_info.npy")
    vel_ws = int(vel_warm[0])
    fc_start = vel_ws + 1
    F = min(vel_pred.shape[0], pos_pred.shape[0])

    with open(REPO / "pipeline_run_summary.json") as fh:
        summary = json.load(fh)
    vel_sink = np.array(summary.get("sinkhorn_per_step_velocity", []))
    pos_sink = np.array(summary.get("sinkhorn_per_step_positions", []))

    # Pick columns
    n_cols = 6
    if F <= n_cols - 1:
        fidxs = list(range(F))
    else:
        fidxs = sorted(set(
            int(round(i * (F - 1) / (n_cols - 2))) for i in range(n_cols - 1)
        ))
    col_abs = [vel_ws] + [fc_start + i for i in fidxs]
    col_abs = col_abs[:n_cols]
    n_cols = len(col_abs)

    def w2_at(arr, idx):
        return np.sqrt(max(arr[idx], 0.0)) if 0 <= idx < len(arr) else None

    GT_C  = "#FF6D00"
    VEL_C = "#00E5FF"
    POS_C = "#FF1744"
    S = 12

    fig = plt.figure(figsize=(3.8 * n_cols, 11.5), facecolor="white")
    gs = gridspec.GridSpec(3, n_cols, hspace=0.18, wspace=0.04,
                           left=0.06, right=0.98, top=0.88, bottom=0.05)

    for ci, t_abs in enumerate(col_abs):
        is_train = (t_abs <= vel_ws)
        fi = t_abs - fc_start
        frame_idx = min(t_abs, len(frames) - 1)
        frame = frames[frame_idx]
        ext = frame["extent"]
        img_ext = [ext[0], ext[1], ext[2], ext[3]]

        for ri in range(3):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(frame["field"], cmap="gray", extent=img_ext,
                      aspect="auto", origin="upper", interpolation="bilinear",
                      alpha=0.9)
            ax.set_xlim(img_ext[0], img_ext[1])
            ax.set_ylim(img_ext[2], img_ext[3])
            ax.tick_params(labelsize=5, length=2)
            if ci > 0:
                ax.set_yticklabels([])
            if ri < 2:
                ax.set_xticklabels([])

            if ri == 0:
                # Ground truth
                if t_abs < len(all_particles_norm):
                    lon, lat = particles_to_geo(all_particles_norm[t_abs], ext)
                    ax.scatter(lon, lat, s=S, c=GT_C, alpha=0.8,
                               edgecolors="black", linewidths=0.2, zorder=3)
                time_str = frame["time"].strftime("%H:%M")
                tag = f"t={t_abs}"
                if is_train:
                    tag += " (train)"
                ax.set_title(f"{tag}  —  {time_str} UTC", fontsize=8,
                             fontweight="bold", pad=4)

            elif ri == 1:
                # Velocity forecast
                if 0 <= fi < F:
                    lon, lat = particles_to_geo(vel_pred[fi], ext)
                    ax.scatter(lon, lat, s=S, c=VEL_C, alpha=0.75,
                               edgecolors="black", linewidths=0.2, zorder=3)
                    w2 = w2_at(vel_sink, fi)
                    if w2 is not None:
                        ax.text(0.97, 0.03, f"W$_2$={w2:.4f}",
                                transform=ax.transAxes, fontsize=6, va="bottom",
                                ha="right", color="white", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc=VEL_C, alpha=0.85, lw=0))
                elif is_train and t_abs < len(all_particles_norm):
                    lon, lat = particles_to_geo(all_particles_norm[t_abs], ext)
                    ax.scatter(lon, lat, s=S, c=VEL_C, alpha=0.4,
                               edgecolors="black", linewidths=0.2, zorder=3)
                    ax.text(0.5, 0.5, "training", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold",
                            bbox=dict(fc="black", alpha=0.45, ec="none",
                                      boxstyle="round,pad=0.3"))

            elif ri == 2:
                # Position forecast
                if 0 <= fi < F:
                    lon, lat = particles_to_geo(pos_pred[fi], ext)
                    ax.scatter(lon, lat, s=S, c=POS_C, alpha=0.75,
                               edgecolors="black", linewidths=0.2, zorder=3)
                    w2 = w2_at(pos_sink, fi)
                    if w2 is not None:
                        ax.text(0.97, 0.03, f"W$_2$={w2:.4f}",
                                transform=ax.transAxes, fontsize=6, va="bottom",
                                ha="right", color="white", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc=POS_C, alpha=0.85, lw=0))
                elif is_train and t_abs < len(all_particles_norm):
                    lon, lat = particles_to_geo(all_particles_norm[t_abs], ext)
                    ax.scatter(lon, lat, s=S, c=POS_C, alpha=0.4,
                               edgecolors="black", linewidths=0.2, zorder=3)
                    ax.text(0.5, 0.5, "training", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold",
                            bbox=dict(fc="black", alpha=0.45, ec="none",
                                      boxstyle="round,pad=0.3"))

    # Row labels
    fig.text(0.018, 0.77, "Ground Truth", fontsize=10, fontweight="bold",
             va="center", ha="center", rotation=90, color=GT_C)
    fig.text(0.018, 0.50, "LOT Velocity\nForecast", fontsize=10,
             fontweight="bold", va="center", ha="center", rotation=90, color=VEL_C)
    fig.text(0.018, 0.22, "LOT Position\nForecast", fontsize=10,
             fontweight="bold", va="center", ha="center", rotation=90, color=POS_C)

    # Legend
    legend_els = [
        Line2D([], [], marker="o", ls="none", color=GT_C, markersize=5,
               label="True cloud particles"),
        Line2D([], [], marker="o", ls="none", color=VEL_C, markersize=5,
               label="Velocity forecast"),
        Line2D([], [], marker="o", ls="none", color=POS_C, markersize=5,
               label="Position forecast"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=8,
               frameon=True, fancybox=True, edgecolor="#ccc",
               bbox_to_anchor=(0.52, -0.005))

    # Title
    vel_w2 = summary.get("sinkhorn_w2_velocity")
    pos_w2 = summary.get("sinkhorn_w2_positions")
    fig.suptitle("GOES-16 Cloud Forecasting on Satellite Imagery",
                 fontsize=13, fontweight="bold", y=0.96)
    if vel_w2 and pos_w2:
        winner = "Velocity" if vel_w2 < pos_w2 else "Position"
        ratio = pos_w2 / max(vel_w2, 1e-12)
        fig.text(0.52, 0.935,
                 f"Sinkhorn W$_2$:  Velocity = {vel_w2:.4f}    "
                 f"Position = {pos_w2:.4f}    ({ratio:.2f}x, {winner} wins)",
                 ha="center", fontsize=9, color="#444")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  GOES-16 Satellite Overlay (proper projection)")
    print("=" * 60)

    # Step 1: Download and subset all frames
    print("\n[1/4] Downloading + subsetting GOES-16 frames...")
    frames = download_frames()
    print(f"  Got {len(frames)} frames, field shape: {frames[0]['field'].shape}")
    print(f"  Extent: {frames[0]['extent']}")

    # Step 2: Extract particles from subsetted fields
    print("\n[2/4] Extracting particles...")
    all_particles = []
    for frame in frames:
        p = extract_particles(frame["field"], N_PARTICLES)
        all_particles.append(p)

    # Verify alignment on first frame
    p0 = all_particles[0]
    lon0, lat0 = particles_to_geo(p0, frames[0]["extent"])
    print(f"  Particle lon range: [{lon0.min():.2f}, {lon0.max():.2f}]")
    print(f"  Particle lat range: [{lat0.min():.2f}, {lat0.max():.2f}]")
    print(f"  Image extent: {frames[0]['extent']}")

    # Step 3: (A)-(B)-(C) extraction figure
    print("\n[3/4] Generating extraction figure...")
    plot_extraction(frames[0], all_particles[0],
                    OUT_DIR / "goes_particle_extraction.png")

    # Step 4: 3-row forecast on satellite
    print("\n[4/4] Generating forecast comparison figure...")
    plot_forecast(frames, all_particles,
                  OUT_DIR / "goes_cloud_forecast_comparison.png")

    print("\nDone!")
