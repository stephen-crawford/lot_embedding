#!/usr/bin/env python3

# =============================================================================
# INSTALLATION (run once)
# =============================================================================
# pip install goes2go xarray netCDF4 h5py matplotlib cartopy numpy scipy

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GOES data access
try:
    from goes2go import GOES
    GOES2GO_AVAILABLE = True
except ImportError:
    GOES2GO_AVAILABLE = False
    print("goes2go not installed. Run: pip install goes2go")

# Optional: for visualization
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("cartopy not available for map visualization")

# =============================================================================
# CONFIGURATION
# =============================================================================
class GOESConfig:
    """Configuration for GOES data acquisition."""
    
    # Target region: Great Plains (good for diurnal convection)
    # Format: [west_lon, east_lon, south_lat, north_lat]
    REGION_GREAT_PLAINS = [-105, -95, 33, 43]  # Kansas/Oklahoma
    REGION_FLORIDA = [-88, -78, 24, 32]         # Florida peninsula
    REGION_GULF_COAST = [-98, -88, 26, 34]      # Texas/Louisiana coast
    
    # Default region
    REGION = REGION_GREAT_PLAINS
    
    # Temporal settings
    CADENCE_MINUTES = 10  # MCMIPC native cadence
    
    # Band selection for cloud density proxy
    # CMI_C02: Visible red (daytime only, best cloud structure)
    # CMI_C13: IR longwave (day AND night, surface+clouds)
    # CMI_C14: IR window (sea surface temp, clouds)
    BAND_VISIBLE = 'CMI_C02'  # Daytime only
    BAND_IR_CLEAN = 'CMI_C13'  # 24-hour (recommended for continuous cycles)
    BAND_IR_WINDOW = 'CMI_C14'
    
    # Default band for LOT pipeline
    PRIMARY_BAND = 'CMI_C13'  # IR works day and night
    
    # Particle conversion settings (to match synthetic benchmarks)
    TARGET_PARTICLES = 500  # Match your swirling_cluster/geodesic setups
    
    # Output paths
    OUTPUT_DIR = Path('./goes_data')


# =============================================================================
# DATA ACQUISITION
# =============================================================================
def download_goes_timeseries(
    start_time: datetime,
    end_time: datetime,
    region: list = None,
    band: str = None,
    save_dir: Path = None,
    verbose: bool = True
) -> list:
    """
    Download a time series of GOES-16 MCMIPC images.
    
    Parameters
    ----------
    start_time : datetime
        Start of time window
    end_time : datetime
        End of time window
    region : list
        [west_lon, east_lon, south_lat, north_lat]
    band : str
        Band name (e.g., 'CMI_C13')
    save_dir : Path
        Directory to save downloaded files
    verbose : bool
        Print progress
        
    Returns
    -------
    list of xarray.Dataset
        Downloaded and subsetted imagery
    """
    if not GOES2GO_AVAILABLE:
        raise ImportError("goes2go required. Install with: pip install goes2go")
    
    region = region or GOESConfig.REGION
    band = band or GOESConfig.PRIMARY_BAND
    save_dir = save_dir or GOESConfig.OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GOES accessor
    G = GOES(satellite=16, product='ABI-L2-MCMIPC', domain='C')
    
    datasets = []
    current_time = start_time
    
    while current_time <= end_time:
        try:
            if verbose:
                print(f"Fetching {current_time.strftime('%Y-%m-%d %H:%M')}...", end=' ')
            
            # Download nearest image to requested time
            ds = G.nearesttime(current_time)
            
            # Subset to region of interest
            ds_subset = subset_region(ds, region)
            
            # Extract the requested band
            if band in ds_subset:
                datasets.append({
                    'time': current_time,
                    'data': ds_subset[band].values,
                    'lats': ds_subset['y'].values if 'y' in ds_subset else None,
                    'lons': ds_subset['x'].values if 'x' in ds_subset else None,
                    'full_ds': ds_subset
                })
                if verbose:
                    print("OK")
            else:
                if verbose:
                    print(f"Band {band} not found")
                    
        except Exception as e:
            if verbose:
                print(f"Failed: {e}")
        
        current_time += timedelta(minutes=GOESConfig.CADENCE_MINUTES)
    
    print(f"\nDownloaded {len(datasets)} images")
    return datasets


def subset_region(ds, region):
    """
    Subset xarray dataset to a geographic region.
    
    Note: GOES uses geostationary projection, so this is approximate.
    For precise subsetting, we'd need to transform coordinates.
    """
    west, east, south, north = region
    
    # Simple bounding box subset (approximate for geostationary)
    # This works reasonably well for CONUS-scale regions
    try:
        ds_subset = ds.sel(
            x=slice(west, east),
            y=slice(north, south)  # Note: y is typically inverted
        )
    except:
        # If coordinate-based selection fails, return full dataset
        ds_subset = ds
    
    return ds_subset


# =============================================================================
# CONVERSION TO PARTICLE REPRESENTATION
# =============================================================================
def cloud_field_to_particles(
    cloud_field: np.ndarray,
    n_particles: int = 500,
    threshold: float = None,
    method: str = 'weighted_sample',
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Convert a 2D cloud field to a particle representation.

    This creates an empirical measure from gridded data, suitable for LOT.

    Parameters
    ----------
    cloud_field : np.ndarray
        2D array of cloud values (reflectance or brightness temp)
    n_particles : int
        Number of particles to generate
    threshold : float
        Minimum value to consider as "cloud" (None = use all)
    method : str
        'weighted_sample': Sample positions weighted by cloud intensity
        'threshold_uniform': Uniform sample from above-threshold regions
        'density_grid': Aggregate to coarse grid with mass weights
        'consistent_quantile': Deterministic quantile-based sampling using a
            fixed set of uniform random numbers.  This produces pseudo-Lagrangian
            particles that track the cloud evolution instead of resampling noise.
    rng : np.random.Generator, optional
        Random number generator. If None, uses global numpy RNG.

    Returns
    -------
    np.ndarray, shape (n_particles, 2) or (n_particles, 3)
        Particle positions (and optionally weights)
    """
    H, W = cloud_field.shape

    # Create coordinate grids (normalized to [0,1] x [0,1])
    yy, xx = np.mgrid[0:H, 0:W]
    xx_norm = xx.flatten() / W
    yy_norm = yy.flatten() / H
    values = cloud_field.flatten()

    # Handle NaN/missing values
    valid_mask = ~np.isnan(values)
    xx_norm = xx_norm[valid_mask]
    yy_norm = yy_norm[valid_mask]
    values = values[valid_mask]

    if method in ('weighted_sample', 'consistent_quantile'):
        # For IR bands (brightness temp), clouds are COLD (low values)
        if values.mean() > 200:  # Likely brightness temperature in Kelvin
            # Adaptive cloud weighting: use the coldest fraction of pixels.
            # At night the surface cools, so a fixed threshold (e.g. 275K)
            # fails — the entire field falls below it.  Instead, define
            # "cloud" as the coldest 30% of pixels relative to this frame's
            # own distribution.  Weight = max(T_cut - BT, 0) concentrates
            # particles on the coldest (highest/thickest) clouds.
            T_cut = np.percentile(values, 30)  # coldest 30%
            T_ceil = np.percentile(values, 90)  # warm tail
            weights = np.maximum(T_cut - values, 0)
            # Add a mild sigmoid taper so the transition isn't a hard step
            steepness = max((T_ceil - T_cut) * 0.1, 1.0)
            taper = 1.0 / (1.0 + np.exp((values - T_cut) / steepness))
            weights = taper * np.maximum(T_ceil - values, 0)
        else:
            # Reflectance: brighter = more cloud
            weights = values

        # Apply threshold if specified
        if threshold is not None:
            weights[weights < threshold] = 0

        # Normalize to probability distribution
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        if method == 'consistent_quantile':
            # Deterministic quantile-based sampling: use evenly-spaced quantiles
            # of the CDF. As the cloud field evolves, the same quantiles track
            # the shifting mass, producing smooth pseudo-Lagrangian trajectories.
            cdf = np.cumsum(weights)
            # Evenly spaced quantile levels in (0, 1)
            quantiles = np.linspace(
                0.5 / n_particles, 1.0 - 0.5 / n_particles, n_particles
            )
            indices = np.searchsorted(cdf, quantiles, side='right')
            indices = np.clip(indices, 0, len(weights) - 1)
        else:
            # Random weighted sampling (original method — noisy for LOT)
            if rng is not None:
                indices = rng.choice(len(weights), size=n_particles, replace=True, p=weights)
            else:
                indices = np.random.choice(
                    len(weights),
                    size=n_particles,
                    replace=True,
                    p=weights
                )

        particles = np.column_stack([xx_norm[indices], yy_norm[indices]])
        
    elif method == 'threshold_uniform':
        # Binary: above threshold = cloud
        if threshold is None:
            threshold = np.median(values)
        
        cloud_mask = values > threshold
        cloud_xx = xx_norm[cloud_mask]
        cloud_yy = yy_norm[cloud_mask]
        
        if len(cloud_xx) >= n_particles:
            indices = np.random.choice(len(cloud_xx), size=n_particles, replace=False)
        else:
            indices = np.random.choice(len(cloud_xx), size=n_particles, replace=True)
        
        particles = np.column_stack([cloud_xx[indices], cloud_yy[indices]])
        
    elif method == 'density_grid':
        # Aggregate to coarser grid, return grid centers with mass weights
        grid_size = int(np.sqrt(n_particles))
        
        # Bin the data
        x_bins = np.linspace(0, 1, grid_size + 1)
        y_bins = np.linspace(0, 1, grid_size + 1)
        
        mass_grid = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                mask = (
                    (xx_norm >= x_bins[i]) & (xx_norm < x_bins[i+1]) &
                    (yy_norm >= y_bins[j]) & (yy_norm < y_bins[j+1])
                )
                if mask.any():
                    mass_grid[j, i] = values[mask].mean()
        
        # Create particles at grid centers with mass weights
        cx = (x_bins[:-1] + x_bins[1:]) / 2
        cy = (y_bins[:-1] + y_bins[1:]) / 2
        cxx, cyy = np.meshgrid(cx, cy)
        
        particles = np.column_stack([
            cxx.flatten(), 
            cyy.flatten(), 
            mass_grid.flatten()
        ])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return particles


def timeseries_to_particle_sequence(
    datasets: list,
    n_particles: int = 500,
    method: str = 'weighted_sample'
) -> np.ndarray:
    """
    Convert a time series of cloud fields to particle sequences.
    
    Parameters
    ----------
    datasets : list
        Output from download_goes_timeseries()
    n_particles : int
        Particles per time step
    method : str
        Conversion method (see cloud_field_to_particles)
        
    Returns
    -------
    np.ndarray, shape (T, n_particles, 2)
        Particle positions over time
    """
    T = len(datasets)
    particles_seq = np.zeros((T, n_particles, 2))
    
    for t, ds in enumerate(datasets):
        cloud_field = ds['data']
        particles = cloud_field_to_particles(
            cloud_field, 
            n_particles=n_particles,
            method=method
        )
        particles_seq[t] = particles[:, :2]  # Just positions
    
    return particles_seq


def timeseries_to_particles_safe(
    datasets: list,
    n_particles: int,
    rng: np.random.Generator,
    method: str = "consistent_quantile",
) -> np.ndarray:
    """
    Like timeseries_to_particle_sequence but tolerates empty or all-NaN fields per frame
    (subset failures, bad downloads) by falling back to uniform [0,1]^2 samples.

    Parameters
    ----------
    method : str
        'consistent_quantile' (default): Deterministic CDF-quantile sampling.
            Produces smooth pseudo-Lagrangian trajectories suitable for LOT
            velocity prediction. Eliminates sampling noise between frames.
        'weighted_sample': Original random weighted sampling (noisy).
    """
    T = len(datasets)
    out = np.zeros((T, n_particles, 2), dtype=np.float32)
    for t, ds in enumerate(datasets):
        cf = np.asarray(ds["data"], dtype=np.float64)
        if cf.size == 0:
            out[t] = rng.random((n_particles, 2)).astype(np.float32)
            continue
        fin = np.isfinite(cf)
        if not np.any(fin):
            out[t] = rng.random((n_particles, 2)).astype(np.float32)
            continue
        cf = np.where(fin, cf, np.nanmedian(cf[fin]))
        try:
            out[t] = cloud_field_to_particles(
                cf, n_particles=n_particles, method=method, rng=rng,
            )[:, :2]
        except ValueError:
            out[t] = rng.random((n_particles, 2)).astype(np.float32)
    return out


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_cloud_field(cloud_field: np.ndarray, title: str = None, ax=None):
    """Plot a single cloud field."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cloud_field, cmap='gray_r', origin='upper')
    plt.colorbar(im, ax=ax, label='Brightness Temperature (K)')
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    return ax


def plot_particles(particles: np.ndarray, title: str = None, ax=None):
    """Plot particle representation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.5, c='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    
    return ax


def plot_comparison(cloud_field: np.ndarray, particles: np.ndarray, time_label: str = None):
    """Side-by-side comparison of cloud field and particle representation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_cloud_field(cloud_field, title='Original Cloud Field', ax=axes[0])
    plot_particles(particles, title=f'Particle Representation (N={len(particles)})', ax=axes[1])
    
    if time_label:
        fig.suptitle(time_label, fontsize=14)
    
    plt.tight_layout()
    return fig


def animate_particle_sequence(particles_seq: np.ndarray, save_path: str = None):
    """Create animation of particle evolution."""
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter([], [], s=1, alpha=0.5, c='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    title = ax.set_title('')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter, title
    
    def update(frame):
        scatter.set_offsets(particles_seq[frame])
        title.set_text(f'Time step {frame}/{len(particles_seq)-1}')
        return scatter, title
    
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(particles_seq), interval=100, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    
    return anim


# =============================================================================
# MAIN WORKFLOW: QUICK TEST
# =============================================================================
def quick_test_single_image():
    """
    Download and visualize a single GOES image.
    Good for testing your setup before running full time series.
    """
    print("=" * 60)
    print("GOES-16 Quick Test: Single Image Download")
    print("=" * 60)
    
    if not GOES2GO_AVAILABLE:
        print("\nERROR: goes2go not installed!")
        print("Run: pip install goes2go xarray netCDF4")
        return None
    
    # Download a recent image
    G = GOES(satellite=16, product='ABI-L2-MCMIPC', domain='C')
    
    print("\nDownloading latest GOES-16 CONUS image...")
    ds = G.latest()
    
    print(f"Dataset variables: {list(ds.keys())[:5]}...")
    print(f"Time: {ds.attrs.get('time_coverage_start', 'unknown')}")
    
    # Extract IR band (works day and night)
    band = 'CMI_C13'
    if band in ds:
        cloud_field = ds[band].values
        print(f"\nBand {band} shape: {cloud_field.shape}")
        print(f"Value range: {np.nanmin(cloud_field):.1f} - {np.nanmax(cloud_field):.1f} K")
        
        # Convert to particles
        particles = cloud_field_to_particles(cloud_field, n_particles=500)
        print(f"Generated {len(particles)} particles")
        
        # Visualize
        fig = plot_comparison(cloud_field, particles, "GOES-16 CMI_C13 (IR Longwave)")
        plt.savefig('goes_test_output.png', dpi=150, bbox_inches='tight')
        print("\nSaved visualization to: goes_test_output.png")
        
        return cloud_field, particles
    else:
        print(f"Band {band} not found in dataset")
        return None


def download_diurnal_cycle(
    date: str = '2024-06-15',
    hours_start: int = 12,  # Start at noon UTC (morning in Central US)
    duration_hours: int = 24,
    region: list = None,
    n_particles: int = 500
):
    """
    Download a full diurnal cycle for LOT+RC training.
    
    Parameters
    ----------
    date : str
        Date in 'YYYY-MM-DD' format
    hours_start : int
        Starting hour (UTC)
    duration_hours : int
        Duration to download
    region : list
        Geographic bounds [west, east, south, north]
    n_particles : int
        Particles per time step
        
    Returns
    -------
    dict with:
        'particles': np.ndarray (T, N, 2)
        'times': list of datetime
        'raw_fields': list of 2D arrays
    """
    region = region or GOESConfig.REGION
    
    start = datetime.strptime(f"{date} {hours_start:02d}:00", "%Y-%m-%d %H:%M")
    end = start + timedelta(hours=duration_hours)
    
    print(f"Downloading {duration_hours}-hour cycle starting {start}")
    print(f"Region: {region}")
    print(f"Expected images: ~{duration_hours * 6} (10-min cadence)")
    
    # Download time series
    datasets = download_goes_timeseries(
        start_time=start,
        end_time=end,
        region=region,
        band=GOESConfig.PRIMARY_BAND,
        verbose=True
    )
    
    # Convert to particle sequences
    particles_seq = timeseries_to_particle_sequence(
        datasets, 
        n_particles=n_particles,
        method='weighted_sample'
    )
    
    result = {
        'particles': particles_seq,
        'times': [ds['time'] for ds in datasets],
        'raw_fields': [ds['data'] for ds in datasets],
        'config': {
            'region': region,
            'band': GOESConfig.PRIMARY_BAND,
            'n_particles': n_particles,
            'start': start.isoformat(),
            'end': end.isoformat()
        }
    }
    
    print(f"\nResult shape: {particles_seq.shape}")
    print(f"  T = {particles_seq.shape[0]} time steps")
    print(f"  N = {particles_seq.shape[1]} particles")
    
    return result


# =============================================================================
# SAVE/LOAD FOR LOT PIPELINE
# =============================================================================
def save_for_lot_pipeline(result: dict, output_dir: str = './goes_data'):
    """
    Save downloaded data in format compatible with LOT+RC pipeline.
    
    Creates files matching your synthetic benchmark structure:
    - particles.npy: (T, N, 2) particle positions
    - times.npy: (T,) timestamps
    - config.json: metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save particles
    np.save(output_dir / 'particles.npy', result['particles'])
    
    # Save times as floats (hours since start)
    t0 = result['times'][0]
    times_hours = np.array([(t - t0).total_seconds() / 3600 for t in result['times']])
    np.save(output_dir / 'times.npy', times_hours)
    
    # Save config
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(result['config'], f, indent=2)
    
    print(f"Saved to {output_dir}:")
    print(f"  - particles.npy: {result['particles'].shape}")
    print(f"  - times.npy: {len(times_hours)} timestamps")
    print(f"  - config.json: metadata")


def fetch_goes_particles_bundle(
    output_dir: Path | str,
    *,
    hours: int,
    n_particles: int,
    date: str,
    hour_utc: int,
    verbose: bool = True,
) -> Path:
    """
    Download GOES-16 MCMIPC, convert to particle trajectories, save ``particles.npy``
    (and ``times.npy``, ``config.json``) under ``output_dir``.

    Returns path to ``particles.npy``. Requires network and ``goes2go``.
    """
    if not GOES2GO_AVAILABLE:
        raise ImportError("goes2go is required. Install: pip install -r requirements-realworld.txt")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    region = GOESConfig.REGION
    band = GOESConfig.PRIMARY_BAND
    start = datetime.strptime(f"{date} {hour_utc:02d}:00", "%Y-%m-%d %H:%M")
    end = start + timedelta(hours=hours)

    if verbose:
        print(f"GOES-16 MCMIPC: {start} → {end} (~{hours * 6} frames @ 10 min)")
        print(f"Region {region}, band {band}, N={n_particles}")

    datasets = download_goes_timeseries(
        start_time=start,
        end_time=end,
        region=region,
        band=band,
        verbose=verbose,
    )
    if len(datasets) < 3:
        raise RuntimeError(
            f"Too few GOES frames ({len(datasets)}). Try another --date / --hour-utc or check network."
        )

    rng = np.random.default_rng(42)
    particles_seq = timeseries_to_particles_safe(datasets, n_particles, rng)
    result = {
        "particles": particles_seq,
        "times": [ds["time"] for ds in datasets],
        "raw_fields": [ds["data"] for ds in datasets],
        "config": {
            "region": region,
            "band": band,
            "n_particles": n_particles,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    }
    save_for_lot_pipeline(result, str(output_dir))
    out_path = output_dir / "particles.npy"
    if verbose:
        print(f"\n[OK] GOES data ready: {out_path}")
    return out_path


def load_goes_particles(data_dir: str = './goes_data') -> np.ndarray:
    """Load particle data for LOT pipeline."""
    data_dir = Path(data_dir)
    return np.load(data_dir / 'particles.npy')


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("GOES-16 Cloud Data for LOT+RC Pipeline")
    print("=" * 50)
    print("\nUsage options:")
    print("  1. quick_test_single_image()  - Test single download")
    print("  2. download_diurnal_cycle()   - Get 24-hour sequence")
    print("\nExample:")
    print("  >>> from goes_cloud_data_acquisition import *")
    print("  >>> quick_test_single_image()")
    print("  >>> result = download_diurnal_cycle('2024-06-15')")
    print("  >>> save_for_lot_pipeline(result)")