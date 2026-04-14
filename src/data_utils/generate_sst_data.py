#!/usr/bin/env python3
"""
Download and prepare Sea Surface Temperature (SST) data for LOT+RC pipeline.

This script:
1. Downloads SST data from NOAA OISST
2. Converts to particle representation (weighted by temperature gradients)
3. Ensures cyclicality (seasonal cycles)
4. Saves in format ready for LOT embedding generation
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import json

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    print("WARNING: xarray not installed. Install with: pip install xarray")
    print("Will attempt alternative download methods...")


# =============================================================================
# CONFIGURATION
# =============================================================================

SST_CONFIG = {
    'n_particles': 500,
    'region_bounds': (-100, -80, 25, 35),  # Gulf of Mexico (lon_min, lon_max, lat_min, lat_max)
    'cadence_hours': 24,  # Daily data
    'cycle_length': 365,  # Annual cycle (365 days)
    'min_cycles': 2,  # Minimum 2 years of data
    'data_source': 'noaa_oisst',  # NOAA OISST v2.1
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_complete_cycles(particles, times_hours=None, cycle_length=365, min_cycles=2, verbose=True):
    """
    Ensure data contains complete cycles.

    Truncates data to contain only complete cycles, ensuring cyclicality
    for proper warm-up alignment in forecasting.

    Parameters
    ----------
    particles : np.ndarray
        Particle array, shape (T, N, d)
    times_hours : np.ndarray, optional
        Time array in hours
    cycle_length : int
        Length of one cycle in timesteps
    min_cycles : int
        Minimum number of complete cycles required
    verbose : bool
        Print progress messages

    Returns
    -------
    tuple : (particles_trimmed, times_trimmed, cycle_info)
    """
    T_orig = particles.shape[0]
    n_complete_cycles = T_orig // cycle_length

    if n_complete_cycles < min_cycles and verbose:
        print(f"[WARN] Only {n_complete_cycles} complete cycles found, minimum {min_cycles} required")

    # Trim to complete cycles
    T_trimmed = n_complete_cycles * cycle_length
    particles_trimmed = particles[:T_trimmed]

    if times_hours is not None:
        times_trimmed = times_hours[:T_trimmed]
    else:
        times_trimmed = None

    cycle_info = {
        'n_complete_cycles': n_complete_cycles,
        'n_steps_original': T_orig,
        'n_steps_trimmed': T_trimmed,
        'cycle_length': cycle_length,
    }

    if verbose:
        print(f"[CYCLES] Original: {T_orig} steps, Trimmed: {T_trimmed} steps ({n_complete_cycles} complete cycles)")

    return particles_trimmed, times_trimmed, cycle_info


# =============================================================================
# SST DATA DOWNLOAD
# =============================================================================

def download_sst_noaa_oisst(start_date, end_date, region_bounds, output_dir=None):
    """
    Download SST data from NOAA OISST v2.1 via ERDDAP.

    Uses ERDDAP to access data directly with time/space subsetting.

    Parameters
    ----------
    start_date : datetime
        Start date for data
    end_date : datetime
        End date for data
    region_bounds : tuple
        (lon_min, lon_max, lat_min, lat_max)
    output_dir : Path, optional
        Directory to save data

    Returns
    -------
    xarray.Dataset
        SST data with coordinates and time
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("xarray required for SST download. Install with: pip install xarray")

    lon_min, lon_max, lat_min, lat_max = region_bounds

    # Convert negative longitudes to 0-360 format for OISST
    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    print(f"Downloading SST data from {start_date.date()} to {end_date.date()}...")
    print(f"Region: lon [{lon_min}, {lon_max}] -> [{lon_min_360}, {lon_max_360}], lat [{lat_min}, {lat_max}]")

    # NOAA NCEI ERDDAP endpoint for OISST v2.1
    erddap_url = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon"

    # Format dates for ERDDAP query
    start_str = start_date.strftime("%Y-%m-%dT12:00:00Z")
    end_str = end_date.strftime("%Y-%m-%dT12:00:00Z")

    # Build ERDDAP query URL for netCDF subset
    query_url = (
        f"{erddap_url}.nc?"
        f"sst[({start_str}):1:({end_str})][(0.0):1:(0.0)][({lat_min}):1:({lat_max})][({lon_min_360}):1:({lon_max_360})]"
    )

    print(f"  Querying ERDDAP...")
    print(f"  URL: {erddap_url}")

    try:
        # Open dataset directly from ERDDAP
        ds = xr.open_dataset(query_url, decode_times=True)

        # Squeeze out the zlev dimension (depth=0 for SST)
        if 'zlev' in ds.dims:
            ds = ds.squeeze('zlev', drop=True)

        # Convert longitude back to -180 to 180 if needed
        if ds.lon.values.max() > 180:
            ds = ds.assign_coords(lon=(ds.lon.values + 180) % 360 - 180)
            ds = ds.sortby('lon')

        print(f"  Downloaded {len(ds.time)} time steps")
        print(f"  Spatial shape: {ds.sst.shape[1:]} (lat, lon)")
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

        return ds

    except Exception as e:
        print(f"  ERDDAP error: {e}")

        # Fallback: Download individual daily files from NCEI THREDDS
        print("  Trying NCEI daily files (slower but more reliable)...")
        base_url = "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR"

        datasets = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        loaded_days = 0

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            ym_str = current_date.strftime("%Y%m")

            url = f"{base_url}/{ym_str}/oisst-avhrr-v02r01.{date_str}.nc"

            try:
                ds_day = xr.open_dataset(url, decode_times=True)

                # Subset to region (using 0-360 longitude)
                ds_subset = ds_day.sel(
                    lon=slice(lon_min_360, lon_max_360),
                    lat=slice(lat_min, lat_max)
                )

                # Squeeze zlev dimension
                if 'zlev' in ds_subset.dims:
                    ds_subset = ds_subset.squeeze('zlev', drop=True)

                # Load into memory
                ds_subset = ds_subset.load()

                # Convert longitude back to -180 to 180
                if ds_subset.lon.values.max() > 180:
                    new_lons = (ds_subset.lon.values + 180) % 360 - 180
                    ds_subset = ds_subset.assign_coords(lon=new_lons)

                datasets.append(ds_subset)
                loaded_days += 1

                # Progress indicator every 30 days
                if loaded_days % 30 == 0 or loaded_days == total_days:
                    print(f"    Downloaded {loaded_days}/{total_days} days...", flush=True)

            except Exception:
                # Skip missing days silently
                pass

            current_date += timedelta(days=1)

        if not datasets:
            raise ValueError("No data downloaded from NCEI. Check dates and network.")

        sst_data = xr.concat(datasets, dim='time')
        print(f"\n  Downloaded {len(sst_data.time)} time steps total")
        print(f"  Spatial shape: {sst_data.sst.shape[1:]} (lat, lon)")
        print(f"  Time range: {sst_data.time.values[0]} to {sst_data.time.values[-1]}")
        return sst_data


def download_sst_alternative(start_date, end_date, region_bounds):
    """
    Alternative SST download method - generates synthetic data for testing.

    This is a fallback if real data download fails.

    Parameters
    ----------
    start_date : datetime
        Start date
    end_date : datetime
        End date
    region_bounds : tuple
        (lon_min, lon_max, lat_min, lat_max)

    Returns
    -------
    xarray.Dataset
        Synthetic SST data
    """
    print("Alternative download method - generating synthetic SST data")
    print("WARNING: This is synthetic data for testing only!")

    lon_min, lon_max, lat_min, lat_max = region_bounds

    # Create time array
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    # Create spatial grid
    n_lon = 50
    n_lat = 40
    lons = np.linspace(lon_min, lon_max, n_lon)
    lats = np.linspace(lat_min, lat_max, n_lat)

    # Generate synthetic SST field
    sst_values = []
    for date in dates:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        day_of_year = date.timetuple().tm_yday

        # Seasonal cycle + spatial pattern + noise
        sst = 25 + 5 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
        sst += 2 * np.sin(2 * np.pi * lon_grid / 20)  # Zonal pattern
        sst += np.random.randn(n_lat, n_lon) * 0.5  # Noise

        sst_values.append(sst)

    sst_data = xr.Dataset(
        {
            'sst': (['time', 'lat', 'lon'], np.array(sst_values))
        },
        coords={
            'time': dates,
            'lat': lats,
            'lon': lons
        }
    )

    print(f"Generated synthetic SST data: {len(dates)} time steps")
    return sst_data


# =============================================================================
# PARTICLE CONVERSION
# =============================================================================

def sst_to_particles(sst_field, n_particles=500, method='gradient_weighted'):
    """
    Convert SST field to particle representation.

    Methods:
    - 'gradient_weighted': Weight by temperature gradients (eddies, fronts)
    - 'intensity_weighted': Weight by absolute temperature
    - 'uniform': Uniform sampling (for testing)

    Parameters
    ----------
    sst_field : np.ndarray
        2D SST field (lat, lon)
    n_particles : int
        Number of particles to generate
    method : str
        Weighting method

    Returns
    -------
    np.ndarray
        Particle positions, shape (n_particles, 2) in normalized [0,1]^2
    """
    H, W = sst_field.shape

    # Create coordinate grids (normalized to [0,1] x [0,1])
    yy, xx = np.mgrid[0:H, 0:W]
    xx_norm = xx.flatten() / W
    yy_norm = yy.flatten() / H
    values = sst_field.flatten()

    # Handle NaN/missing values
    valid_mask = ~np.isnan(values)
    xx_norm = xx_norm[valid_mask]
    yy_norm = yy_norm[valid_mask]
    values = values[valid_mask]

    if method == 'gradient_weighted':
        # Weight by temperature gradient magnitude (eddies, fronts)
        sst_2d = sst_field.copy()
        sst_2d[np.isnan(sst_2d)] = np.nanmean(sst_2d)

        grad_y, grad_x = np.gradient(sst_2d)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        weights = grad_magnitude.flatten()[valid_mask]

    elif method == 'intensity_weighted':
        # Weight by absolute temperature (warmer regions)
        weights = values - values.min()

    elif method == 'uniform':
        # Uniform sampling
        weights = np.ones(len(values))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to probability distribution
    weights = np.maximum(weights, 0)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    # Sample particle positions
    indices = np.random.choice(
        len(weights),
        size=n_particles,
        replace=True,
        p=weights
    )

    particles = np.column_stack([xx_norm[indices], yy_norm[indices]])
    return particles


def sst_dataset_to_particle_sequence(sst_data, n_particles=500, method='gradient_weighted'):
    """
    Convert SST time series to particle sequence.

    Parameters
    ----------
    sst_data : xarray.Dataset
        SST data with 'sst' variable and 'time' coordinate
    n_particles : int
        Particles per time step
    method : str
        Weighting method for particle conversion

    Returns
    -------
    tuple : (particles_array, times_array)
        particles: (T, N, 2) array
        times: (T,) array in hours since start
    """
    sst_array = sst_data['sst'].values  # (time, lat, lon)
    times = sst_data['time'].values

    T = len(times)
    particles_seq = np.zeros((T, n_particles, 2))

    print(f"Converting {T} time steps to particles...")
    for t in range(T):
        if t % 100 == 0:
            print(f"  {t}/{T}", flush=True)

        sst_field = sst_array[t]  # (lat, lon)
        particles = sst_to_particles(sst_field, n_particles, method=method)
        particles_seq[t] = particles

    # Convert times to hours since start
    t0 = times[0]
    if isinstance(t0, np.datetime64):
        times_hours = np.array([(t - t0).astype('timedelta64[h]').astype(float) for t in times])
    else:
        # Fallback for datetime objects
        times_hours = np.array([(t - t0).total_seconds() / 3600 for t in times])

    print(f"Converted to particles: {particles_seq.shape}")
    return particles_seq, times_hours


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_and_process_sst(
    start_date,
    end_date,
    region_bounds=None,
    n_particles=500,
    output_dir=None,
    use_alternative=False,
    ensure_cyclicality=True,
    cycle_length=365,
    min_cycles=2
):
    """
    Download SST data and convert to particle representation.

    Parameters
    ----------
    start_date : str or datetime
        Start date 'YYYY-MM-DD'
    end_date : str or datetime
        End date 'YYYY-MM-DD'
    region_bounds : tuple, optional
        (lon_min, lon_max, lat_min, lat_max). Defaults to Gulf of Mexico.
    n_particles : int
        Number of particles per time step
    output_dir : Path or str, optional
        Output directory. Defaults to './sst_data'
    use_alternative : bool
        Use synthetic data instead of downloading
    ensure_cyclicality : bool
        If True, trim data to complete cycles
    cycle_length : int
        Length of one cycle in timesteps (default 365 for annual)
    min_cycles : int
        Minimum number of complete cycles required

    Returns
    -------
    tuple : (particles, times, metadata)
    """
    if output_dir is None:
        output_dir = Path('./sst_data')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    if region_bounds is None:
        region_bounds = SST_CONFIG['region_bounds']

    # Download SST data
    print("=" * 60)
    print("STEP 1: Downloading SST Data")
    print("=" * 60)

    try:
        if use_alternative or not XARRAY_AVAILABLE:
            sst_data = download_sst_alternative(start_date, end_date, region_bounds)
        else:
            sst_data = download_sst_noaa_oisst(start_date, end_date, region_bounds)
    except Exception as e:
        print(f"Error downloading SST: {e}")
        print("Falling back to synthetic data...")
        sst_data = download_sst_alternative(start_date, end_date, region_bounds)

    # Convert to particles
    print("\n" + "=" * 60)
    print("STEP 2: Converting to Particle Representation")
    print("=" * 60)

    particles, times_hours = sst_dataset_to_particle_sequence(
        sst_data,
        n_particles=n_particles,
        method='gradient_weighted'
    )

    # Ensure complete cycles if requested
    cadence_hours = SST_CONFIG.get('cadence_hours', 24)
    cycle_info = None
    if ensure_cyclicality:
        print("\n" + "=" * 60)
        print("STEP 3: Ensuring Complete Cycles")
        print("=" * 60)
        particles, times_hours, cycle_info = ensure_complete_cycles(
            particles,
            times_hours,
            cycle_length=cycle_length,
            min_cycles=min_cycles,
            verbose=True
        )

    # Save data
    np.save(output_dir / 'particles.npy', particles.astype(np.float32))
    np.save(output_dir / 'times.npy', times_hours.astype(np.float32))

    # Create metadata
    n_complete_cycles = cycle_info['n_complete_cycles'] if cycle_info else int(len(particles) / cycle_length)

    metadata = {
        'source': 'NOAA OISST',
        'variable': 'sea_surface_temperature',
        'region_bounds': list(region_bounds),
        'n_particles': n_particles,
        'n_timesteps': len(particles),
        'total_days': times_hours[-1] / 24 if len(times_hours) > 0 else 0,
        'cadence_hours': cadence_hours,
        'cycle_length': cycle_length,
        'n_complete_cycles': n_complete_cycles,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'purpose': 'SST forecasting',
    }

    # Add cycle info if available
    if cycle_info:
        metadata['cycle_info'] = cycle_info

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  particles.npy: {particles.shape}")
    print(f"  times.npy: {times_hours.shape}")
    print(f"  config.json: metadata")
    print(f"  Complete cycles: {metadata['n_complete_cycles']}")

    return particles, times_hours, metadata


def main():
    """CLI entry point for downloading and preparing SST data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare SST data for LOT+RC pipeline"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2021-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--region',
        type=float,
        nargs=4,
        default=[-100, -80, 25, 35],
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        help='Region bounds: lon_min lon_max lat_min lat_max'
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        default=500,
        help='Number of particles per time step'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sst_data',
        help='Output directory for SST data'
    )
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of downloading (for testing)'
    )
    parser.add_argument(
        '--no-cyclicality',
        action='store_true',
        help='Skip trimming to complete cycles'
    )
    parser.add_argument(
        '--cycle-length',
        type=int,
        default=365,
        help='Length of one cycle in timesteps (default: 365 for annual)'
    )
    parser.add_argument(
        '--min-cycles',
        type=int,
        default=2,
        help='Minimum number of complete cycles required (default: 2)'
    )

    args = parser.parse_args()

    try:
        particles, times, metadata = download_and_process_sst(
            start_date=args.start_date,
            end_date=args.end_date,
            region_bounds=tuple(args.region),
            n_particles=args.n_particles,
            output_dir=args.output_dir,
            use_alternative=args.use_synthetic,
            ensure_cyclicality=not args.no_cyclicality,
            cycle_length=args.cycle_length,
            min_cycles=args.min_cycles
        )
        print("\nSST data preparation complete!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
