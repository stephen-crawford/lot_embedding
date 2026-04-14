# LOT-RC Forecasting

## Project

Reservoir Computing on LOT-Embedded Measure-Valued Dynamical Systems.
Paper: `paper.tex`. Pipeline code: `src/`, `scripts/`.

## Critical: Read Before Making Changes

**Always read `PAPER_PIPELINE_LOCK.md` before modifying any pipeline code.**
It maps every paper equation to its code implementation and lists known discrepancies.

## Pipeline Architecture

```
satellite field (H x W brightness temp)
  -> particle extraction (cloud_field_to_particles, adaptive percentile)
  -> trajectory (T, N, 2) in lon/lat or normalized coords
  -> LOT embedding (generate_lot_embeddings.py, OT + barycentric projection)
     -> lot_maps.npy (T, R, d): transport map values u_t(x_i) at reference points
     -> velocities.npy (T-1, R, d): Delta_t = u_{t+1} - u_t
     -> reference.npy (R, d): sigma points
  -> RC forecast (velocity_forecast.py or positions_forecast.py)
     -> predicted_measures_X.npy (H, R, d): forecast particle positions
  -> evaluation (Sinkhorn W2, LOT-RMSE)
```

## Key Invariants

1. **pred_measures_X = pushforward particles**: These ARE the predicted particle
   positions, computed as u_hat(x_i) for each reference point x_i. They are NOT
   raw particle positions from the original trajectory — they have R points (reference
   count), not N points (original particle count).

2. **LOT maps are OT map values**: `lot_maps[t]` contains T_sigma^{mu_t}(x_i) for
   each reference point x_i. The transport map sends reference -> target, so the map
   values ARE the target particle positions (as seen by the reference).

3. **Velocity = map difference**: `velocities[t] = lot_maps[t+1] - lot_maps[t]`.
   This is the LOT velocity Delta_t in the paper.

4. **Coordinate systems matter**:
   - GOES satellite images use geostationary projection (x/y in radians)
   - Particles can be in pixel space, lon/lat, or normalized coordinates
   - The LOT pipeline works in whatever coordinate space the particles are in
   - Visualization must use the SAME coordinate system as the satellite image
   - For pixel-space overlay: extract particles as (col, row), plot with imshow origin='upper'
   - NEVER use imshow(extent=[lon_min, lon_max, ...]) — the geostationary projection is non-linear

5. **Fixed vs per_frame OT assignment**:
   - `fixed`: solve OT(sigma -> mu_0) once, reuse coupling for all frames. Preserves temporal coherence.
   - `per_frame`: solve OT(sigma -> mu_t) independently each frame. Theoretically correct LOT embedding but can have permutation jumps between frames.

## Paper-Code Mapping (Critical Parameters)

| Paper Section | Parameter | Paper Value | Code Location |
|---|---|---|---|
| Section 5.2 | spectral_radius | 0.7 | rc_computer.py ReservoirConfig |
| Section 5.2 | input_scaling | 0.1 | rc_computer.py ReservoirConfig |
| Section 5.2 | leak_rate | 0.7 | rc_computer.py ReservoirConfig |
| Section 5.2 | ridge (LOT) | 1e-6 | velocity_forecast.py |
| Section 5.2 | ridge (RAW) | 1e-4 | positions_forecast.py |
| Section 5.2 | sparsity | 0.1 (10%) | rc_computer.py |
| Section 5.2 | activation | tanh | rc_computer.py |
| Section 5.1 | OT regularization | 0.01 (Sinkhorn) | realworld_experiment_pipeline.py |
| Section 5.3 | velocity paradigm | v_t -> v_{t+1} | velocity_forecast.py line 1053 |

## GOES Satellite Data

- Band CMI_C13 (10.3um IR): clouds are COLD (low brightness temp), surface is WARM
- Particle extraction: adaptive percentile weighting (coldest 30% = cloud)
- Geostationary projection: x/y are scan angles in radians, NOT lon/lat
- Use pixel-space for visualization, lon/lat only for axis labels
- Night vs day: surface cools at night, reducing cloud/surface contrast.
  Fixed thresholds (e.g. 275K) fail at night. Use adaptive percentile.

## Common Pitfalls

- **Normalizing to [0,1]^2 kills velocity advantage**: The breathing spiral needs
  raw coordinate range (radius 1-6) for velocity forecasting to outperform position.
- **Mixed lon/lat scales**: lon spans ~15 degrees, lat spans ~8 degrees. If fed raw
  into RC, the readout weights explode. Normalize independently first.
- **imshow extent with geostationary coords**: Non-linear projection means linear
  extent mapping misplaces particles. Always use pixel coordinates for overlay.
