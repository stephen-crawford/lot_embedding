# GOES Cloud Data Experiments: Velocity vs Positions LOT-RC Forecasting

## Overview

This document summarizes all experiments comparing **velocity-based** and
**positions-based** LOT-RC forecasting on real-world satellite cloud and
ocean data. Every trajectory comes from actual observations — no synthetic
data is used.

**Bottom line:** With the right LOT embedding configuration, velocity-based
forecasting beats positions on 13 out of 15 hyperparameter settings (87%)
and achieves up to +58% improvement in Sinkhorn divergence on GOES-16
Great Plains cloud data.

---

## Datasets

| Dataset | Source | Duration | T | N | Cadence | Physical regime |
|---------|--------|----------|---|---|---------|-----------------|
| GOES GP 72h | GOES-16 CMI_C13 IR | 72 hours | 145 | 200 | 30 min | Continental diurnal convection |
| GOES Gulf 48h | GOES-16 CMI_C13 IR | 48 hours | 75 | 200 | 30 min | Tropical maritime convection |
| SST 180d | NOAA OISST | 180 days | 165 | 200 | daily | Seasonal ocean warming |
| SST 365d | NOAA OISST | 365 days | 366 | 200 | daily | Full annual SST cycle |

---

## Test Files

### 1. `tests/test_realworld_extended_forecasting.py`

**Purpose:** Baseline comparison across all four real datasets with 8
hyperparameter configs each.

**Method:** Calls `run_pipeline()` which generates LOT embeddings once,
then runs both velocity and positions forecasting with their respective
default hyperparameters (velocity: sr=0.8, lk=0.8, ridge=1e-2; positions:
sr=0.9, lk=1.0, ridge=1e-6).

**Key results:**

| Dataset | Velocity wins | Best improvement |
|---------|:---:|---:|
| GOES GP 72h | 5/8 (62%) | +32.2% |
| GOES Gulf 48h | 0/8 (0%) | -29.1% |
| SST 180d | 0/8 (0%) | -11.3% |
| SST 365d | 0/8 (0%) | -18.1% |

**Why only GOES GP wins:** The Great Plains data has strong diurnal
convective cycling — cloud fields expand during afternoon heating and
contract overnight, creating quasi-periodic velocity fields. The Gulf
has stochastic tropical convection without clean periodicity. SST is
trend-dominated (seasonal warming/cooling), not oscillatory.

**Plots:** `plots/realworld_extended/` (42 files) — error curves,
crossover analysis, HP sensitivity, and summary bars for each dataset.

---

### 2. `tests/test_interpolated_cloud_forecasting.py`

**Purpose:** Test whether Wasserstein geodesic interpolation between
consecutive measures and time reparameterization improve velocity
forecasting.

**Preprocessing methods tested:**
- **raw**: Original 30-min cadence trajectory
- **interp_2x**: 1 OT-interpolated frame between each pair (T: 145 -> 289)
- **interp_4x**: 3 interpolated frames (T: 145 -> 577)
- **const_vel**: Arc-length reparameterization for constant velocity magnitude
- **const_accel**: Reparameterization for constant acceleration magnitude
- **interp2x_constvel**: Interpolation + constant velocity
- **interp2x_constaccel**: Interpolation + constant acceleration

**Wasserstein interpolation method:** For each pair (mu_t, mu_{t+1}),
solve OT(mu_t, mu_{t+1}) to get the transport map T, then create
intermediate measures mu_s = (1-s)*mu_t + s*T(mu_t) for s in (0,1).
This produces physically meaningful intermediate cloud states along
the Wasserstein geodesic.

**Key results (gaussian_iso / fixed assignment):**

| Preprocessing | L2 improvement | Sinkhorn improvement | Velocity wins? |
|---|---:|---:|:---:|
| raw | +0.7% | -19.0% | L2 only |
| interp_2x | +9.9% | **+48.5%** | Both |
| const_accel | +17.9% | **+63.2%** | Both |
| interp2x_constvel | +8.2% | +7.4% | Both |

**Key results (gaussian_iso / per_frame assignment):**

| Preprocessing | L2 improvement | Sinkhorn improvement | Velocity wins? |
|---|---:|---:|:---:|
| raw | **+32.2%** | **+52.7%** | Both |
| const_vel | +12.6% | -0.6% | L2 only |
| interp2x_constvel | +9.5% | **+36.3%** | Both |

**Why interpolation helps:** Denser time sampling makes the velocity
field smoother and more learnable. The 2x interpolation is the sweet
spot — 4x+ creates too-long forecast horizons where both methods
degrade.

**Why const_accel works:** Redistributing time points so acceleration
is uniform makes the velocity field even more stationary, which is
exactly what the velocity RC leverages.

**Plots:** `plots/interpolated_cloud/` (39 files) — interpolation
verification, velocity profiles, preprocessing comparisons, dual-metric
error curves, tau sweep trends.

---

### 3. `tests/test_velocity_dominance_sweep.py`

**Purpose:** Find hyperparameter settings where velocity **always** beats
positions, following the paper's exact mathematical pipeline.

**Critical design:** Both methods get a fair comparison:
- `hyperparameter_sweep=True` lets each method auto-select its best
  (sr, lk, ridge) from a 48-config grid by minimizing map RMSE
- Both share the **same LOT embeddings** (same reference, same OT plan)
- Both use the **same warm-up / forecast split**
- Evaluated with L2(sigma) RMSE AND Sinkhorn divergence

#### Test A: Theory-motivated configs (HP sweep per method)

Each method gets to pick its own best HPs from a 4x4x3 grid.

| Config | Map RMSE impr. | L2 impr. | Sinkhorn impr. | All 3 win? |
|--------|---:|---:|---:|:---:|
| gauss_iso / per_frame / cl=48 | +51.9% | +35.7% | +50.6% | Yes |
| gauss_iso / per_frame / cl=48 / big reservoir | +47.6% | +31.7% | +58.3% | Yes |
| snap_begin / fixed / cl=48 | -22.2% | -21.5% | +53.5% | Sinkhorn only |
| gauss_iso / per_frame / cl=36 | -88.0% | -86.2% | -159.8% | No |
| gauss_iso / per_frame / cl=24 | -96.5% | -95.1% | -143.7% | No |
| snap_begin / per_frame / cl=48 | -0.8% | -14.3% | -67.2% | No |

**Finding:** `gaussian_iso + per_frame + cycle_length=48` is the winning
recipe. Cycle length is critical — it must match the actual diurnal period
(48 steps = 24 hours at 30-min cadence). Shorter cycles leave insufficient
warm-up; longer cycles waste data.

#### Test B: Shared hyperparameters (identical HPs for both methods)

This isolates the pure paradigm effect. Same (sr, lk, ridge) for both
velocity and positions, with gaussian_iso / per_frame / cl=48.

| Hyperparameters | Velocity wins all 3? | Sinkhorn impr. |
|---|:---:|---:|
| sr=0.7, lk=0.5, ridge=1e-3 | Yes | +49.0% |
| sr=0.7, lk=0.7, ridge=1e-2 | Yes | +54.4% |
| sr=0.7, lk=0.9, ridge=1e-2 | Yes | +53.8% |
| sr=0.8, lk=0.5, ridge=1e-3 | Yes | +41.7% |
| sr=0.8, lk=0.7, ridge=1e-2 | Yes | +50.8% |
| sr=0.8, lk=0.9, ridge=1e-2 | Yes | +55.3% |
| sr=0.8, lk=0.9, ridge=1e-1 | **No** | -199.9% |
| sr=0.9, lk=0.3, ridge=1e-3 | Yes | +42.1% |
| sr=0.9, lk=0.5, ridge=1e-2 | Yes | +40.4% |
| sr=0.9, lk=0.7, ridge=1e-2 | Yes | +46.1% |
| sr=0.9, lk=0.9, ridge=1e-2 | Yes | **+57.9%** |
| sr=0.9, lk=0.9, ridge=1e-1 | **No** | -210.1% |
| sr=0.95, lk=0.5, ridge=1e-2 | Yes | +34.5% |
| sr=0.95, lk=0.7, ridge=1e-2 | Yes | +44.4% |
| sr=0.95, lk=0.9, ridge=1e-2 | Yes | **+57.9%** |

**Result: 13/15 configs (87%) — velocity wins on ALL three metrics.**

The only failures have ridge=0.1 (heavy regularization). With ridge
<= 0.01, velocity wins for every spectral radius (0.7-0.95) and every
leak rate (0.3-0.9) tested.

**Plots:** `plots/velocity_dominance/` (27 files) — summary bars,
error time series, all-metric winner analysis.

---

## Why Velocity Wins (and When It Doesn't)

### The paper's argument, validated on real data

1. **LOT embedding lifts measures to Hilbert space** (paper Sec. 2):
   The OT map u_t = T_sigma^{mu_t} embeds each cloud snapshot into
   L^2(sigma), where vector operations (addition, subtraction) are
   well-defined.

2. **Velocity is more stationary than position** (paper Sec. 3):
   The velocity Delta_t = u_{t+1} - u_t captures the *change* in cloud
   configuration. For diurnal convective cycling, this change repeats
   every ~24 hours: clouds expand in the afternoon, contract at night.
   The velocity RC only needs to learn a time-invariant v_t -> v_{t+1}
   map.

3. **Linear vs exponential error growth** (paper Sec. 5):
   Velocity integrates: u_hat_{t+1} = u_hat_t + Delta_hat_t. Each
   velocity prediction error adds to the position, giving O(T) error
   growth. Positions predicts maps directly, but feeds predictions back
   as inputs, compounding errors exponentially.

4. **Per_frame OT factors out rotation** (paper Sec. 2):
   Solving OT(sigma, mu_t) independently per frame with a rotationally
   symmetric reference (isotropic Gaussian) removes global rotation
   from the LOT maps, making the velocity field even more stationary.

### Critical requirements (from experiments)

| Requirement | Why | Evidence |
|---|---|---|
| Per_frame OT assignment | Factors out rotation, makes velocity stationary | Per_frame: +32.2%; fixed: +0.7% |
| Isotropic Gaussian reference | Rotationally symmetric -> rotation factored out | gaussian_iso: +32.2%; snapshot_begin: +15.1% |
| Cycle length = diurnal period | Warm-up must cover one complete cycle | cl=48: +51.9%; cl=24: -96.5% |
| Ridge <= 0.01 | RC needs enough expressiveness for velocity dynamics | ridge=0.01: 13/13 wins; ridge=0.1: 0/2 wins |
| Quasi-periodic dynamics | Velocity stationarity requires periodicity | GOES GP: +32.2%; SST: -11.3% |

### When velocity loses

- **Non-periodic data** (SST): Sea surface temperature evolves by
  seasonal trends, not oscillations. The velocity field is non-stationary,
  so the v_t -> v_{t+1} mapping changes over time and the RC can't
  generalize.

- **Too-short time series** (GOES Gulf 48h, T=75): Not enough data for
  the RC to learn velocity dynamics with sufficient warm-up.

- **Over-regularized RC** (ridge=0.1): The readout W_out is too
  constrained to capture velocity transitions. Positions degrades more
  gracefully under over-regularization because it directly predicts
  maps rather than integrating velocity errors.

- **Wrong cycle length**: If cycle_length doesn't match the actual
  dynamics period, the warm-up either covers incomplete cycles (RC
  hasn't seen the full dynamic range) or is too long (leaving too
  few forecast steps for meaningful evaluation).

---

## Paper Figures

All figures are in `plots/paper/` in both PDF and PNG format.

| Figure | Content | File |
|--------|---------|------|
| Fig 1 | Cross-dataset comparison: where velocity wins vs loses | `fig1_cross_dataset.pdf` |
| Fig 2 | HP robustness: 13/15 shared-HP configs win | `fig2_hp_robustness.pdf` |
| Fig 3 | Per-step error curves for best config | `fig3_error_curves.pdf` |
| Fig 4 | Preprocessing effect: interpolation and reparameterization | `fig4_preprocessing_effect.pdf` |
| Fig 5 | Snapshot vs geometric LOT reference comparison | `fig5_snapshot_vs_geometric.pdf` |
| Fig 6 | Crossover analysis: when velocity overtakes positions | `fig6_crossover.pdf` |

### Figure descriptions for paper text

**Fig 1:** Velocity improvement and win rate across four real-world
datasets. Velocity achieves +32.2% improvement on GOES Great Plains
(diurnal convective cycling) but underperforms on Gulf of Mexico
(stochastic tropical convection) and SST data (trend-dominated).

**Fig 2:** With identical hyperparameters for both methods and per-frame
Gaussian LOT embedding, velocity wins on all three metrics (map RMSE,
L^2(sigma), Sinkhorn divergence) for 13 of 15 HP configurations. The
only failures occur with ridge=0.1.

**Fig 3:** Per-step and cumulative L^2(sigma) forecast error over a
48-hour horizon (GOES GP 72h, sr=0.9, lk=0.9, ridge=0.01). Velocity
error stays consistently below positions, with +37% cumulative RMSE
improvement. Blue shading indicates timesteps where velocity is better.

**Fig 4:** Effect of trajectory preprocessing on velocity improvement.
Constant-acceleration reparameterization provides the largest Sinkhorn
improvement (+63.2%) by making the velocity field more uniform. 2x
Wasserstein interpolation provides +48.5% Sinkhorn improvement by
increasing trajectory density.

**Fig 5:** Comparison of geometric (Gaussian, uniform) vs data-driven
(snapshot) LOT reference measures. Per_frame assignment is the dominant
factor. On raw data, Gaussian/per_frame achieves +32.2%; on
constant-velocity data, Snapshot-begin/per_frame achieves +48.4%.

**Fig 6:** Per-step error comparison showing the crossover dynamics
over the forecast horizon. Velocity maintains its advantage
throughout most of the 48-hour forecast window, with the positions
method's error growing faster due to compounding.

---

## Reproducibility

All tests can be re-run from the repository root:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-realworld.txt

# Run individual test suites
pytest tests/test_realworld_extended_forecasting.py -xvs
pytest tests/test_interpolated_cloud_forecasting.py -xvs
pytest tests/test_velocity_dominance_sweep.py -xvs

# Regenerate paper figures
python scripts/generate_paper_figures.py
```

Required data directories:
- `goes_data_gp_summer_72h/particles.npy` (T=145, N=200)
- `goes_data_gulf_48h/particles.npy` (T=75, N=200)
- `sst_data_180d/particles.npy` (T=165, N=200)
- `sst_data_365d/particles.npy` (T=366, N=200)
