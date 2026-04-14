# LOT-RC Forecasting

Linearized Optimal Transport (LOT) embeddings combined with Reservoir Computing for forecasting probability distributions in measure-valued dynamical systems.

## Overview

This repository implements a novel approach for forecasting evolving probability measures by:
1. **LOT Embedding**: Lifting probability measures into a Hilbert space L²(σ) via Linearized Optimal Transport
2. **Reservoir Computing**: Learning dynamics directly in the embedded space
3. **Reconstruction**: Mapping predictions back to probability measures

We compare two forecasting paradigms:
- **VELOCITY**: Predict LOT velocity fields, then integrate (recommended)
- **POSITIONS**: Predict LOT maps directly

The velocity-based approach exhibits superior error accumulation properties, particularly for oscillatory systems.

## Repository Structure
```
lot-rc-forecasting/
├── notebooks/                    # Experimental notebooks
│   ├── 01_geodesic_system.ipynb
│   ├── 02_swirling_cluster.ipynb
│   └── 03_evaluation_pipeline.ipynb
├── src/
│   ├── data_utils/               # Data generation utilities
│   │   └── simulation/
│   │       ├── measure_dynamical_systems.py  # Benchmark systems
│   │       └── generate_lot_embeddings.py    # LOT embedding generation
│   ├── run_forecast/             # Forecasting scripts
│   │   ├── velocity_forecast.py  # VELOCITY method
│   │   ├── positions_forecast.py # POSITIONS method
│   │   └── warmup_utils.py       # Cycle-aware warm-up calculation
│   ├── lotrc_eval/               # Evaluation pipeline
│   │   ├── block_a_config.py     # Configuration
│   │   ├── block_b_io.py         # Data loading
│   │   ├── block_c_llt.py        # LLT + L²(σ) cost matrix
│   │   ├── block_d_sinkhorn.py   # Sinkhorn divergence
│   │   ├── block_e_align_viz.py  # DTW alignment & visualization
│   │   ├── block_f_metrics.py    # Summary metrics
│   │   └── panel_wasserstein_video.py  # Comparison videos
│   └── rc_computer.py            # Reservoir computing implementation
├── lot_maps/                     # Generated LOT embeddings
├── forecast_output/              # Forecast results
├── eval_output/                  # Evaluation outputs
└── results/                      # Raw trajectory data
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lot-rc-forecasting.git
cd lot-rc-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy, scipy, matplotlib
- POT (Python Optimal Transport)
- jupyter

## Quick Start

### 1. Generate Trajectories
```python
from data_utils.simulation.measure_dynamical_systems import GeodesicTransportSystem, SwirlingClusterSystem

# Geodesic transport (circle → triangle morphing)
system = GeodesicTransportSystem(N=500, T=200)
trajectory = system.generate()

# Swirling cluster (breathing + rotation)
system = SwirlingClusterSystem(N=500, T=250)
trajectory = system.generate()
```

### 2. Generate LOT Embeddings
```python
from data_utils.simulation.generate_lot_embeddings import generate_lot_embeddings

generate_lot_embeddings(
    system="geodesic_transport",
    N_list=[500],
    kinds=["gaussian_iso", "uniform_square"],
    fractions=[100],
)
```

### 3. Run Forecasts
```bash
# VELOCITY method (recommended)
python -m run_forecast.velocity_forecast --system geodesic_transport --N 500 --kind gaussian_iso

# POSITIONS method
python -m run_forecast.positions_forecast --system geodesic_transport --N 500 --kind gaussian_iso
```

### 4. Evaluate & Compare
```python
from lotrc_eval import load_run_measures, build_delta_llt_l2, save_alignment_artifacts

# Load runs
vel = load_run_measures(velocity_dir, name="VELOCITY")
pos = load_run_measures(positions_dir, name="POSITIONS")

# Compute cost matrices
D_vel, _ = build_delta_llt_l2(vel)
D_pos, _ = build_delta_llt_l2(pos)

# Save alignment artifacts
save_alignment_artifacts("VELOCITY", D_vel, eval_out / "velocity", gamma_base=0.10)
save_alignment_artifacts("POSITIONS", D_pos, eval_out / "positions", gamma_base=0.10)
```

## Benchmark Systems

| System | Dynamics | Key Challenge |
|--------|----------|---------------|
| **Geodesic Transport** | Circle ↔ Triangle morphing | Shape deformation, multi-modal |
| **Swirling Cluster** | Breathing + Rotation | Coupled oscillatory modes |


## Reference Measure Types

| Type | Description | Use Case |
|------|-------------|----------|
| `gaussian_iso` | Isotropic Gaussian | General purpose |
| `uniform_square` | Uniform on [-1,1]² | Good coverage |
| `clean_circle_small` | Points on unit circle | Radial dynamics |
| `snapshot_begin` | Initial measure μ₀ | Track particle-level dynamics |

**Note**: Rotationally symmetric references cannot capture rotation of symmetric distributions.

## Citation
```bibtex
@article{khurana2025lot,
  title={Reservoir Computing on LOT-Embedded Measure-Valued Dynamical Systems},
  author={Busuladzic-Begic, Mia and Khurana, Varun},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.