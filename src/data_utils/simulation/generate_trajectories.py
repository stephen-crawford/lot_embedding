#!/usr/bin/env python3
"""
Generate particle trajectories for geodesic_transport, swirling_cluster, and lorenz_gmm systems.

Creates trajectory_jittered.npy files for each N value.

Usage:
    python generate_trajectories.py geodesic_transport
    python generate_trajectories.py swirling_cluster
    python generate_trajectories.py lorenz_gmm
    python generate_trajectories.py all
"""

from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Dict, List

from data_utils.simulation.measure_dynamical_systems import (
    SwirlingClusterSystem,
    GeodesicTransportSimulator,
    GaussianMixtureFlowSystem,
    add_posthoc_noise,
)


# ═══════════════════════════════════════════════════════════════
# SHAPE GENERATORS (for geodesic transport endpoints)
# ═══════════════════════════════════════════════════════════════

def make_circle(N, radius=1.0, center=(0, 0)):
    """Generate N points uniformly on a circle."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    return np.stack((x, y), axis=1)


def make_triangle_uniform(N, scale=1.5, center=(0, 0)):
    """Generate N points uniformly on triangle edges."""
    A = np.array([0, scale])
    B = np.array([scale * np.sin(np.pi / 3), -scale / 2])
    C = np.array([-scale * np.sin(np.pi / 3), -scale / 2])
    edges = [(A, B), (B, C), (C, A)]

    total_len = sum(np.linalg.norm(b - a) for a, b in edges)
    segment_lengths = [np.linalg.norm(b - a) for a, b in edges]
    segment_counts = [int(np.round(N * (l / total_len))) for l in segment_lengths]

    # Adjust to ensure exact point count
    diff = N - sum(segment_counts)
    segment_counts[0] += diff

    points = []
    for (a, b), count in zip(edges, segment_counts):
        for i in range(count):
            t = i / max(count, 1)
            points.append((1 - t) * a + t * b)

    points = np.array(points) + np.array(center)
    return points[:N]


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY GENERATORS
# ═══════════════════════════════════════════════════════════════

def generate_geodesic_transport(N, n_steps=400, n_cycles=2, seed=42):
    """
    Generate geodesic transport trajectory: circle ↔ triangle with BREATHING motion.
    
    Uses sinusoidal interpolation for smooth velocity (no jumps).
    
    Args:
        N: number of particles
        n_steps: total frames in trajectory
        n_cycles: number of full oscillations (circle→triangle→circle = 1 cycle)
    
    Returns trajectory of shape (n_steps, N, 2)
    """
    # # Create endpoints
    # circle = make_circle(N, radius=1.0)
    # triangle = make_triangle_uniform(N, scale=1.5)
    
    # # Initialize simulator
    # sim = GeodesicTransportSimulator(
    #     source_points=circle,
    #     target_points=triangle,
    # )
    
    # # Generate breathing trajectory with sinusoidal interpolation
    # trajectory = []
    # for i in range(n_steps):
    #     # Sinusoidal parameter: smoothly oscillates between 0 and 1
    #     theta = n_cycles * 2 * np.pi * i / n_steps
    #     t = (1 - np.cos(theta)) / 2  # t ∈ [0, 1], smooth
        
    #     points = sim.state_at(t)
    #     trajectory.append(points)
    
    # return np.array(trajectory)

    
    # Create endpoints
    circle = make_circle(N, radius=1.0)
    triangle = make_triangle_uniform(N, scale=1.5)
    
    # Initialize simulator (computes optimal transport map internally)
    sim = GeodesicTransportSimulator(
        source_points=circle,
        target_points=triangle,
        # seed=seed
    )
    
    # Use class method for trajectory generation
    # This ensures consistency and avoids code duplication
    trajectory = sim.get_cyclical_trajectory(n_cycles=n_cycles, n_steps=n_steps)
    
    return trajectory

    

# def generate_geodesic_transport(N, n_steps=100, n_cycles=2, seed=42):
#     """
#     Generate geodesic transport trajectory: circle ↔ triangle.
    
#     Args:
#         N: number of particles
#         n_steps: steps per half-cycle (circle→triangle or triangle→circle)
#         n_cycles: number of full cycles (circle→triangle→circle = 1 cycle)
    
#     Returns trajectory of shape (T, N, 2)
#     For n_cycles=2: circle→triangle→circle→triangle→circle
#     """
#     # Create endpoints
#     circle = make_circle(N, radius=1.0)
#     triangle = make_triangle_uniform(N, scale=1.5)
    
#     # Initialize simulator
#     sim = GeodesicTransportSimulator(
#         source_points=circle,
#         target_points=triangle,
#     )
    
#     # Generate one forward sweep: t ∈ [0, 1] (circle → triangle)
#     forward = []
#     for t in np.linspace(0, 1, n_steps):
#         points = sim.state_at(t)
#         forward.append(points)
#     forward = np.array(forward)
    
#     # Reverse sweep: t ∈ [1, 0] (triangle → circle)
#     reverse = forward[::-1][1:]    # Only remove duplicate t=1 → 99 frames  

#     # Build full trajectory with n_cycles
#     # cycle pattern: forward + reverse + forward + reverse + ...
#     # For n_cycles=2: circle→triangle→circle→triangle→circle
#     #   = forward + reverse + forward + reverse + [endpoint]
    
#     trajectory = forward.copy()
#     for i in range(n_cycles * 2 - 1):  # 2*n_cycles - 1 more segments
#         if i % 2 == 0:
#             # Add reverse (triangle → circle), skip first frame (duplicate)
#             trajectory = np.concatenate([trajectory, reverse], axis=0)
#         else:
#             # Add forward (circle → triangle), skip first frame (duplicate)
#             trajectory = np.concatenate([trajectory, forward[1:]], axis=0)
    
#     return trajectory


def generate_swirling_cluster(N, n_cycles=2, n_steps=250, max_radius=6.0, model_noise_scale=0.0, seed=42):
    """
    Generate swirling cluster trajectory with radial breathing.
    
    Particles rotate while expanding/contracting.
    
    Returns trajectory of shape (T, N, 2)
    """
    system = SwirlingClusterSystem(N=N, dim=2, noise_scale=model_noise_scale, seed=seed)
    
    trajectory = np.array(system.get_cyclical_trajectory(
        n_cycles=n_cycles, 
        n_steps=n_steps,
        max_radius=max_radius
    ))
    
    return trajectory


def generate_lorenz_gmm(N, n_steps=400, n_cycles=2, dim=3, project_to_2d=True, seed=42):
    """
    Generate Lorenz-driven Gaussian Mixture Model trajectory.
    
    The system is 3D by default, but can be projected to 2D for LOT embeddings.
    
    Args:
        N: number of particles
        n_steps: total frames in trajectory
        n_cycles: number of cycles (for consistency with other systems)
        dim: dimension (3 for 3D Lorenz, will be projected to 2D if project_to_2d=True)
        project_to_2d: if True, project 3D trajectory to 2D using PCA
        seed: random seed
    
    Returns trajectory of shape (T, N, 2) if project_to_2d=True, else (T, N, 3)
    """
    system = GaussianMixtureFlowSystem(
        N=N,
        K=3,  # 3 mixture components
        dim=dim,
        lorenz_params={'sigma': 10.0, 'rho': 28.0, 'beta': 8/3},
        dt=0.01,
        seed=seed
    )
    
    # Generate trajectory
    trajectory_3d = []
    system.reset()
    for i in range(n_steps):
        system.step()
        trajectory_3d.append(system.get_state())
    
    trajectory_3d = np.array(trajectory_3d)  # (T, N, 3)
    
    # Project to 2D if requested
    if project_to_2d and dim == 3:
        try:
            from sklearn.decomposition import PCA
            # Flatten for PCA
            T, N_traj, d3 = trajectory_3d.shape
            traj_flat = trajectory_3d.reshape(-1, d3)
            
            # Fit PCA on all data
            pca = PCA(n_components=2)
            traj_2d_flat = pca.fit_transform(traj_flat)
            trajectory = traj_2d_flat.reshape(T, N_traj, 2)
            
            # Print variance explained
            var_explained = pca.explained_variance_ratio_.sum()
            print(f"  [Lorenz] Projected 3D→2D: explained variance = {var_explained:.2%}")
        except ImportError:
            print("  [WARN] sklearn not available, using first 2 dimensions")
            trajectory = trajectory_3d[:, :, :2]
    else:
        trajectory = trajectory_3d
    
    return trajectory


# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════

def generate_all_trajectories(
    system: str,
    N_list: List[int],
    results_root: Path,
    noise_scale: float = 0.01,
    noise_seed: int = 999,
    n_steps: Optional[int] = None,
    n_cycles: int = 2,
    verbose: bool = True,
) -> Dict[int, Path]:
    """
    Generate trajectories for all N values and save to disk.
    
    Args:
        system: 'geodesic_transport' or 'swirling_cluster'
        N_list: list of particle counts to generate
        results_root: base directory for outputs
        noise_scale: post-hoc Gaussian noise scale (position-space)
        noise_seed: random seed for reproducible noise
        n_steps: trajectory length (default: 400 for geodesic, 250 for swirling)
        n_cycles: number of oscillation cycles
        verbose: print progress
        
    Returns:
        paths: dict mapping N → saved file path
    """
    results_root = Path(results_root)
    saved_paths = {}
    
    # Set default n_steps based on system
    if n_steps is None:
        if system == "geodesic_transport":
            n_steps = 400
        elif system == "swirling_cluster":
            n_steps = 250
        elif system == "lorenz_gmm":
            n_steps = 400
        else:
            n_steps = 400  # Default
    
    for N in N_list:
        if verbose:
            print(f"[{system}] Generating N={N}...")
        
        # Generate trajectory using appropriate system
        if system == "geodesic_transport":
            trajectory = generate_geodesic_transport(
                N=N,
                n_steps=n_steps,
                n_cycles=n_cycles,
            )
        elif system == "swirling_cluster":
            trajectory = generate_swirling_cluster(
                N=N,
                n_cycles=n_cycles,
                n_steps=n_steps,
                max_radius=6.0,
            )
        elif system == "lorenz_gmm":
            trajectory = generate_lorenz_gmm(
                N=N,
                n_steps=n_steps,
                n_cycles=n_cycles,  # For consistency, though Lorenz doesn't have clear cycles
                dim=3,
                project_to_2d=True,  # Project to 2D for LOT embeddings
                seed=42,
            )
        else:
            raise ValueError(
                f"Unknown system: {system}. "
                f"Expected 'geodesic_transport', 'swirling_cluster', or 'lorenz_gmm'."
            )
        
        # Add post-hoc noise (position-space perturbation)
        # Note: This is distinct from transport-space error discussed in the theory
        trajectory_jittered = add_posthoc_noise(trajectory, noise_scale, noise_seed)
        
        # Save trajectory
        out_dir = results_root / f"{system}_N_{N}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_dir / "trajectory_jittered.npy"
        np.save(out_path, trajectory_jittered.astype(np.float32))
        saved_paths[N] = out_path
        
        if verbose:
            T = trajectory_jittered.shape[0]
            print(f"  → Saved: {out_path} | shape: ({T}, {N}, 2)")
    
    if verbose:
        print(f"\n[DONE] Generated {len(N_list)} trajectories for {system}")
    
    return saved_paths


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate particle trajectories for LOT+RC experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "system",
        choices=["geodesic_transport", "swirling_cluster", "lorenz_gmm", "all"],
        help="Which dynamical system to generate",
    )
    parser.add_argument(
        "--N",
        type=int,
        nargs="+",
        default=[100, 250, 500, 750, 1000, 1500],
        help="Particle counts to generate",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Output directory root",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.01,
        help="Post-hoc Gaussian noise scale",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=999,
        help="Random seed for noise",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Trajectory length (default: 400 for geodesic, 250 for swirling)",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=2,
        help="Number of oscillation cycles",
    )
    args = parser.parse_args()
    
    # Determine which systems to generate
    if args.system == "all":
        systems = ["geodesic_transport", "swirling_cluster", "lorenz_gmm"]
    else:
        systems = [args.system]
    
    for system in systems:
        generate_all_trajectories(
            system=system,
            N_list=args.N,
            results_root=args.results_root,
            noise_scale=args.noise_scale,
            noise_seed=args.noise_seed,
            n_steps=args.n_steps,
            n_cycles=args.n_cycles,
        )


if __name__ == "__main__":
    main()