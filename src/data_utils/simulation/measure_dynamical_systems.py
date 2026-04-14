"""
Measure-valued dynamical systems for LOT+RC experiments.

Classes:
    - SwirlingClusterSystem: 2D breathing spiral (rotation + radial oscillation)
    - GeodesicTransportSimulator: W2 geodesic between two empirical measures
    - LorenzParticleSystem: 3D Lorenz attractor with persistent particles (RC benchmark)
    - CuckerSmaleSystem: Flocking dynamics
    - GaussianMixtureFlowSystem: Lorenz-driven GMM (legacy, prefer LorenzParticleSystem)

IMPORTANT DYNAMICS NOTES:
------------------------
SwirlingClusterSystem has TWO modes:
  1. step() method: Pure rotation on fixed-radius circle
  2. get_cyclical_trajectory(): Breathing spiral (rotation + radial oscillation)
  
For LOT+RC experiments, use get_cyclical_trajectory() which provides
non-trivial dynamics that break rotational symmetry.
"""

from abc import ABC, abstractmethod
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


class MeasureDynamicalSystem(ABC):
    """
    Abstract base class for measure-valued dynamical systems.
    """
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, dt=0.01):
        pass

    @abstractmethod
    def get_state(self):
        pass

    def run_steps(self, n, dt=0.01):
        """Run the system for n steps and return trajectory of states."""
        self.reset()
        trajectory = []
        for _ in range(n):
            self.step(dt)
            trajectory.append(self.get_state())
        return trajectory

    def run_cyclical_steps(self, n, dt=0.01):
        self.reset()
        trajectory = []
    
        # Forward pass
        for _ in range(n):
            self.step(dt)
            trajectory.append(self.get_state())
    
        # Reverse velocities to simulate return path
        if hasattr(self, 'velocities'):
            self.velocities = -self.velocities
    
        # Backward pass
        for _ in range(n):
            self.step(dt)
            trajectory.append(self.get_state())
    
        return trajectory

    def get_config(self):
        """Return a dictionary of configuration parameters."""
        return {}


class SwirlingClusterSystem(MeasureDynamicalSystem):
    """
    2D breathing spiral dynamics: particles rotate while oscillating radially.
    
    This system provides two distinct dynamics modes:
    
    1. step() method - PURE ROTATION:
       - Particles rotate on a fixed-radius circle
       - Dynamics: dx/dt = ω * S * x, where S is the 90° rotation matrix
       - WARNING: With rotationally-symmetric references (e.g., circles),
         OT solutions are degenerate and LOT velocities may be meaningless.
    
    2. get_cyclical_trajectory() - BREATHING SPIRAL (recommended for LOT+RC):
       - Particles rotate while expanding/contracting radially
       - Radius oscillates sinusoidally: r(t) = 1 + (r_max - 1) * (1 - cos(phase))/2
       - This breaks rotational symmetry, making OT solutions unique
       - Produces learnable, non-trivial dynamics for reservoir computing
    
    For LOT+RC experiments, ALWAYS use get_cyclical_trajectory() to generate
    training data. The breathing motion provides the radial dynamics needed
    for meaningful LOT velocity predictions.
    
    Cycle Structure:
    ----------------
    With n_cycles=2, n_steps=250:
    - One complete oscillation (r=1 → r=max → r=1) takes n_steps/n_cycles = 125 frames
    - cycle_length = 125 for warm-up alignment
    
    Attributes:
        N: Number of particles
        dim: Spatial dimension (must be 2)
        noise_scale: Gaussian noise magnitude (position-space perturbation)
        initial_positions: Starting configuration (unit circle)
        weights: Uniform probability weights (1/N each)
    """
    
    def __init__(self, N, dim=2, noise_scale=0.0, seed=None):
        assert dim == 2, "SwirlingClusterSystem only supports 2D"
        self.N = N
        self.dim = dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Store initial configuration for LOT reference
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        self.initial_positions = np.column_stack((np.cos(theta), np.sin(theta)))
        self.weights = np.full(self.N, 1.0 / self.N)
        
        self.reset()
    
    def reset(self):
        # Initialize N particles uniformly spaced on unit circle (DETERMINISTIC)
        self.positions = self.initial_positions.copy()
        self.history = []
    
    def step(self, dt=0.01, swirl_strength=10.0):
        """
        Pure rotation step (fixed radius).
        
        WARNING: This produces pure rotational dynamics. For rotationally-symmetric
        references, OT solutions are degenerate. Use get_cyclical_trajectory() 
        for LOT+RC experiments instead.
        
        Uses exact rotation matrix to avoid numerical drift.
        """
        theta = dt * swirl_strength  # angle to rotate
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        self.positions = (self.positions @ R.T + 
                          self.noise_scale * np.sqrt(dt) * self.rng.normal(size=self.positions.shape))
        self.history.append(self.get_state())
    
    def get_state(self):
        """Current particle positions."""
        return self.positions.copy()
    
    def get_measure(self):
        """(positions, weights) — what Sinkhorn/LOT expect."""
        return self.positions.copy(), self.weights.copy()
    
    def get_reference(self):
        """Natural LOT reference σ: use initial circular configuration μ₀."""
        return self.initial_positions.copy(), self.weights.copy()
    
    def get_config(self):
        return {
            'system': 'SwirlingClusterSystem',
            'N': self.N,
            'dim': self.dim,
            'noise_scale': self.noise_scale,
            'seed': self.seed
        }
    
    def get_cyclical_trajectory(self, n_cycles=2, n_steps=250, max_radius=6.0):
        """
        Generate breathing spiral trajectory (RECOMMENDED for LOT+RC).
        
        Particles rotate continuously while their radius oscillates sinusoidally.
        This provides:
        - Rotational dynamics (angular motion)
        - Radial dynamics (breathing) that break rotational symmetry
        - Smooth, periodic motion suitable for reservoir learning
        
        Dynamics:
            θ(t) = θ_0 + (n_cycles * 2π) * (t / n_steps)     [rotation]
            r(t) = 1 + (max_radius - 1) * (1 - cos(phase))/2  [breathing]
            phase = n_cycles * 2π * (t / n_steps)
        
        Args:
            n_cycles: Number of complete breathing oscillations (r=1→max→1)
            n_steps: Total frames in trajectory
            max_radius: Maximum radius during expansion (starts and ends at 1.0)
        
        Returns:
            trajectory: np.ndarray of shape (n_steps, N, 2)
        
        Cycle Structure:
            - One oscillation = n_steps / n_cycles frames
            - For warm-up alignment, set cycle_length = n_steps // n_cycles
            
        Example:
            n_cycles=2, n_steps=250 → cycle_length=125
            Frame 0:    r=1.0 (minimum)
            Frame 62:   r=6.0 (maximum)  
            Frame 125:  r=1.0 (minimum) ← end of first cycle
            Frame 187:  r=6.0 (maximum)
            Frame 250:  r=1.0 (minimum) ← end of second cycle
        """
        self.reset()
        
        # Store initial angles (particles start on unit circle)
        initial_angles = np.arctan2(self.initial_positions[:, 1], self.initial_positions[:, 0])
        
        # Total rotation over entire trajectory
        total_rotation = n_cycles * 2 * np.pi
        
        trajectory = []
        for i in range(n_steps):
            t = i / n_steps  # progress [0, 1)
            
            # Rotation: continuous rotation throughout
            angle_offset = total_rotation * t
            angles = initial_angles + angle_offset
            
            # Breathing: sinusoidal radius oscillation
            # Goes from 1.0 -> max_radius -> 1.0 for each cycle
            phase = n_cycles * 2 * np.pi * t
            radius = 1.0 + (max_radius - 1.0) * (1 - np.cos(phase)) / 2
            
            # Compute positions
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            self.positions = np.column_stack((x, y))
            
            if self.noise_scale > 0:
                self.positions += self.noise_scale * self.rng.normal(size=self.positions.shape)
            
            trajectory.append(self.get_state())
        
        return np.array(trajectory)
    
    @staticmethod
    def compute_cycle_length(n_steps: int, n_cycles: int) -> int:
        """
        Compute the cycle length for warm-up alignment.
        
        One cycle = one complete breathing oscillation (r=1 → max → 1).
        
        Args:
            n_steps: Total trajectory length
            n_cycles: Number of oscillations
            
        Returns:
            cycle_length: Frames per oscillation
        """
        return n_steps // n_cycles


class GeodesicTransportSimulator(MeasureDynamicalSystem):
    """
    W2 geodesic between two empirical measures with uniform weights.
    
    Interpolates along the Monge map T: x0_i -> x1_{perm[i]} found by 
    Hungarian assignment on squared Euclidean cost.
    
    Dynamics:
        μ_t = ((1-t)Id + t*T)_# μ_0
        
    where T is the optimal transport map from μ_0 to μ_1.
    
    Cycle Structure:
    ----------------
    With sinusoidal interpolation (get_cyclical_trajectory):
        t(θ) = (1 - cos(θ)) / 2
        
    This gives smooth velocity at the endpoints (no discontinuities).
    
    With n_cycles=2, n_steps=400:
    - One complete oscillation (circle → triangle → circle) = 200 frames
    - cycle_length = 200 for warm-up alignment
    - Velocity magnitude: |v| ∝ sin(θ), zero at endpoints, max at midpoint
    
    Note on velocity stationarity:
        The velocity magnitude is NOT constant — it oscillates sinusoidally.
        This is still learnable but requires the reservoir to capture the
        periodic structure of the velocity field.
    """

    def __init__(self, source_points, target_points):
        assert source_points.shape == target_points.shape, "Source and target must match in shape."
        self.x0 = np.asarray(source_points)
        self.x1 = np.asarray(target_points)
        self.N, self.dim = self.x0.shape

        # Compute a Monge map via assignment on squared distances (W2^2)
        M = ot.dist(self.x0, self.x1, metric="euclidean") ** 2
        row_ind, col_ind = linear_sum_assignment(M)
        self.perm = col_ind[np.argsort(row_ind)]
        self.mapped_targets = self.x1[self.perm]  # T(x0_i)

        # Uniform weights for the empirical measures
        self.weights = np.full(self.N, 1.0 / self.N)

        self.reset()

    def reset(self):
        self.t = 0.0
        self.direction = 1
        self.current_points = self.x0.copy()
        self.history = []

    def state_at(self, t):
        """Closed-form geodesic state at arbitrary t in [0,1]."""
        t = float(np.clip(t, 0.0, 1.0))
        return (1.0 - t) * self.x0 + t * self.mapped_targets

    def step(self, dt=0.01):
        """Ping-pong between t=0 and t=1 while recording states."""
        self.t += self.direction * dt
        if self.t >= 1.0:
            self.t = 1.0
            self.direction = -1
        elif self.t <= 0.0:
            self.t = 0.0
            self.direction = 1
        self.current_points = self.state_at(self.t)
        self.history.append((self.t, self.current_points.copy()))

    def get_state(self):
        """Positions only (backward-compatible)."""
        return self.current_points.copy()

    def get_measure(self):
        """(positions, weights) — what Sinkhorn/LOT expect."""
        return self.current_points.copy(), self.weights.copy()

    def get_reference(self):
        """Natural LOT reference σ: use μ0 support with uniform weights."""
        return self.x0.copy(), self.weights.copy()

    def displacement_map(self):
        """T(x) - x evaluated on σ=x0; used for LOT-velocity experiments."""
        return self.mapped_targets - self.x0

    def get_config(self):
        return {
            'system': 'GeodesicTransportSimulator',
            'N': self.N,
            'dim': self.dim,
            't': self.t
        }

    def get_cyclical_trajectory(self, n_cycles=2, n_steps=400):
        """
        Generate breathing geodesic trajectory with smooth velocity.
        
        Uses sinusoidal interpolation: t = (1 - cos(θ)) / 2
        This gives continuous velocity (no jumps at boundaries).
        
        Dynamics:
            θ(i) = n_cycles * 2π * i / n_steps
            t(θ) = (1 - cos(θ)) / 2
            
        Velocity structure:
            dt/dθ = sin(θ) / 2
            |velocity| is zero at t=0,1 (endpoints) and maximum at t=0.5 (midpoint)
        
        Args:
            n_cycles: Number of circle→triangle→circle oscillations
            n_steps: Total frames in trajectory
            
        Returns:
            trajectory: (n_steps, N, dim) array of particle positions
            
        Cycle Structure:
            One oscillation = n_steps / n_cycles frames
            For warm-up alignment, set cycle_length = n_steps // n_cycles
            
        Example:
            n_cycles=2, n_steps=400 → cycle_length=200
            Frame 0:    t=0.0 (circle)
            Frame 100:  t=1.0 (triangle)
            Frame 200:  t=0.0 (circle) ← end of first cycle
            Frame 300:  t=1.0 (triangle)
            Frame 400:  t=0.0 (circle) ← end of second cycle
        """
        self.reset()
        
        trajectory = []
        for i in range(n_steps):
            # Sinusoidal parameter: smoothly oscillates between 0 and 1
            theta = n_cycles * 2 * np.pi * i / n_steps
            t = (1 - np.cos(theta)) / 2  # t ∈ [0, 1], smooth
            
            # Geodesic interpolation at parameter t
            self.current_points = self.state_at(t)
            self.t = t
            
            trajectory.append(self.get_state())
        
        return np.array(trajectory)
    
    @staticmethod
    def compute_cycle_length(n_steps: int, n_cycles: int) -> int:
        """
        Compute the cycle length for warm-up alignment.
        
        One cycle = one complete oscillation (source → target → source).
        
        Args:
            n_steps: Total trajectory length
            n_cycles: Number of oscillations
            
        Returns:
            cycle_length: Frames per oscillation
        """
        return n_steps // n_cycles


class CuckerSmaleSystem(MeasureDynamicalSystem):
    """Cucker-Smale flocking dynamics."""
    
    def __init__(self, N, dim=2, beta=0.3, seed=None):
        self.N = N
        self.dim = dim
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reset()

    def reset(self):
        self.positions = self.rng.random((self.N, self.dim)) + np.array([2.0] + [0.0] * (self.dim - 1))
        self.velocities = self.rng.normal(scale=3.0, size=(self.N, self.dim))
        self.history = []

    def step(self, dt=0.01):
        x = self.positions
        v = self.velocities

        diffs = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diffs, axis=-1) + 1e-8
        weights = 1.0 / (1.0 + dist**2) ** self.beta

        alignment = (weights[..., None] * (v[None, :, :] - v[:, None, :])).sum(axis=1)
        self.velocities += dt * alignment / self.N
        self.positions += dt * self.velocities
        self.history.append(self.get_state())

    def get_state(self):
        return self.positions.copy()

    def get_config(self):
        return {
            'system': 'CuckerSmaleSystem',
            'N': self.N,
            'dim': self.dim,
            'beta': self.beta,
            'seed': self.seed
        }

    def get_cyclical_trajectory(self, n_steps=250, dt=0.01):
        self.reset()
        trajectory = []
        for _ in range(n_steps):
            self.step(dt)
            trajectory.append(self.get_state())
        return np.array(trajectory + trajectory[::-1][1:-1])


class LorenzParticleSystem(MeasureDynamicalSystem):
    """
    3D Lorenz attractor with N persistent particles.

    Standard RC benchmark: each particle follows the Lorenz ODE with small
    initial perturbations. Particles maintain identity across timesteps,
    making this suitable for LOT+RC experiments.

    Dynamics (Lorenz '63):
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    Default parameters (σ=10, ρ=28, β=8/3) produce chaotic dynamics
    on the famous butterfly-shaped strange attractor.

    For LOT+RC:
    - Particles are persistent (same particle i at each timestep)
    - 3D dynamics (x, y, z)
    - Chaotic but deterministic - good for testing RC prediction
    - Use get_cyclical_trajectory() for training data
    """

    def __init__(self, N, dim=3, lorenz_params=None, dt=0.01, seed=None,
                 init_spread=1.0, init_center=None):
        """
        Args:
            N: Number of particles
            dim: Must be 3 for Lorenz
            lorenz_params: {'sigma': 10, 'rho': 28, 'beta': 8/3}
            dt: Integration timestep
            seed: Random seed
            init_spread: Spread of initial particle positions
            init_center: Center of initial cloud (default: near attractor)
        """
        assert dim == 3, "Lorenz system is 3D"
        self.N = N
        self.dim = dim
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
        self.init_spread = init_spread
        # Start near the attractor
        self.init_center = init_center if init_center is not None else np.array([1.0, 1.0, 25.0])

        # Uniform weights for LOT
        self.weights = np.full(self.N, 1.0 / self.N)

        self.reset()

    def lorenz_deriv(self, state):
        """Compute Lorenz derivatives for a single point or array of points."""
        s = self.lorenz_params['sigma']
        r = self.lorenz_params['rho']
        b = self.lorenz_params['beta']

        if state.ndim == 1:
            x, y, z = state
            return np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        else:
            x, y, z = state[:, 0], state[:, 1], state[:, 2]
            return np.column_stack([s * (y - x), x * (r - z) - y, x * y - b * z])

    def reset(self):
        """Initialize N particles near the attractor with small spread."""
        # All particles start near init_center with small perturbations
        self.positions = self.init_center + self.init_spread * self.rng.standard_normal((self.N, 3))
        self.initial_positions = self.positions.copy()
        self.history = []

    def step(self, dt=None):
        """Integrate all particles forward using Euler (consistent with other systems)."""
        dt = dt or self.dt
        self.positions = self.positions + dt * self.lorenz_deriv(self.positions)
        self.history.append(self.get_state())

    def get_state(self):
        return self.positions.copy()

    def get_measure(self):
        """(positions, weights) for LOT/Sinkhorn."""
        return self.positions.copy(), self.weights.copy()

    def get_reference(self):
        """LOT reference: initial particle configuration."""
        return self.initial_positions.copy(), self.weights.copy()

    def get_config(self):
        return {
            'system': 'LorenzParticleSystem',
            'N': self.N,
            'dim': self.dim,
            'dt': self.dt,
            'lorenz_params': self.lorenz_params,
            'seed': self.seed,
            'init_spread': self.init_spread
        }

    def get_cyclical_trajectory(self, n_steps=500, warmup=100):
        """
        Generate Lorenz trajectory suitable for LOT+RC.

        Note: Lorenz is chaotic, NOT periodic. However, it has
        quasi-periodic behavior around the two lobes of the attractor.

        For RC training:
        - Use warmup steps to let transients decay onto the attractor
        - The trajectory is NOT cyclic, but RC can learn the dynamics
        - For fair comparison, use the same warm_steps analysis as tornado data

        Args:
            n_steps: Number of trajectory steps after warmup
            warmup: Warmup steps to reach the attractor (discarded)

        Returns:
            trajectory: (n_steps, N, 3) array of particle positions
        """
        self.reset()

        # Warmup: let particles settle onto the attractor
        for _ in range(warmup):
            self.step()

        # Reset initial positions to current (on-attractor) state
        self.initial_positions = self.positions.copy()

        # Generate trajectory
        trajectory = []
        for _ in range(n_steps):
            self.step()
            trajectory.append(self.get_state())

        return np.array(trajectory)

    @staticmethod
    def compute_cycle_length(n_steps: int, n_cycles: int = 1) -> int:
        """
        Lorenz is chaotic, not cyclic. Return n_steps for compatibility.

        For Lorenz, warm_start should be based on:
        - Lyapunov time (~1/λ ≈ 1.1 time units)
        - Or empirical cyclicality analysis (will show low confidence)
        """
        return n_steps // max(n_cycles, 1)


class GaussianMixtureFlowSystem(MeasureDynamicalSystem):
    """
    Lorenz-driven Gaussian mixture model (legacy).

    NOTE: For LOT+RC experiments, prefer LorenzParticleSystem which
    maintains particle identity across timesteps.

    This system randomly resamples N points from GMM each step,
    which breaks particle correspondence needed for LOT velocity.
    """

    def __init__(self, N, K=3, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
        self.N = N
        self.K = K
        self.dim = dim
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.max_clip = max_clip
        self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
        self.cov_scale = cov_scale
        self.reset()
        self.means_trajectory = []

    def lorenz_deriv(self, mean):
        x, y, z = mean
        s = self.lorenz_params['sigma']
        r = self.lorenz_params['rho']
        b = self.lorenz_params['beta']
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        return np.array([dx, dy, dz])

    def reset(self):
        self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
        self.cov = np.eye(3) * self.cov_scale
        self.current_points = np.zeros((self.N, self.dim))
        self.history = []
        self.means_trajectory = []

    def step(self, dt=None):
        dt = dt or self.dt
        derivs = np.array([self.lorenz_deriv(m) for m in self.means])
        derivs = np.nan_to_num(derivs, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)

        self.means += dt * derivs
        self.means = np.clip(self.means, -self.max_clip, self.max_clip)
        self.means_trajectory.append(self.means.copy())

        self.assignments = self.rng.integers(self.K, size=self.N)
        points = np.array([self.rng.multivariate_normal(self.means[k], self.cov)
                           for k in self.assignments])
        points += self.rng.normal(scale=0.02, size=points.shape)

        self.current_points = points
        self.history.append(self.get_state())

    def get_state(self):
        return self.current_points.copy()

    def get_config(self):
        return {
            'system': 'GaussianMixtureFlowSystem',
            'N': self.N,
            'K': self.K,
            'dim': self.dim,
            'dt': self.dt,
            'lorenz_params': self.lorenz_params,
            'seed': self.seed
        }


def add_posthoc_noise(trajectory, noise_scale=0.01, seed=None):
    """Add Gaussian noise to trajectory."""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(scale=noise_scale, size=trajectory.shape)
    return trajectory + noise


# ═══════════════════════════════════════════════════════════════
# CYCLE LENGTH UTILITIES
# ═══════════════════════════════════════════════════════════════

def get_system_cycle_length(system_name: str, n_steps: int, n_cycles: int) -> int:
    """
    Get the correct cycle length for a system.
    
    This is the number of frames in one complete oscillation.
    Use for warm-up alignment in forecasting experiments.
    
    Args:
        system_name: "geodesic_transport" or "swirling_cluster"
        n_steps: Total trajectory length
        n_cycles: Number of oscillations in trajectory
        
    Returns:
        cycle_length: Frames per oscillation
        
    Example:
        >>> get_system_cycle_length("geodesic_transport", 400, 2)
        200
        >>> get_system_cycle_length("swirling_cluster", 250, 2)
        125
    """
    return n_steps // n_cycles


# Default trajectory parameters for each system
SYSTEM_DEFAULTS = {
    "geodesic_transport": {
        "n_steps": 400,
        "n_cycles": 2,
        "cycle_length": 200,  # = 400 // 2
    },
    "swirling_cluster": {
        "n_steps": 250,
        "n_cycles": 2,
        "cycle_length": 125,  # = 250 // 2
    },
    "lorenz": {
        "n_steps": 500,
        "warmup": 100,
        "dt": 0.01,
        "is_cyclic": False,  # Lorenz is chaotic, not periodic
        "recommended_warm_steps": 50,  # ~0.5 Lyapunov time
    },
}


# """
# Measure-valued dynamical systems for LOT+RC experiments.

# Classes:
#     - SwirlingClusterSystem: 2D rotation flow on a circle
#     - GeodesicTransportSimulator: W2 geodesic between two empirical measures
#     - CuckerSmaleSystem: Flocking dynamics
#     - GaussianMixtureFlowSystem: Lorenz-driven GMM
# """

# from abc import ABC, abstractmethod
# import numpy as np
# import ot
# from scipy.optimize import linear_sum_assignment


# class MeasureDynamicalSystem(ABC):
#     """
#     Abstract base class for measure-valued dynamical systems.
#     """
#     @abstractmethod
#     def reset(self):
#         pass

#     @abstractmethod
#     def step(self, dt=0.01):
#         pass

#     @abstractmethod
#     def get_state(self):
#         pass

#     def run_steps(self, n, dt=0.01):
#         """Run the system for n steps and return trajectory of states."""
#         self.reset()
#         trajectory = []
#         for _ in range(n):
#             self.step(dt)
#             trajectory.append(self.get_state())
#         return trajectory

#     def run_cyclical_steps(self, n, dt=0.01):
#         self.reset()
#         trajectory = []
    
#         # Forward pass
#         for _ in range(n):
#             self.step(dt)
#             trajectory.append(self.get_state())
    
#         # Reverse velocities to simulate return path
#         if hasattr(self, 'velocities'):
#             self.velocities = -self.velocities
    
#         # Backward pass
#         for _ in range(n):
#             self.step(dt)
#             trajectory.append(self.get_state())
    
#         return trajectory

#     def get_config(self):
#         """Return a dictionary of configuration parameters."""
#         return {}


# class SwirlingClusterSystem(MeasureDynamicalSystem):
#     """
#     2D rotation flow: particles on a circle rotating with angular velocity ω.
#     Dynamics: dx/dt = ω * S * x, where S is the 90° rotation matrix.
#     """
#     def __init__(self, N, dim=2, noise_scale=0.0, seed=None):
#         assert dim == 2, "SwirlingClusterSystem only supports 2D"
#         self.N = N
#         self.dim = dim
#         self.noise_scale = noise_scale
#         self.rng = np.random.default_rng(seed)
#         self.seed = seed
        
#         # Store initial configuration for LOT reference
#         theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
#         self.initial_positions = np.column_stack((np.cos(theta), np.sin(theta)))
#         self.weights = np.full(self.N, 1.0 / self.N)
        
#         self.reset()
    
#     def reset(self):
#         # Initialize N particles uniformly spaced on unit circle (DETERMINISTIC)
#         self.positions = self.initial_positions.copy()
#         self.history = []
    
#     def step(self, dt=0.01, swirl_strength=10.0):
#         """Exact rotation using rotation matrix (no numerical drift)."""
#         theta = dt * swirl_strength  # angle to rotate
#         c, s = np.cos(theta), np.sin(theta)
#         R = np.array([[c, -s], [s, c]])
        
#         self.positions = (self.positions @ R.T + 
#                           self.noise_scale * np.sqrt(dt) * self.rng.normal(size=self.positions.shape))
#         self.history.append(self.get_state())
    
#     def get_state(self):
#         """Current particle positions."""
#         return self.positions.copy()
    
#     def get_measure(self):
#         """(positions, weights) — what Sinkhorn/LOT expect."""
#         return self.positions.copy(), self.weights.copy()
    
#     def get_reference(self):
#         """Natural LOT reference σ: use initial circular configuration μ₀."""
#         return self.initial_positions.copy(), self.weights.copy()
    
#     def get_config(self):
#         return {
#             'system': 'SwirlingClusterSystem',
#             'N': self.N,
#             'dim': self.dim,
#             'noise_scale': self.noise_scale,
#             'seed': self.seed
#         }
    
#     def get_cyclical_trajectory(self, n_cycles=2, n_steps=250, max_radius=6.0):
#         """
#         Generate swirling trajectory with radial breathing.
        
#         Particles rotate while expanding to max_radius then contracting back.
        
#         Args:
#             n_cycles: Number of expand/contract cycles
#             n_steps: Total frames in trajectory
#             max_radius: Maximum radius during expansion (default 6.0, starts at 1.0)
#         """
#         self.reset()
        
#         # Store initial angles (particles start on unit circle)
#         initial_angles = np.arctan2(self.initial_positions[:, 1], self.initial_positions[:, 0])
        
#         # Total rotation over entire trajectory
#         total_rotation = n_cycles * 2 * np.pi
        
#         trajectory = []
#         for i in range(n_steps):
#             t = i / n_steps  # progress [0, 1)
            
#             # Rotation: continuous rotation throughout
#             angle_offset = total_rotation * t
#             angles = initial_angles + angle_offset
            
#             # Breathing: sinusoidal radius oscillation
#             # Goes from 1.0 -> max_radius -> 1.0 for each cycle
#             phase = n_cycles * 2 * np.pi * t
#             radius = 1.0 + (max_radius - 1.0) * (1 - np.cos(phase)) / 2
            
#             # Compute positions
#             x = radius * np.cos(angles)
#             y = radius * np.sin(angles)
#             self.positions = np.column_stack((x, y))
            
#             if self.noise_scale > 0:
#                 self.positions += self.noise_scale * self.rng.normal(size=self.positions.shape)
            
#             trajectory.append(self.get_state())
        
#         return np.array(trajectory)


# class GeodesicTransportSimulator(MeasureDynamicalSystem):
#     """
#     W2 geodesic between two empirical measures with uniform weights.
#     Interpolates along the Monge map T: x0_i -> x1_{perm[i]} found by Hungarian assignment.
#     """

#     def __init__(self, source_points, target_points):
#         assert source_points.shape == target_points.shape, "Source and target must match in shape."
#         self.x0 = np.asarray(source_points)
#         self.x1 = np.asarray(target_points)
#         self.N, self.dim = self.x0.shape

#         # Compute a Monge map via assignment on squared distances (W2^2)
#         M = ot.dist(self.x0, self.x1, metric="euclidean") ** 2
#         row_ind, col_ind = linear_sum_assignment(M)
#         self.perm = col_ind[np.argsort(row_ind)]
#         self.mapped_targets = self.x1[self.perm]  # T(x0_i)

#         # Uniform weights for the empirical measures
#         self.weights = np.full(self.N, 1.0 / self.N)

#         self.reset()

#     def reset(self):
#         self.t = 0.0
#         self.direction = 1
#         self.current_points = self.x0.copy()
#         self.history = []

#     def state_at(self, t):
#         """Closed-form geodesic state at arbitrary t in [0,1]."""
#         t = float(np.clip(t, 0.0, 1.0))
#         return (1.0 - t) * self.x0 + t * self.mapped_targets

#     def step(self, dt=0.01):
#         """Ping-pong between t=0 and t=1 while recording states."""
#         self.t += self.direction * dt
#         if self.t >= 1.0:
#             self.t = 1.0
#             self.direction = -1
#         elif self.t <= 0.0:
#             self.t = 0.0
#             self.direction = 1
#         self.current_points = self.state_at(self.t)
#         self.history.append((self.t, self.current_points.copy()))

#     def get_state(self):
#         """Positions only (backward-compatible)."""
#         return self.current_points.copy()

#     def get_measure(self):
#         """(positions, weights) — what Sinkhorn/LOT expect."""
#         return self.current_points.copy(), self.weights.copy()

#     def get_reference(self):
#         """Natural LOT reference σ: use μ0 support with uniform weights."""
#         return self.x0.copy(), self.weights.copy()

#     def displacement_map(self):
#         """T(x) - x evaluated on σ=x0; used for LOT-velocity experiments."""
#         return self.mapped_targets - self.x0

#     def get_config(self):
#         return {
#             'system': 'GeodesicTransportSimulator',
#             'N': self.N,
#             'dim': self.dim,
#             't': self.t
#         }

#     def get_cyclical_trajectory(self, n_cycles=2, n_steps=400):
#         """
#         Generate breathing geodesic trajectory with smooth velocity.
        
#         Uses sinusoidal interpolation: t = (1 - cos(θ)) / 2
#         This gives continuous velocity (no jumps at boundaries).
        
#         Args:
#             n_cycles: Number of circle→triangle→circle oscillations
#             n_steps: Total frames in trajectory
            
#         Returns:
#             trajectory: (n_steps, N, dim) array of particle positions
#         """
#         self.reset()
        
#         trajectory = []
#         for i in range(n_steps):
#             # Sinusoidal parameter: smoothly oscillates between 0 and 1
#             theta = n_cycles * 2 * np.pi * i / n_steps
#             t = (1 - np.cos(theta)) / 2  # t ∈ [0, 1], smooth
            
#             # Geodesic interpolation at parameter t
#             self.current_points = self.state_at(t)
#             self.t = t
            
#             trajectory.append(self.get_state())
        
#         return np.array(trajectory)


# class CuckerSmaleSystem(MeasureDynamicalSystem):
#     """Cucker-Smale flocking dynamics."""
    
#     def __init__(self, N, dim=2, beta=0.3, seed=None):
#         self.N = N
#         self.dim = dim
#         self.beta = beta
#         self.rng = np.random.default_rng(seed)
#         self.seed = seed
#         self.reset()

#     def reset(self):
#         self.positions = self.rng.random((self.N, self.dim)) + np.array([2.0] + [0.0] * (self.dim - 1))
#         self.velocities = self.rng.normal(scale=3.0, size=(self.N, self.dim))
#         self.history = []

#     def step(self, dt=0.01):
#         x = self.positions
#         v = self.velocities

#         diffs = x[:, None, :] - x[None, :, :]
#         dist = np.linalg.norm(diffs, axis=-1) + 1e-8
#         weights = 1.0 / (1.0 + dist**2) ** self.beta

#         alignment = (weights[..., None] * (v[None, :, :] - v[:, None, :])).sum(axis=1)
#         self.velocities += dt * alignment / self.N
#         self.positions += dt * self.velocities
#         self.history.append(self.get_state())

#     def get_state(self):
#         return self.positions.copy()

#     def get_config(self):
#         return {
#             'system': 'CuckerSmaleSystem',
#             'N': self.N,
#             'dim': self.dim,
#             'beta': self.beta,
#             'seed': self.seed
#         }

#     def get_cyclical_trajectory(self, n_steps=250, dt=0.01):
#         self.reset()
#         trajectory = []
#         for _ in range(n_steps):
#             self.step(dt)
#             trajectory.append(self.get_state())
#         return np.array(trajectory + trajectory[::-1][1:-1])


# class GaussianMixtureFlowSystem(MeasureDynamicalSystem):
#     """Lorenz-driven Gaussian mixture model."""
    
#     def __init__(self, N, K=3, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
#         self.N = N
#         self.K = K
#         self.dim = dim
#         self.dt = dt
#         self.rng = np.random.default_rng(seed)
#         self.seed = seed
#         self.max_clip = max_clip
#         self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
#         self.cov_scale = cov_scale
#         self.reset()
#         self.means_trajectory = []

#     def lorenz_deriv(self, mean):
#         x, y, z = mean
#         s = self.lorenz_params['sigma']
#         r = self.lorenz_params['rho']
#         b = self.lorenz_params['beta']
#         dx = s * (y - x)
#         dy = x * (r - z) - y
#         dz = x * y - b * z
#         return np.array([dx, dy, dz])

#     def reset(self):
#         self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
#         self.cov = np.eye(3) * self.cov_scale
#         self.current_points = np.zeros((self.N, self.dim))
#         self.history = []
#         self.means_trajectory = []

#     def step(self, dt=None):
#         dt = dt or self.dt
#         derivs = np.array([self.lorenz_deriv(m) for m in self.means])
#         derivs = np.nan_to_num(derivs, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)
        
#         self.means += dt * derivs
#         self.means = np.clip(self.means, -self.max_clip, self.max_clip)
#         self.means_trajectory.append(self.means.copy())
    
#         self.assignments = self.rng.integers(self.K, size=self.N)
#         points = np.array([self.rng.multivariate_normal(self.means[k], self.cov)
#                            for k in self.assignments])
#         points += self.rng.normal(scale=0.02, size=points.shape)
    
#         self.current_points = points
#         self.history.append(self.get_state())

#     def get_state(self):
#         return self.current_points.copy()

#     def get_config(self):
#         return {
#             'system': 'GaussianMixtureFlowSystem',
#             'N': self.N,
#             'K': self.K,
#             'dim': self.dim,
#             'dt': self.dt,
#             'lorenz_params': self.lorenz_params,
#             'seed': self.seed
#         }


# def add_posthoc_noise(trajectory, noise_scale=0.01, seed=None):
#     """Add Gaussian noise to trajectory."""
#     if seed is not None:
#         np.random.seed(seed)
#     noise = np.random.normal(scale=noise_scale, size=trajectory.shape)
#     return trajectory + noise




    

# # """
# # Measure-valued dynamical systems for LOT+RC experiments.

# # Classes:
# #     - SwirlingClusterSystem: 2D rotation flow on a circle
# #     - GeodesicTransportSimulator: W2 geodesic between two empirical measures
# #     - CuckerSmaleSystem: Flocking dynamics
# #     - GaussianMixtureFlowSystem: Lorenz-driven GMM
# # """

# # from abc import ABC, abstractmethod
# # import numpy as np
# # import ot
# # from scipy.optimize import linear_sum_assignment


# # class MeasureDynamicalSystem(ABC):
# #     """
# #     Abstract base class for measure-valued dynamical systems.
# #     """
# #     @abstractmethod
# #     def reset(self):
# #         pass

# #     @abstractmethod
# #     def step(self, dt=0.01):
# #         pass

# #     @abstractmethod
# #     def get_state(self):
# #         pass

# #     def run_steps(self, n, dt=0.01):
# #         """Run the system for n steps and return trajectory of states."""
# #         self.reset()
# #         trajectory = []
# #         for _ in range(n):
# #             self.step(dt)
# #             trajectory.append(self.get_state())
# #         return trajectory

# #     def run_cyclical_steps(self, n, dt=0.01):
# #         self.reset()
# #         trajectory = []
    
# #         # Forward pass
# #         for _ in range(n):
# #             self.step(dt)
# #             trajectory.append(self.get_state())
    
# #         # Reverse velocities to simulate return path
# #         if hasattr(self, 'velocities'):
# #             self.velocities = -self.velocities
    
# #         # Backward pass
# #         for _ in range(n):
# #             self.step(dt)
# #             trajectory.append(self.get_state())
    
# #         return trajectory

# #     def get_config(self):
# #         """Return a dictionary of configuration parameters."""
# #         return {}


# # class SwirlingClusterSystem(MeasureDynamicalSystem):
# #     """
# #     2D rotation flow: particles on a circle rotating with angular velocity ω.
# #     Dynamics: dx/dt = ω * S * x, where S is the 90° rotation matrix.
# #     """
# #     def __init__(self, N, dim=2, noise_scale=0.0, seed=None):
# #         assert dim == 2, "SwirlingClusterSystem only supports 2D"
# #         self.N = N
# #         self.dim = dim
# #         self.noise_scale = noise_scale
# #         self.rng = np.random.default_rng(seed)
# #         self.seed = seed
        
# #         # Store initial configuration for LOT reference
# #         theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
# #         self.initial_positions = np.column_stack((np.cos(theta), np.sin(theta)))
# #         self.weights = np.full(self.N, 1.0 / self.N)
        
# #         self.reset()
    
# #     def reset(self):
# #         # Initialize N particles uniformly spaced on unit circle (DETERMINISTIC)
# #         self.positions = self.initial_positions.copy()
# #         self.history = []
    
# #     def step(self, dt=0.01, swirl_strength=10.0):
# #         x = self.positions
# #         v = np.stack((-x[:, 1], x[:, 0]), axis=1)  # 90° rotation: v = (-y, x)
        
# #         self.positions += (
# #             dt * swirl_strength * v
# #             + self.noise_scale * np.sqrt(dt) * self.rng.normal(size=x.shape)
# #         )
# #         self.history.append(self.get_state())
    
# #     def get_state(self):
# #         """Current particle positions."""
# #         return self.positions.copy()
    
# #     def get_measure(self):
# #         """(positions, weights) — what Sinkhorn/LOT expect."""
# #         return self.positions.copy(), self.weights.copy()
    
# #     def get_reference(self):
# #         """Natural LOT reference σ: use initial circular configuration μ₀."""
# #         return self.initial_positions.copy(), self.weights.copy()
    
# #     def get_config(self):
# #         return {
# #             'system': 'SwirlingClusterSystem',
# #             'N': self.N,
# #             'dim': self.dim,
# #             'noise_scale': self.noise_scale,
# #             'seed': self.seed
# #         }
    
# #     def get_cyclical_trajectory(self, n_rotations=2, dt=0.02, swirl_strength=10.0):
# #         """
# #         Generate trajectory completing n full rotations.
# #         Smooth looping - ends exactly where it started.
# #         """
# #         self.reset()
        
# #         # Angular velocity = swirl_strength (rad/unit time)
# #         # Steps for n rotations: n * 2π / (swirl_strength * dt)
# #         n_steps = int(n_rotations * 2 * np.pi / (swirl_strength * dt))
        
# #         trajectory = []
# #         for _ in range(n_steps):
# #             self.step(dt, swirl_strength=swirl_strength)
# #             trajectory.append(self.get_state())
        
# #         return np.array(trajectory)


# # class GeodesicTransportSimulator(MeasureDynamicalSystem):
# #     """
# #     W2 geodesic between two empirical measures with uniform weights.
# #     Interpolates along the Monge map T: x0_i -> x1_{perm[i]} found by Hungarian assignment.
# #     """

# #     def __init__(self, source_points, target_points):
# #         assert source_points.shape == target_points.shape, "Source and target must match in shape."
# #         self.x0 = np.asarray(source_points)
# #         self.x1 = np.asarray(target_points)
# #         self.N, self.dim = self.x0.shape

# #         # Compute a Monge map via assignment on squared distances (W2^2)
# #         M = ot.dist(self.x0, self.x1, metric="euclidean") ** 2
# #         row_ind, col_ind = linear_sum_assignment(M)
# #         self.perm = col_ind[np.argsort(row_ind)]
# #         self.mapped_targets = self.x1[self.perm]  # T(x0_i)

# #         # Uniform weights for the empirical measures
# #         self.weights = np.full(self.N, 1.0 / self.N)

# #         self.reset()

# #     def reset(self):
# #         self.t = 0.0
# #         self.direction = 1
# #         self.current_points = self.x0.copy()
# #         self.history = []

# #     def state_at(self, t):
# #         """Closed-form geodesic state at arbitrary t in [0,1]."""
# #         t = float(np.clip(t, 0.0, 1.0))
# #         return (1.0 - t) * self.x0 + t * self.mapped_targets

# #     def step(self, dt=0.01):
# #         """Ping-pong between t=0 and t=1 while recording states."""
# #         self.t += self.direction * dt
# #         if self.t >= 1.0:
# #             self.t = 1.0
# #             self.direction = -1
# #         elif self.t <= 0.0:
# #             self.t = 0.0
# #             self.direction = 1
# #         self.current_points = self.state_at(self.t)
# #         self.history.append((self.t, self.current_points.copy()))

# #     def get_state(self):
# #         """Positions only (backward-compatible)."""
# #         return self.current_points.copy()

# #     def get_measure(self):
# #         """(positions, weights) — what Sinkhorn/LOT expect."""
# #         return self.current_points.copy(), self.weights.copy()

# #     def get_reference(self):
# #         """Natural LOT reference σ: use μ0 support with uniform weights."""
# #         return self.x0.copy(), self.weights.copy()

# #     def displacement_map(self):
# #         """T(x) - x evaluated on σ=x0; used for LOT-velocity experiments."""
# #         return self.mapped_targets - self.x0

# #     def get_config(self):
# #         return {
# #             'system': 'GeodesicTransportSimulator',
# #             'N': self.N,
# #             'dim': self.dim,
# #             't': self.t
# #         }


# # class CuckerSmaleSystem(MeasureDynamicalSystem):
# #     """Cucker-Smale flocking dynamics."""
    
# #     def __init__(self, N, dim=2, beta=0.3, seed=None):
# #         self.N = N
# #         self.dim = dim
# #         self.beta = beta
# #         self.rng = np.random.default_rng(seed)
# #         self.seed = seed
# #         self.reset()

# #     def reset(self):
# #         self.positions = self.rng.random((self.N, self.dim)) + np.array([2.0] + [0.0] * (self.dim - 1))
# #         self.velocities = self.rng.normal(scale=3.0, size=(self.N, self.dim))
# #         self.history = []

# #     def step(self, dt=0.01):
# #         x = self.positions
# #         v = self.velocities

# #         diffs = x[:, None, :] - x[None, :, :]
# #         dist = np.linalg.norm(diffs, axis=-1) + 1e-8
# #         weights = 1.0 / (1.0 + dist**2) ** self.beta

# #         alignment = (weights[..., None] * (v[None, :, :] - v[:, None, :])).sum(axis=1)
# #         self.velocities += dt * alignment / self.N
# #         self.positions += dt * self.velocities
# #         self.history.append(self.get_state())

# #     def get_state(self):
# #         return self.positions.copy()

# #     def get_config(self):
# #         return {
# #             'system': 'CuckerSmaleSystem',
# #             'N': self.N,
# #             'dim': self.dim,
# #             'beta': self.beta,
# #             'seed': self.seed
# #         }

# #     def get_cyclical_trajectory(self, n_steps=250, dt=0.01):
# #         self.reset()
# #         trajectory = []
# #         for _ in range(n_steps):
# #             self.step(dt)
# #             trajectory.append(self.get_state())
# #         return np.array(trajectory + trajectory[::-1][1:-1])


# # class GaussianMixtureFlowSystem(MeasureDynamicalSystem):
# #     """Lorenz-driven Gaussian mixture model."""
    
# #     def __init__(self, N, K=3, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
# #         self.N = N
# #         self.K = K
# #         self.dim = dim
# #         self.dt = dt
# #         self.rng = np.random.default_rng(seed)
# #         self.seed = seed
# #         self.max_clip = max_clip
# #         self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
# #         self.cov_scale = cov_scale
# #         self.reset()
# #         self.means_trajectory = []

# #     def lorenz_deriv(self, mean):
# #         x, y, z = mean
# #         s = self.lorenz_params['sigma']
# #         r = self.lorenz_params['rho']
# #         b = self.lorenz_params['beta']
# #         dx = s * (y - x)
# #         dy = x * (r - z) - y
# #         dz = x * y - b * z
# #         return np.array([dx, dy, dz])

# #     def reset(self):
# #         self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
# #         self.cov = np.eye(3) * self.cov_scale
# #         self.current_points = np.zeros((self.N, self.dim))
# #         self.history = []
# #         self.means_trajectory = []

# #     def step(self, dt=None):
# #         dt = dt or self.dt
# #         derivs = np.array([self.lorenz_deriv(m) for m in self.means])
# #         derivs = np.nan_to_num(derivs, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)
        
# #         self.means += dt * derivs
# #         self.means = np.clip(self.means, -self.max_clip, self.max_clip)
# #         self.means_trajectory.append(self.means.copy())
    
# #         self.assignments = self.rng.integers(self.K, size=self.N)
# #         points = np.array([self.rng.multivariate_normal(self.means[k], self.cov)
# #                            for k in self.assignments])
# #         points += self.rng.normal(scale=0.02, size=points.shape)
    
# #         self.current_points = points
# #         self.history.append(self.get_state())

# #     def get_state(self):
# #         return self.current_points.copy()

# #     def get_config(self):
# #         return {
# #             'system': 'GaussianMixtureFlowSystem',
# #             'N': self.N,
# #             'K': self.K,
# #             'dim': self.dim,
# #             'dt': self.dt,
# #             'lorenz_params': self.lorenz_params,
# #             'seed': self.seed
# #         }


# # def add_posthoc_noise(trajectory, noise_scale=0.01, seed=None):
# #     """Add Gaussian noise to trajectory."""
# #     if seed is not None:
# #         np.random.seed(seed)
# #     noise = np.random.normal(scale=noise_scale, size=trajectory.shape)
# #     return trajectory + noise



# # # """
# # # Measure-valued dynamical systems for LOT+RC experiments.

# # # Classes:
# # #     - SwirlingClusterSystem: 2D rotation flow on a circle
# # #     - GeodesicTransportSimulator: W2 geodesic between two empirical measures
# # #     - CuckerSmaleSystem: Flocking dynamics
# # #     - GaussianMixtureFlowSystem: Lorenz-driven GMM
# # # """

# # # from abc import ABC, abstractmethod
# # # import numpy as np
# # # import ot
# # # from scipy.optimize import linear_sum_assignment


# # # class MeasureDynamicalSystem(ABC):
# # #     """
# # #     Abstract base class for measure-valued dynamical systems.
# # #     """
# # #     @abstractmethod
# # #     def reset(self):
# # #         pass

# # #     @abstractmethod
# # #     def step(self, dt=0.01):
# # #         pass

# # #     @abstractmethod
# # #     def get_state(self):
# # #         pass

# # #     def run_steps(self, n, dt=0.01):
# # #         """Run the system for n steps and return trajectory of states."""
# # #         self.reset()
# # #         trajectory = []
# # #         for _ in range(n):
# # #             self.step(dt)
# # #             trajectory.append(self.get_state())
# # #         return trajectory

# # #     def run_cyclical_steps(self, n, dt=0.01):
# # #         self.reset()
# # #         trajectory = []
    
# # #         # Forward pass
# # #         for _ in range(n):
# # #             self.step(dt)
# # #             trajectory.append(self.get_state())
    
# # #         # Reverse velocities to simulate return path
# # #         if hasattr(self, 'velocities'):
# # #             self.velocities = -self.velocities
    
# # #         # Backward pass
# # #         for _ in range(n):
# # #             self.step(dt)
# # #             trajectory.append(self.get_state())
    
# # #         return trajectory

# # #     def get_config(self):
# # #         """Return a dictionary of configuration parameters."""
# # #         return {}


# # # class SwirlingClusterSystem(MeasureDynamicalSystem):
# # #     """
# # #     2D rotation flow: particles on a circle rotating with angular velocity ω.
# # #     Dynamics: dx/dt = ω * S * x, where S is the 90° rotation matrix.
# # #     """
# # #     def __init__(self, N, dim=2, noise_scale=0.0, seed=None):
# # #         assert dim == 2, "SwirlingClusterSystem only supports 2D"
# # #         self.N = N
# # #         self.dim = dim
# # #         self.noise_scale = noise_scale
# # #         self.rng = np.random.default_rng(seed)
# # #         self.seed = seed
        
# # #         # Store initial configuration for LOT reference
# # #         theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
# # #         self.initial_positions = np.column_stack((np.cos(theta), np.sin(theta)))
# # #         self.weights = np.full(self.N, 1.0 / self.N)
        
# # #         self.reset()
    
# # #     def reset(self):
# # #         # Initialize N particles uniformly spaced on unit circle (DETERMINISTIC)
# # #         self.positions = self.initial_positions.copy()
# # #         self.history = []
    
# # #     def step(self, dt=0.01, swirl_strength=10.0):
# # #         x = self.positions
# # #         v = np.stack((-x[:, 1], x[:, 0]), axis=1)  # 90° rotation: v = (-y, x)
        
# # #         self.positions += (
# # #             dt * swirl_strength * v
# # #             + self.noise_scale * np.sqrt(dt) * self.rng.normal(size=x.shape)
# # #         )
# # #         self.history.append(self.get_state())
    
# # #     def get_state(self):
# # #         """Current particle positions."""
# # #         return self.positions.copy()
    
# # #     def get_measure(self):
# # #         """(positions, weights) — what Sinkhorn/LOT expect."""
# # #         return self.positions.copy(), self.weights.copy()
    
# # #     def get_reference(self):
# # #         """Natural LOT reference σ: use initial circular configuration μ₀."""
# # #         return self.initial_positions.copy(), self.weights.copy()
    
# # #     def get_config(self):
# # #         return {
# # #             'system': 'SwirlingClusterSystem',
# # #             'N': self.N,
# # #             'dim': self.dim,
# # #             'noise_scale': self.noise_scale,
# # #             'seed': self.seed
# # #         }
    
# # #     def get_cyclical_trajectory(self, n_steps=300, dt=0.02):
# # #         """
# # #         Generate continuous rotation trajectory.
# # #         No mirroring needed - rotation naturally cycles.
# # #         """
# # #         self.reset()
# # #         trajectory = []
# # #         for _ in range(n_steps):
# # #             self.step(dt)
# # #             trajectory.append(self.get_state())
        
# # #         return np.array(trajectory)


# # # class GeodesicTransportSimulator(MeasureDynamicalSystem):
# # #     """
# # #     W2 geodesic between two empirical measures with uniform weights.
# # #     Interpolates along the Monge map T: x0_i -> x1_{perm[i]} found by Hungarian assignment.
# # #     """

# # #     def __init__(self, source_points, target_points):
# # #         assert source_points.shape == target_points.shape, "Source and target must match in shape."
# # #         self.x0 = np.asarray(source_points)
# # #         self.x1 = np.asarray(target_points)
# # #         self.N, self.dim = self.x0.shape

# # #         # Compute a Monge map via assignment on squared distances (W2^2)
# # #         M = ot.dist(self.x0, self.x1, metric="euclidean") ** 2
# # #         row_ind, col_ind = linear_sum_assignment(M)
# # #         self.perm = col_ind[np.argsort(row_ind)]
# # #         self.mapped_targets = self.x1[self.perm]  # T(x0_i)

# # #         # Uniform weights for the empirical measures
# # #         self.weights = np.full(self.N, 1.0 / self.N)

# # #         self.reset()

# # #     def reset(self):
# # #         self.t = 0.0
# # #         self.direction = 1
# # #         self.current_points = self.x0.copy()
# # #         self.history = []

# # #     def state_at(self, t):
# # #         """Closed-form geodesic state at arbitrary t in [0,1]."""
# # #         t = float(np.clip(t, 0.0, 1.0))
# # #         return (1.0 - t) * self.x0 + t * self.mapped_targets

# # #     def step(self, dt=0.01):
# # #         """Ping-pong between t=0 and t=1 while recording states."""
# # #         self.t += self.direction * dt
# # #         if self.t >= 1.0:
# # #             self.t = 1.0
# # #             self.direction = -1
# # #         elif self.t <= 0.0:
# # #             self.t = 0.0
# # #             self.direction = 1
# # #         self.current_points = self.state_at(self.t)
# # #         self.history.append((self.t, self.current_points.copy()))

# # #     def get_state(self):
# # #         """Positions only (backward-compatible)."""
# # #         return self.current_points.copy()

# # #     def get_measure(self):
# # #         """(positions, weights) — what Sinkhorn/LOT expect."""
# # #         return self.current_points.copy(), self.weights.copy()

# # #     def get_reference(self):
# # #         """Natural LOT reference σ: use μ0 support with uniform weights."""
# # #         return self.x0.copy(), self.weights.copy()

# # #     def displacement_map(self):
# # #         """T(x) - x evaluated on σ=x0; used for LOT-velocity experiments."""
# # #         return self.mapped_targets - self.x0

# # #     def get_config(self):
# # #         return {
# # #             'system': 'GeodesicTransportSimulator',
# # #             'N': self.N,
# # #             'dim': self.dim,
# # #             't': self.t
# # #         }


# # # class CuckerSmaleSystem(MeasureDynamicalSystem):
# # #     """Cucker-Smale flocking dynamics."""
    
# # #     def __init__(self, N, dim=2, beta=0.3, seed=None):
# # #         self.N = N
# # #         self.dim = dim
# # #         self.beta = beta
# # #         self.rng = np.random.default_rng(seed)
# # #         self.seed = seed
# # #         self.reset()

# # #     def reset(self):
# # #         self.positions = self.rng.random((self.N, self.dim)) + np.array([2.0] + [0.0] * (self.dim - 1))
# # #         self.velocities = self.rng.normal(scale=3.0, size=(self.N, self.dim))
# # #         self.history = []

# # #     def step(self, dt=0.01):
# # #         x = self.positions
# # #         v = self.velocities

# # #         diffs = x[:, None, :] - x[None, :, :]
# # #         dist = np.linalg.norm(diffs, axis=-1) + 1e-8
# # #         weights = 1.0 / (1.0 + dist**2) ** self.beta

# # #         alignment = (weights[..., None] * (v[None, :, :] - v[:, None, :])).sum(axis=1)
# # #         self.velocities += dt * alignment / self.N
# # #         self.positions += dt * self.velocities
# # #         self.history.append(self.get_state())

# # #     def get_state(self):
# # #         return self.positions.copy()

# # #     def get_config(self):
# # #         return {
# # #             'system': 'CuckerSmaleSystem',
# # #             'N': self.N,
# # #             'dim': self.dim,
# # #             'beta': self.beta,
# # #             'seed': self.seed
# # #         }

# # #     def get_cyclical_trajectory(self, n_steps=250, dt=0.01):
# # #         self.reset()
# # #         trajectory = []
# # #         for _ in range(n_steps):
# # #             self.step(dt)
# # #             trajectory.append(self.get_state())
# # #         return np.array(trajectory + trajectory[::-1][1:-1])


# # # class GaussianMixtureFlowSystem(MeasureDynamicalSystem):
# # #     """Lorenz-driven Gaussian mixture model."""
    
# # #     def __init__(self, N, K=3, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
# # #         self.N = N
# # #         self.K = K
# # #         self.dim = dim
# # #         self.dt = dt
# # #         self.rng = np.random.default_rng(seed)
# # #         self.seed = seed
# # #         self.max_clip = max_clip
# # #         self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
# # #         self.cov_scale = cov_scale
# # #         self.reset()
# # #         self.means_trajectory = []

# # #     def lorenz_deriv(self, mean):
# # #         x, y, z = mean
# # #         s = self.lorenz_params['sigma']
# # #         r = self.lorenz_params['rho']
# # #         b = self.lorenz_params['beta']
# # #         dx = s * (y - x)
# # #         dy = x * (r - z) - y
# # #         dz = x * y - b * z
# # #         return np.array([dx, dy, dz])

# # #     def reset(self):
# # #         self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
# # #         self.cov = np.eye(3) * self.cov_scale
# # #         self.current_points = np.zeros((self.N, self.dim))
# # #         self.history = []
# # #         self.means_trajectory = []

# # #     def step(self, dt=None):
# # #         dt = dt or self.dt
# # #         derivs = np.array([self.lorenz_deriv(m) for m in self.means])
# # #         derivs = np.nan_to_num(derivs, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)
        
# # #         self.means += dt * derivs
# # #         self.means = np.clip(self.means, -self.max_clip, self.max_clip)
# # #         self.means_trajectory.append(self.means.copy())
    
# # #         self.assignments = self.rng.integers(self.K, size=self.N)
# # #         points = np.array([self.rng.multivariate_normal(self.means[k], self.cov)
# # #                            for k in self.assignments])
# # #         points += self.rng.normal(scale=0.02, size=points.shape)
    
# # #         self.current_points = points
# # #         self.history.append(self.get_state())

# # #     def get_state(self):
# # #         return self.current_points.copy()

# # #     def get_config(self):
# # #         return {
# # #             'system': 'GaussianMixtureFlowSystem',
# # #             'N': self.N,
# # #             'K': self.K,
# # #             'dim': self.dim,
# # #             'dt': self.dt,
# # #             'lorenz_params': self.lorenz_params,
# # #             'seed': self.seed
# # #         }


# # # def add_posthoc_noise(trajectory, noise_scale=0.01, seed=None):
# # #     """Add Gaussian noise to trajectory."""
# # #     if seed is not None:
# # #         np.random.seed(seed)
# # #     noise = np.random.normal(scale=noise_scale, size=trajectory.shape)
# # #     return trajectory + noise



    


# # # # from abc import ABC, abstractmethod
# # # # import numpy as np
# # # # import ot 
# # # # # from scipy.optimize import linear_sum_assignment

# # # # class MeasureDynamicalSystem(ABC):
# # # #     """
# # # #     Abstract base class for measure-valued dynamical systems.
# # # #     """
# # # #     @abstractmethod
# # # #     def reset(self):
# # # #         pass

# # # #     @abstractmethod
# # # #     def step(self, dt=0.01):
# # # #         pass

# # # #     @abstractmethod
# # # #     def get_state(self):
# # # #         pass

# # # #     def run_steps(self, n, dt=0.01):
# # # #         """
# # # #         Run the system for n steps and return trajectory of states.
# # # #         """
# # # #         self.reset()
# # # #         trajectory = []
# # # #         for _ in range(n):
# # # #             self.step(dt)
# # # #             trajectory.append(self.get_state())
# # # #         return trajectory

# # # #     def run_cyclical_steps(self, n, dt=0.01):
# # # #         self.reset()
# # # #         trajectory = []
    
# # # #         # Forward pass
# # # #         for _ in range(n):
# # # #             self.step(dt)
# # # #             trajectory.append(self.get_state())
    
# # # #         # Reverse velocities to simulate return path
# # # #         if hasattr(self, 'velocities'):
# # # #             self.velocities = -self.velocities
    
# # # #         # Backward pass
# # # #         for _ in range(n):
# # # #             self.step(dt)
# # # #             trajectory.append(self.get_state())
    
# # # #         return trajectory

# # # #     def get_config(self):
# # # #         """
# # # #         Return a dictionary of configuration parameters.
# # # #         Override in subclass.
# # # #         """
# # # #         return {}

        

# # # # class CuckerSmaleSystem(MeasureDynamicalSystem):
# # # #     def __init__(self, N, dim=2, beta=0.3, seed=None):
# # # #         self.N = N
# # # #         self.dim = dim
# # # #         self.beta = beta
# # # #         self.rng = np.random.default_rng(seed)
# # # #         self.seed = seed
# # # #         self.reset()

# # # #     def reset(self):
# # # #         # Uniform positions in [0, 1], then shift center +2 along x-axis
# # # #         self.positions = self.rng.random((self.N, self.dim)) + np.array([2.0] + [0.0] * (self.dim - 1))

# # # #         # Gaussian initial velocities
# # # #         self.velocities = self.rng.normal(scale=3.0, size=(self.N, self.dim))
# # # #         self.history = []

# # # #     def step(self, dt=0.01):
# # # #         x = self.positions
# # # #         v = self.velocities

# # # #         # Pairwise differences
# # # #         diffs = x[:, None, :] - x[None, :, :]
# # # #         dist = np.linalg.norm(diffs, axis=-1) + 1e-8

# # # #         # Influence weight matrix
# # # #         # weights = 1.0 / (dist ** self.beta)
# # # #         weights = 1.0 / (1.0 + dist**2) ** self.beta

# # # #         # Alignment force computation
# # # #         alignment = (weights[..., None] * (v[None, :, :] - v[:, None, :])).sum(axis=1)
# # # #         self.velocities += dt * alignment / self.N
# # # #         self.positions += dt * self.velocities
# # # #         self.history.append(self.get_state())

# # # #     def get_state(self):
# # # #         return self.positions.copy()

# # # #     def get_config(self):
# # # #         return {
# # # #             'system': 'CuckerSmaleSystem',
# # # #             'N': self.N,
# # # #             'dim': self.dim,
# # # #             'beta': self.beta,
# # # #             'seed': self.seed
# # # #         }

# # # #    # Mirror the trajectory rather than simulate reverse dynamics
# # # #     def get_cyclical_trajectory(self, n_steps=250, dt=0.01):
# # # #         """
# # # #         Return a mirrored cyclic trajectory as a NumPy array.
# # # #         Equivalent to: forward trajectory + reversed path (excluding midpoint).
# # # #         """
# # # #         self.reset()
# # # #         trajectory = []
# # # #         for _ in range(n_steps):
# # # #             self.step(dt)
# # # #             trajectory.append(self.get_state())
# # # #         return np.array(trajectory + trajectory[::-1][1:-1])



# # # # # make covariance smaller (ie 0.2)
# # # # # make dt small enough to see where it goes -- needs to be 3D
# # # # class GaussianMixtureFlowSystem(MeasureDynamicalSystem):
# # # #     def __init__(self, N, K=3, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
# # # #         self.N = N
# # # #         self.K = K
# # # #         self.dim = dim
# # # #         self.dt = dt
# # # #         self.rng = np.random.default_rng(seed)
# # # #         self.seed = seed
# # # #         self.max_clip = max_clip
# # # #         self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
# # # #         self.cov_scale = cov_scale
# # # #         self.reset()
# # # #         self.means_trajectory = []  # initialize trajectory tracking

# # # #     def lorenz_deriv(self, mean):
# # # #         x, y, z = mean
# # # #         s = self.lorenz_params['sigma']
# # # #         r = self.lorenz_params['rho']
# # # #         b = self.lorenz_params['beta']
# # # #         dx = s * (y - x)
# # # #         dy = x * (r - z) - y
# # # #         dz = x * y - b * z
# # # #         return np.array([dx, dy, dz])

# # # #     def reset(self):
# # # #         # Sample K initial mean vectors in 3D, spaced out along the x-axis
# # # #         self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
        
# # # #         self.cov = np.eye(3) * self.cov_scale
       
# # # #         self.current_points = np.zeros((self.N, self.dim))
# # # #         self.history = []
# # # #         self.means_trajectory = []  # clear trajectory on reset

# # # #     def step(self, dt=None):
# # # #         dt = dt or self.dt
# # # #         derivs = np.array([self.lorenz_deriv(m) for m in self.means])  # shape: (K, 3)
# # # #         derivs = np.nan_to_num(derivs, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)
        
# # # #         # Update each mean separately + optional jitter
# # # #         self.means += dt * derivs
# # # #         # self.means += self.rng.normal(scale=0.05, size=self.means.shape)
# # # #         self.means = np.clip(self.means, -self.max_clip, self.max_clip)
    
# # # #         self.means_trajectory.append(self.means.copy())
    
# # # #         # Assign each of the N points to one of the K means
# # # #         self.assignments = self.rng.integers(self.K, size=self.N)
# # # #         points = np.array([self.rng.multivariate_normal(self.means[k], self.cov)
# # # #                            for k in self.assignments])
# # # #         points += self.rng.normal(scale=0.02, size=points.shape)
    
# # # #         self.current_points = points
# # # #         self.history.append(self.get_state())

# # # #     def get_state(self):
# # # #         return self.current_points.copy()

# # # #     def get_config(self):
# # # #         return {
# # # #             'system': 'GaussianMixtureFlowSystem',
# # # #             'N': self.N,
# # # #             'K': self.K,
# # # #             'dim': self.dim,
# # # #             'dt': self.dt,
# # # #             'lorenz_params': self.lorenz_params,
# # # #             'seed': self.seed
# # # #         }


# # # # # Includes in class jitter
# # # # class OneGaussianMixtureFlowSystem(MeasureDynamicalSystem):
# # # #     def __init__(self, N, K=1, dim=3, lorenz_params=None, dt=0.01, seed=None, max_clip=100.0, cov_scale=1.5):
# # # #         self.N = N
# # # #         self.K = K
# # # #         self.dim = dim
# # # #         self.dt = dt
# # # #         self.rng = np.random.default_rng(seed)
# # # #         self.seed = seed
# # # #         self.max_clip = max_clip
# # # #         self.lorenz_params = lorenz_params or {'sigma': 10.0, 'rho': 28.0, 'beta': 8/3}
# # # #         self.cov_scale = cov_scale
# # # #         self.reset()
# # # #         self.means_trajectory = []  # initialize trajectory tracking

# # # #     def lorenz_deriv(self, mean):
# # # #         x, y, z = mean
# # # #         s = self.lorenz_params['sigma']
# # # #         r = self.lorenz_params['rho']
# # # #         b = self.lorenz_params['beta']
# # # #         dx = s * (y - x)
# # # #         dy = x * (r - z) - y
# # # #         dz = x * y - b * z
# # # #         return np.array([dx, dy, dz])

# # # #     def reset(self):
# # # #         # Sample K initial mean vectors in 3D, spaced out along the x-axis
# # # #         self.means = self.rng.normal(loc=np.linspace(-3, 3, self.K)[:, None], scale=1.0, size=(self.K, 3))
        
# # # #         self.cov = np.eye(3) * self.cov_scale
       
# # # #         self.current_points = np.zeros((self.N, self.dim))
# # # #         self.history = []
# # # #         self.means_trajectory = []  # clear trajectory on reset

# # # #     def step(self, dt=None):
# # # #         dt = dt or self.dt
# # # #         deriv = self.lorenz_deriv(self.means[0])  # shape (3,)
# # # #         deriv = np.nan_to_num(deriv, nan=0.0, posinf=self.max_clip, neginf=-self.max_clip)
# # # #         self.means[0] += dt * deriv 
# # # #         self.means[0] += self.rng.normal(scale=0.05, size=(3,))  # jitter
# # # #         self.means = np.clip(self.means, -self.max_clip, self.max_clip)
    
# # # #         self.means_trajectory.append(self.means.copy())
    
# # # #         # Generate all N points from the single Gaussian component
# # # #         self.current_points = self.rng.multivariate_normal(self.means[0], self.cov, size=self.N)
# # # #         self.current_points += self.rng.normal(scale=0.02, size=self.current_points.shape)
# # # #         self.history.append(self.get_state())

# # # #         #take all the points, drive them forward, then this gives new state for all points, then take average (avgs should make it less), create point cloud around this


# # # #     def get_state(self):
# # # #         return self.current_points.copy()

# # # #     def get_config(self):
# # # #         return {
# # # #             'system': 'GaussianMixtureFlowSystem',
# # # #             'N': self.N,
# # # #             'K': self.K,
# # # #             'dim': self.dim,
# # # #             'dt': self.dt,
# # # #             'lorenz_params': self.lorenz_params,
# # # #             'seed': self.seed
# # # #         } 


# # # # class SwirlingClusterSystem(MeasureDynamicalSystem):
# # # #     """
# # # #     2D rotation flow: particles on a circle rotating with angular velocity ω.
# # # #     Dynamics: dx/dt = ω * S * x, where S is the 90° rotation matrix.
# # # #     """
# # # #     def __init__(self, N, dim=2, noise_scale=0.0, seed=None):
# # # #         assert dim == 2, "SwirlingClusterSystem only supports 2D"
# # # #         self.N = N
# # # #         self.dim = dim
# # # #         self.noise_scale = noise_scale
# # # #         self.rng = np.random.default_rng(seed)
# # # #         self.seed = seed
        
# # # #         # Store initial configuration for LOT reference
# # # #         theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
# # # #         self.initial_positions = np.column_stack((np.cos(theta), np.sin(theta)))
# # # #         self.weights = np.full(self.N, 1.0 / self.N)
        
# # # #         self.reset()
    
# # # #     def reset(self):
# # # #         # Initialize N particles uniformly spaced on unit circle (DETERMINISTIC)
# # # #         self.positions = self.initial_positions.copy()
# # # #         self.history = []
    
# # # #     def step(self, dt=0.01, swirl_strength=10.0):
# # # #         x = self.positions
# # # #         v = np.stack((-x[:, 1], x[:, 0]), axis=1)  # 90° rotation: v = (-y, x)
        
# # # #         self.positions += (
# # # #             dt * swirl_strength * v
# # # #             + self.noise_scale * np.sqrt(dt) * self.rng.normal(size=x.shape)
# # # #         )
# # # #         self.history.append(self.get_state())
    
# # # #     def get_state(self):
# # # #         """Current particle positions."""
# # # #         return self.positions.copy()
    
# # # #     def get_measure(self):
# # # #         """(positions, weights) — what Sinkhorn/LOT expect."""
# # # #         return self.positions.copy(), self.weights.copy()
    
# # # #     def get_reference(self):
# # # #         """Natural LOT reference σ: use initial circular configuration μ₀."""
# # # #         return self.initial_positions.copy(), self.weights.copy()
    
# # # #     def get_config(self):
# # # #         return {
# # # #             'system': 'SwirlingClusterSystem',
# # # #             'N': self.N,
# # # #             'dim': self.dim,
# # # #             'noise_scale': self.noise_scale,
# # # #             'seed': self.seed
# # # #         }
    
# # # #     def get_cyclical_trajectory(self, n_steps=300, dt=0.02):
# # # #         self.reset()
# # # #         trajectory = []
# # # #         for _ in range(n_steps):
# # # #             self.step(dt)
# # # #             trajectory.append(self.get_state())
        
# # # #         # Mirror for smooth cyclicity
# # # #         return np.array(trajectory + trajectory[::-1][1:-1])



# # # # from scipy.optimize import linear_sum_assignment

# # # # class GeodesicTransportSimulator(MeasureDynamicalSystem):
# # # #     """
# # # #     W2 geodesic between two empirical measures with uniform weights.
# # # #     Interpolates along the Monge map T: x0_i -> x1_{perm[i]} found by Hungarian assignment.
# # # #     """

# # # #     def __init__(self, source_points, target_points):
# # # #         assert source_points.shape == target_points.shape, "Source and target must match in shape."
# # # #         self.x0 = np.asarray(source_points)
# # # #         self.x1 = np.asarray(target_points)
# # # #         self.N, self.dim = self.x0.shape

# # # #         # --- Compute a Monge map via assignment on squared distances (W2^2) ---
# # # #         M = ot.dist(self.x0, self.x1, metric="euclidean") ** 2  # squared cost for W2
# # # #         row_ind, col_ind = linear_sum_assignment(M)
# # # #         # row_ind will be [0,1,...,N-1] if M is dense; keep generality:
# # # #         self.perm = col_ind[np.argsort(row_ind)]
# # # #         self.mapped_targets = self.x1[self.perm]                 # T(x0_i)

# # # #         # Uniform weights for the empirical measures
# # # #         self.weights = np.full(self.N, 1.0 / self.N)

# # # #         self.reset()

# # # #     # --------- Lifecycle ----------
# # # #     def reset(self):
# # # #         self.t = 0.0
# # # #         self.direction = 1
# # # #         self.current_points = self.x0.copy()
# # # #         self.history = []   # list of (t, points)

# # # #     def state_at(self, t):
# # # #         """Closed-form geodesic state at arbitrary t in [0,1]."""
# # # #         t = float(np.clip(t, 0.0, 1.0))
# # # #         return (1.0 - t) * self.x0 + t * self.mapped_targets

# # # #     def step(self, dt=0.01):
# # # #         """Ping-pong between t=0 and t=1 while recording states."""
# # # #         self.t += self.direction * dt
# # # #         if self.t >= 1.0:
# # # #             self.t = 1.0
# # # #             self.direction = -1
# # # #         elif self.t <= 0.0:
# # # #             self.t = 0.0
# # # #             self.direction = 1
# # # #         self.current_points = self.state_at(self.t)
# # # #         self.history.append((self.t, self.current_points.copy()))

# # # #     # --------- Interfaces for downstream code ----------
# # # #     def get_state(self):
# # # #         """Positions only (backward-compatible)."""
# # # #         return self.current_points.copy()

# # # #     def get_measure(self):
# # # #         """(positions, weights) — what Sinkhorn/LOT expect."""
# # # #         return self.current_points.copy(), self.weights.copy()

# # # #     def get_reference(self):
# # # #         """Natural LOT reference σ: use μ0 support with uniform weights."""
# # # #         return self.x0.copy(), self.weights.copy()

# # # #     def displacement_map(self):
# # # #         """T(x) - x evaluated on σ=x0; used for LOT-velocity experiments."""
# # # #         return self.mapped_targets - self.x0

# # # #     def get_config(self):
# # # #         return {
# # # #             'system': 'GeodesicTransportSimulator',
# # # #             'N': self.N,
# # # #             'dim': self.dim,
# # # #             't': self.t
# # # #         }


# # # # def add_posthoc_noise(trajectory, noise_scale=0.01, seed=None):
# # # #     if seed is not None:
# # # #         np.random.seed(seed)
# # # #     noise = np.random.normal(scale=noise_scale, size=trajectory.shape)
# # # #     return trajectory + noise



