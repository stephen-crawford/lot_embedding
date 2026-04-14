# #rc_computer.py

# import numpy as np
# import os
# import json
# from scipy.stats import skew, kurtosis
# from abc import ABC, abstractmethod


# # === Optional Activations ===
# def relu(x):
#     return np.maximum(0, x)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # -----------------------
# # Dynamical Systems Classes
# # -----------------------
# class DynamicalSystem(ABC):
#     @abstractmethod
#     def simulate(self, timesteps, initial_condition, **kwargs):
#         pass

# class LogisticMap(DynamicalSystem):
#     def __init__(self, r=3.7):
#         self.r = r
#     def simulate(self, timesteps, initial_condition, **kwargs):
#         x = np.zeros(timesteps)
#         x[0] = initial_condition
#         for t in range(1, timesteps):
#             x[t] = self.r * x[t-1] * (1 - x[t-1])
#         return x.reshape(-1, 1)

# class LorenzAttractor(DynamicalSystem):
#     def __init__(self, sigma=10.0, beta=8.0/3.0, rho=28.0, dt=0.010):
#         self.sigma = sigma
#         self.beta = beta
#         self.rho = rho
#         self.dt = dt
        
#     def simulate(self, timesteps, initial_condition, **kwargs):
#         state = np.zeros((timesteps, 3))
#         state[0] = initial_condition
#         for t in range(1, timesteps):
#             x, y, z = state[t-1]
#             dx = self.sigma * (y - x)
#             dy = x * (self.rho - z) - y
#             dz = x * y - self.beta * z
#             state[t, 0] = x + dx * self.dt
#             state[t, 1] = y + dy * self.dt
#             state[t, 2] = z + dz * self.dt
#         return state

# # === Reservoir Computing Class ===
# class ReservoirComputer:
#     def __init__(self, input_size, reservoir_size, output_size,
#                  spectral_radius=0.9, input_scaling=0.1, leak_rate=1.0,
#                  ridge_param=1e-6, activation=np.tanh, random_seed=None,
#                  init_scheme="uniform", sparsity=0.1, avg_in_degree=5, bias_range=0.0):

#         self.input_size = input_size
#         self.reservoir_size = reservoir_size
#         self.output_size = output_size
#         self.spectral_radius = spectral_radius
#         self.input_scaling = input_scaling
#         self.leak_rate = leak_rate
#         self.ridge_param = ridge_param
#         self.activation = activation
#         self.init_scheme = init_scheme

#         if random_seed is not None:
#             np.random.seed(random_seed)
#             self.random_seed = random_seed
#         else:
#             self.random_seed = None

#         self._initialize_weights(sparsity, avg_in_degree, bias_range)
#         self.reset_state()

#         self.config = {
#             "input_size": input_size,
#             "reservoir_size": reservoir_size,
#             "output_size": output_size,
#             "spectral_radius": spectral_radius,
#             "input_scaling": input_scaling,
#             "leak_rate": leak_rate,
#             "ridge_param": ridge_param,
#             "activation": activation.__name__ if hasattr(activation, '__name__') else str(activation),
#             "random_seed": self.random_seed,
#             "init_scheme": init_scheme,
#             "sparsity": sparsity,
#             "avg_in_degree": avg_in_degree,
#             "bias_range": bias_range
#         }

#     def _initialize_weights(self, sparsity, avg_in_degree, bias_range):
#         # === Input weights ===
#         if self.init_scheme == "paper":
#             self.W_in = np.zeros((self.reservoir_size, self.input_size))
#             for i in range(self.reservoir_size):
#                 idx = np.random.randint(0, self.input_size)
#                 self.W_in[i, idx] = np.random.uniform(-self.input_scaling, self.input_scaling)
#             self.bias = np.random.uniform(-bias_range, bias_range, (self.reservoir_size, 1)) if bias_range > 0 else np.zeros((self.reservoir_size, 1))
#         elif self.init_scheme in ["normal", "sparse"]:
#             self.W_in = np.random.randn(self.reservoir_size, self.input_size) * self.input_scaling
#             self.bias = np.zeros((self.reservoir_size, 1))
#         else:  # uniform
#             self.W_in = np.random.uniform(-1, 1, (self.reservoir_size, self.input_size)) * self.input_scaling
#             self.bias = np.zeros((self.reservoir_size, 1))

#         # === Recurrent weights ===
#         if self.init_scheme == "normal":
#             W = np.random.randn(self.reservoir_size, self.reservoir_size)
#         elif self.init_scheme == "sparse":
#             W = np.random.randn(self.reservoir_size, self.reservoir_size)
#             mask = np.random.rand(self.reservoir_size, self.reservoir_size) < sparsity
#             W *= mask
#         elif self.init_scheme == "paper":
#             W = np.zeros((self.reservoir_size, self.reservoir_size))
#             for i in range(self.reservoir_size):
#                 indices = np.random.choice(self.reservoir_size, size=avg_in_degree, replace=False)
#                 W[i, indices] = np.random.uniform(-1, 1, size=avg_in_degree)
#         else:
#             W = np.random.uniform(-1, 1, (self.reservoir_size, self.reservoir_size))

#         # Spectral normalization
#         eigs = np.linalg.eigvals(W)
#         self.W = W * (self.spectral_radius / np.max(np.abs(eigs)))

#     def reset_state(self):
#         self.state = np.zeros((self.reservoir_size, 1))
#         self.reservoir_statistics = []

#     def compute_state_statistics(self, state):
#         s = state.flatten()
#         return {
#             "mean": float(np.mean(s)),
#             "variance": float(np.var(s)),
#             "sparsity": float(np.mean(np.abs(s) < 0.01)),
#             "skewness": float(skew(s)),
#             "kurtosis": float(kurtosis(s))
#         }

#     def _update(self, u):
#         pre_act = self.W @ self.state + self.W_in @ u
#         if self.init_scheme == "paper":
#             pre_act += self.bias
#         activated = self.activation(pre_act)
#         self.state = (1 - self.leak_rate) * self.state + self.leak_rate * activated
#         self.reservoir_statistics.append(self.compute_state_statistics(self.state))
#         return self.state

#     def run(self, input_sequence):
#         T = input_sequence.shape[0]
#         states = np.zeros((T, self.reservoir_size))
#         self.reset_state()
#         for t in range(T):
#             u = input_sequence[t].reshape(-1, 1)
#             state = self._update(u)
#             states[t] = state[:, 0]
#         return states

#     def train(self, states, targets):
#         X = np.hstack([states, np.ones((states.shape[0], 1))])
#         Y = targets
#         ridge = self.ridge_param * np.eye(X.shape[1])
#         self.W_out = np.linalg.solve(X.T @ X + ridge, X.T @ Y)

#     def predict(self, states):
#         if self.W_out is None:
#             raise ValueError("Readout weights not trained.")
#         X = np.hstack([states, np.ones((states.shape[0], 1))])
#         return X @ self.W_out

#     def run_autonomous(self, warm_start_input, autonomous_steps):
#         self.reset_state()
#         states = []
#         predictions = []

#         # Warm-up
#         for t in range(warm_start_input.shape[0]):
#             u = warm_start_input[t].reshape(-1, 1)
#             state = self._update(u)
#             states.append(state[:, 0])

#         # Autonomous generation
#         for _ in range(autonomous_steps):
#             extended = np.vstack([self.state, [[1]]])
#             pred = (self.W_out.T @ extended).flatten()
#             predictions.append(pred)
#             self._update(pred.reshape(-1, 1))
#             states.append(self.state[:, 0])

#         return np.array(states), np.array(predictions)

#     def save_configuration(self, folder):
#         os.makedirs(folder, exist_ok=True)
#         with open(os.path.join(folder, "rc_config.json"), "w") as f:
#             json.dump(self.config, f, indent=2)





"""
Reservoir Computing for LOT-Embedded Measure-Valued Time Series.

This module implements Echo State Networks (ESN) for processing LOT embeddings
as described in Section 4 of the paper.

Key equations:
    - State update (Eq. 9): r_{t+1} = φ(A r_t + B u_{t+1} + b)
    - Readout: y_t = W_out r_t
    - ESP condition (Prop. 1): L_σ ||A||_2 < 1

The input u_t is a flattened LOT embedding of shape (N*d,), representing
the transport map T_σ^{μ_t} in L²(σ; ℝᵈ).

Classes:
    ReservoirComputer: Main ESN implementation with ridge regression training
"""

import numpy as np
import json
import os
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def tanh(x: np.ndarray) -> np.ndarray:
    """Standard tanh activation (Lipschitz constant L = 1)."""
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation (Lipschitz constant L = 1)."""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation (Lipschitz constant L = 0.25)."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def identity(x: np.ndarray) -> np.ndarray:
    """Identity activation (linear reservoir)."""
    return x


# Activation registry with Lipschitz constants
ACTIVATIONS = {
    "tanh": (tanh, 1.0),
    "relu": (relu, 1.0),
    "sigmoid": (sigmoid, 0.25),
    "identity": (identity, 1.0),
}


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class InitScheme(Enum):
    """Weight initialization schemes for reservoir."""
    UNIFORM = "uniform"      # Uniform [-1, 1]
    NORMAL = "normal"        # Gaussian N(0, 1)
    SPARSE = "sparse"        # Sparse Gaussian
    SPARSE_UNIFORM = "sparse_uniform"  # Sparse uniform (common in literature)


@dataclass
class ReservoirConfig:
    """
    Configuration for reservoir computer.
    
    Attributes:
        input_size: Dimension of input u_t (should equal N*d for LOT embeddings)
        reservoir_size: Number of reservoir neurons (n in theory)
        output_size: Dimension of prediction target
        spectral_radius: Target spectral radius for W (ρ). For ESP, need L_σ * ρ < 1
        input_scaling: Scale factor for input weights W_in
        bias_scale: Scale factor for bias vector b
        leak_rate: Leaky integration parameter α ∈ (0, 1]. α=1 gives standard ESN.
        ridge_param: Ridge regression regularization parameter
        activation: Activation function name ("tanh", "relu", "sigmoid", "identity")
        init_scheme: Weight initialization scheme
        sparsity: Fraction of non-zero weights for sparse schemes
        random_seed: Random seed for reproducibility
        use_operator_norm: If True, scale by operator norm (||A||_2). 
                          If False, scale by spectral radius (legacy behavior).
    """
    input_size: int
    reservoir_size: int
    output_size: int
    spectral_radius: float = 0.7
    input_scaling: float = 0.1
    bias_scale: float = 0.0       # Paper §5.2: "No additive bias (zero vector)"
    leak_rate: float = 0.7
    ridge_param: float = 1e-6
    activation: str = "tanh"
    init_scheme: InitScheme = InitScheme.SPARSE  # Paper §5.2: sparse, 10% density
    sparsity: float = 0.1
    random_seed: Optional[int] = None
    use_operator_norm: bool = False  # Paper §5.2: scales by spectral radius ρ(W)


# ═══════════════════════════════════════════════════════════════
# RESERVOIR COMPUTER
# ═══════════════════════════════════════════════════════════════

class ReservoirComputer:
    """
    Echo State Network for processing LOT-embedded time series.
    
    Implements the state update from Eq. 9:
        r_{t+1} = (1 - α) r_t + α · φ(A r_t + B u_{t+1} + b)
    
    With α = 1 (default), this reduces to the standard ESN:
        r_{t+1} = φ(A r_t + B u_{t+1} + b)
    
    The readout is trained via ridge regression:
        W_out = argmin_W ||R W - Y||² + λ ||W||²
    
    ESP Condition (Prop. 1):
        For the echo state property, we need L_σ ||A||_2 < 1,
        where L_σ is the Lipschitz constant of the activation.
        When use_operator_norm=True, we enforce ||A||_2 = spectral_radius.
    
    Example usage:
        >>> config = ReservoirConfig(
        ...     input_size=200,      # N*d for LOT embedding
        ...     reservoir_size=500,
        ...     output_size=200,     # Predict next LOT embedding
        ...     spectral_radius=0.9,
        ... )
        >>> rc = ReservoirComputer(config)
        >>> states = rc.run(lot_embeddings[:-1])  # Shape: (T-1, reservoir_size)
        >>> rc.train(states, lot_embeddings[1:])  # Train to predict next step
        >>> predictions = rc.predict(states)
    """
    
    def __init__(self, config: ReservoirConfig):
        """
        Initialize reservoir computer.
        
        Args:
            config: ReservoirConfig instance
        """
        self.config = config
        
        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Get activation function and its Lipschitz constant
        if config.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {config.activation}. "
                           f"Choose from {list(ACTIVATIONS.keys())}")
        self._activation_fn, self._lipschitz = ACTIVATIONS[config.activation]
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize state
        self.reset_state()
        
        # Readout weights (set by train())
        self.W_out: Optional[np.ndarray] = None
        
        # Check ESP condition
        self._check_esp()
    
    def _initialize_weights(self):
        """Initialize input weights W_in, recurrent weights W, and bias b."""
        n = self.config.reservoir_size
        m = self.config.input_size
        
        # === Input weights W_in: shape (n, m) ===
        if self.config.init_scheme == InitScheme.NORMAL:
            self.W_in = np.random.randn(n, m) * self.config.input_scaling
        elif self.config.init_scheme == InitScheme.SPARSE:
            self.W_in = np.random.randn(n, m)
            mask = np.random.rand(n, m) < self.config.sparsity
            self.W_in = self.W_in * mask * self.config.input_scaling
        elif self.config.init_scheme == InitScheme.SPARSE_UNIFORM:
            self.W_in = np.random.uniform(-1, 1, (n, m))
            mask = np.random.rand(n, m) < self.config.sparsity
            self.W_in = self.W_in * mask * self.config.input_scaling
        else:  # UNIFORM (default)
            self.W_in = np.random.uniform(-1, 1, (n, m)) * self.config.input_scaling
        
        # === Recurrent weights W: shape (n, n) ===
        if self.config.init_scheme == InitScheme.NORMAL:
            W = np.random.randn(n, n)
        elif self.config.init_scheme in [InitScheme.SPARSE, InitScheme.SPARSE_UNIFORM]:
            W = np.random.randn(n, n)
            mask = np.random.rand(n, n) < self.config.sparsity
            W = W * mask
        else:  # UNIFORM
            W = np.random.uniform(-1, 1, (n, n))
        
        # Scale to target spectral radius
        if self.config.use_operator_norm:
            # Use operator norm (largest singular value) — correct for ESP
            sigma_max = np.linalg.svd(W, compute_uv=False)[0]
            if sigma_max > 0:
                self.W = W * (self.config.spectral_radius / sigma_max)
            else:
                self.W = W
        else:
            # Use spectral radius (largest eigenvalue magnitude) — legacy
            eigenvalues = np.linalg.eigvals(W)
            rho = np.max(np.abs(eigenvalues))
            if rho > 0:
                self.W = W * (self.config.spectral_radius / rho)
            else:
                self.W = W
        
        # === Bias b: shape (n, 1) ===
        self.bias = np.random.uniform(
            -self.config.bias_scale, 
            self.config.bias_scale, 
            (n, 1)
        )
    
    def _check_esp(self):
        """Check and warn about Echo State Property condition."""
        if self.config.use_operator_norm:
            # We scaled by operator norm, so ||W||_2 = spectral_radius
            operator_norm = self.config.spectral_radius
        else:
            # Compute actual operator norm
            operator_norm = np.linalg.svd(self.W, compute_uv=False)[0]
        
        esp_bound = self._lipschitz * operator_norm
        
        if esp_bound >= 1.0:
            print(f"[WARNING] ESP condition may be violated: "
                  f"L_σ * ||A||_2 = {self._lipschitz} * {operator_norm:.4f} = {esp_bound:.4f} >= 1")
        
    def reset_state(self):
        """Reset reservoir state to zeros."""
        self.state = np.zeros((self.config.reservoir_size, 1))
    
    def _update(self, u: np.ndarray) -> np.ndarray:
        """
        Single reservoir state update.
        
        Implements: r_{t+1} = (1 - α) r_t + α · φ(A r_t + B u + b)
        
        Args:
            u: Input vector, shape (input_size,) or (input_size, 1)
            
        Returns:
            Updated state, shape (reservoir_size, 1)
        """
        u = u.reshape(-1, 1)
        
        # Pre-activation: A r_t + B u + b
        pre_activation = self.W @ self.state + self.W_in @ u + self.bias
        
        # Apply activation
        activated = self._activation_fn(pre_activation)
        
        # Leaky integration
        alpha = self.config.leak_rate
        self.state = (1 - alpha) * self.state + alpha * activated
        
        return self.state
    
    def run(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Run reservoir on input sequence (teacher forcing).
        
        Args:
            input_sequence: Input time series, shape (T, input_size)
            
        Returns:
            Reservoir states, shape (T, reservoir_size)
        """
        T = input_sequence.shape[0]
        states = np.zeros((T, self.config.reservoir_size))
        
        self.reset_state()
        
        for t in range(T):
            self._update(input_sequence[t])
            states[t] = self.state.flatten()
        
        return states
    
    def train(
        self, 
        states: np.ndarray, 
        targets: np.ndarray,
        washout: int = 0,
    ) -> np.ndarray:
        """
        Train readout weights via ridge regression.
        
        Solves: W_out = argmin_W ||R W - Y||² + λ ||W||²
        
        Closed form: W_out = (R^T R + λI)^{-1} R^T Y
        
        Args:
            states: Reservoir states, shape (T, reservoir_size)
            targets: Target outputs, shape (T, output_size)
            washout: Number of initial timesteps to discard (for transient)
            
        Returns:
            Trained readout weights W_out, shape (reservoir_size + 1, output_size)
        """
        # Discard washout period
        R = states[washout:]
        Y = targets[washout:]
        
        # Append bias column
        R_aug = np.hstack([R, np.ones((R.shape[0], 1))])
        
        # Ridge regression: (R^T R + λI)^{-1} R^T Y
        ridge = self.config.ridge_param * np.eye(R_aug.shape[1])
        self.W_out = np.linalg.solve(R_aug.T @ R_aug + ridge, R_aug.T @ Y)
        
        return self.W_out
    
    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Predict outputs from reservoir states.
        
        Args:
            states: Reservoir states, shape (T, reservoir_size)
            
        Returns:
            Predictions, shape (T, output_size)
        """
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")
        
        # Append bias column
        R_aug = np.hstack([states, np.ones((states.shape[0], 1))])
        
        return R_aug @ self.W_out
    
    def run_autonomous(
        self, 
        warmup_input: np.ndarray, 
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run reservoir in autonomous (closed-loop) mode.
        
        1. Warm up with teacher forcing on warmup_input
        2. Generate n_steps predictions, feeding predictions back as input
        
        Args:
            warmup_input: Warmup sequence, shape (T_warmup, input_size)
            n_steps: Number of autonomous steps to generate
            
        Returns:
            states: All reservoir states, shape (T_warmup + n_steps, reservoir_size)
            predictions: Autonomous predictions, shape (n_steps, output_size)
        """
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")
        
        self.reset_state()
        
        # Warm-up phase (teacher forcing)
        warmup_states = []
        for t in range(warmup_input.shape[0]):
            self._update(warmup_input[t])
            warmup_states.append(self.state.flatten())
        
        # Autonomous phase (closed loop)
        autonomous_states = []
        predictions = []
        
        for _ in range(n_steps):
            # Predict from current state
            state_aug = np.append(self.state.flatten(), 1.0)
            pred = state_aug @ self.W_out
            predictions.append(pred)
            
            # Feed prediction back as input
            self._update(pred)
            autonomous_states.append(self.state.flatten())
        
        # Combine states
        all_states = np.vstack([
            np.array(warmup_states),
            np.array(autonomous_states)
        ])
        
        return all_states, np.array(predictions)

    def run_autonomous_damped(
        self,
        warmup_input: np.ndarray,
        n_steps: int,
        damping: float = 0.95,
        decay: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autonomous rollout with velocity damping.

        At each step the predicted velocity is scaled by
        ``alpha_t = damping * decay^t`` before being fed back.
        This pulls predictions toward zero velocity (persistence /
        stationary assumption) as confidence decreases over the
        forecast horizon.

        Args:
            warmup_input: Warmup sequence, shape (T_warmup, input_size)
            n_steps: Number of autonomous steps
            damping: Initial blend factor (1.0 = no damping)
            decay: Per-step multiplicative decay for damping

        Returns:
            states: All reservoir states, shape (T_warmup + n_steps, reservoir_size)
            predictions: Damped autonomous predictions, shape (n_steps, output_size)
        """
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")

        self.reset_state()
        warmup_states = []
        for t in range(warmup_input.shape[0]):
            self._update(warmup_input[t])
            warmup_states.append(self.state.flatten())

        autonomous_states = []
        predictions = []
        for step in range(n_steps):
            state_aug = np.append(self.state.flatten(), 1.0)
            pred = state_aug @ self.W_out
            alpha_t = damping * (decay ** step)
            pred_damped = alpha_t * pred
            predictions.append(pred_damped)
            self._update(pred_damped)
            autonomous_states.append(self.state.flatten())

        all_states = np.vstack([
            np.array(warmup_states),
            np.array(autonomous_states),
        ])
        return all_states, np.array(predictions)

    def run_autonomous_nudged(
        self,
        warmup_input: np.ndarray,
        n_steps: int,
        observations: np.ndarray,
        nudge_interval: int = 4,
        nudge_strength: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autonomous rollout with reservoir state nudging.

        Instead of hard-resetting the input at observation times (which
        causes RC state divergence — see REALWORLD_CLOUD_ANALYSIS.md
        Section 5.3), this method *nudges* the reservoir state toward the
        state that would result from processing the true observation.

        At every ``nudge_interval`` autonomous steps:
          1. Compute r_auto = normal autonomous update(pred)
          2. Compute r_obs  = same update equation but with the true
             observed velocity as input
          3. Blend: r = (1 - nudge_strength) * r_auto + nudge_strength * r_obs

        This keeps the reservoir's internal dynamics coherent while
        gently correcting drift.

        Args:
            warmup_input: Warmup sequence, shape (T_warmup, input_size)
            n_steps: Number of autonomous steps
            observations: True values at each step, shape (n_steps, input_size).
                Only used at nudge points; ignored at other steps.
            nudge_interval: Apply nudging every K autonomous steps
            nudge_strength: Blend factor in [0, 1]. 0 = pure autonomous,
                1 = fully replace with observation-driven state.

        Returns:
            states: All reservoir states, shape (T_warmup + n_steps, reservoir_size)
            predictions: Autonomous predictions, shape (n_steps, output_size)
        """
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")

        self.reset_state()
        warmup_states = []
        for t in range(warmup_input.shape[0]):
            self._update(warmup_input[t])
            warmup_states.append(self.state.flatten())

        autonomous_states = []
        predictions = []
        alpha = self.config.leak_rate

        for step in range(n_steps):
            # Predict from current state
            state_aug = np.append(self.state.flatten(), 1.0)
            pred = state_aug @ self.W_out
            predictions.append(pred)

            # Autonomous update with the prediction
            self._update(pred)

            # Nudge at observation intervals
            if (step + 1) % nudge_interval == 0 and step < len(observations):
                obs = observations[step].reshape(-1, 1)
                # Compute what the state *would* be if driven by the observation
                pre_obs = self.W @ self.state + self.W_in @ obs + self.bias
                activated_obs = self._activation_fn(pre_obs)
                r_obs = (1 - alpha) * self.state + alpha * activated_obs
                # Blend: nudge toward observation-driven state
                self.state = (1 - nudge_strength) * self.state + nudge_strength * r_obs

            autonomous_states.append(self.state.flatten())

        all_states = np.vstack([
            np.array(warmup_states),
            np.array(autonomous_states),
        ])
        return all_states, np.array(predictions)

    def get_config_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary (for saving)."""
        config_dict = {
            "input_size": self.config.input_size,
            "reservoir_size": self.config.reservoir_size,
            "output_size": self.config.output_size,
            "spectral_radius": self.config.spectral_radius,
            "input_scaling": self.config.input_scaling,
            "bias_scale": self.config.bias_scale,
            "leak_rate": self.config.leak_rate,
            "ridge_param": self.config.ridge_param,
            "activation": self.config.activation,
            "init_scheme": self.config.init_scheme.value,
            "sparsity": self.config.sparsity,
            "random_seed": self.config.random_seed,
            "use_operator_norm": self.config.use_operator_norm,
        }
        return config_dict
    
    def save(self, folder: str):
        """
        Save reservoir configuration and weights.
        
        Args:
            folder: Output directory
        """
        os.makedirs(folder, exist_ok=True)
        
        # Save config
        with open(os.path.join(folder, "rc_config.json"), "w") as f:
            json.dump(self.get_config_dict(), f, indent=2)
        
        # Save weights
        np.save(os.path.join(folder, "W.npy"), self.W)
        np.save(os.path.join(folder, "W_in.npy"), self.W_in)
        np.save(os.path.join(folder, "bias.npy"), self.bias)
        
        if self.W_out is not None:
            np.save(os.path.join(folder, "W_out.npy"), self.W_out)

    save_configuration = save

    @classmethod
    def load(cls, folder: str) -> "ReservoirComputer":
        """
        Load reservoir from saved files.
        
        Args:
            folder: Directory containing saved files
            
        Returns:
            Loaded ReservoirComputer instance
        """
        # Load config
        with open(os.path.join(folder, "rc_config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Convert init_scheme string back to enum
        config_dict["init_scheme"] = InitScheme(config_dict["init_scheme"])
        
        config = ReservoirConfig(**config_dict)
        
        # Create instance (will initialize random weights)
        rc = cls(config)
        
        # Override with saved weights
        rc.W = np.load(os.path.join(folder, "W.npy"))
        rc.W_in = np.load(os.path.join(folder, "W_in.npy"))
        rc.bias = np.load(os.path.join(folder, "bias.npy"))
        
        W_out_path = os.path.join(folder, "W_out.npy")
        if os.path.exists(W_out_path):
            rc.W_out = np.load(W_out_path)
        
        return rc


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_reservoir_for_lot(
    lot_embedding_dim: int,
    reservoir_size: int = 500,
    spectral_radius: float = 0.9,
    random_seed: Optional[int] = None,
) -> ReservoirComputer:
    """
    Create a reservoir computer configured for LOT embeddings.
    
    Args:
        lot_embedding_dim: Dimension of flattened LOT embedding (N * d)
        reservoir_size: Number of reservoir neurons
        spectral_radius: Target spectral radius (< 1/L_σ for ESP)
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured ReservoirComputer instance
    """
    config = ReservoirConfig(
        input_size=lot_embedding_dim,
        reservoir_size=reservoir_size,
        output_size=lot_embedding_dim,  # Predict same-dimensional output
        spectral_radius=spectral_radius,
        input_scaling=0.1,
        bias_scale=0.1,
        leak_rate=1.0,
        ridge_param=1e-6,
        activation="tanh",
        init_scheme=InitScheme.SPARSE_UNIFORM,
        sparsity=0.1,
        random_seed=random_seed,
        use_operator_norm=True,
    )
    
    return ReservoirComputer(config)


def train_one_step_predictor(
    rc: ReservoirComputer,
    lot_embeddings: np.ndarray,
    washout: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train reservoir to predict u_{t+1} from u_t.
    
    Args:
        rc: ReservoirComputer instance
        lot_embeddings: LOT embeddings, shape (T, N*d)
        washout: Timesteps to discard for transient
        
    Returns:
        states: Reservoir states, shape (T-1, reservoir_size)
        predictions: Predictions, shape (T-1, N*d)
        targets: True next-step embeddings, shape (T-1, N*d)
    """
    # Input: u_0, u_1, ..., u_{T-2}
    # Target: u_1, u_2, ..., u_{T-1}
    inputs = lot_embeddings[:-1]
    targets = lot_embeddings[1:]
    
    # Run reservoir
    states = rc.run(inputs)
    
    # Train readout
    rc.train(states, targets, washout=washout)
    
    # Predict
    predictions = rc.predict(states)
    
    return states, predictions, targets


def compute_forecast_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    washout: int = 0,
) -> Dict[str, float]:
    """
    Compute forecast error metrics.
    
    Args:
        predictions: Predicted values, shape (T, d)
        targets: True values, shape (T, d)
        washout: Timesteps to exclude from error computation
        
    Returns:
        Dictionary with MSE, RMSE, and normalized RMSE
    """
    pred = predictions[washout:]
    true = targets[washout:]
    
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    # Normalize by target variance
    target_var = np.var(true)
    nrmse = rmse / np.sqrt(target_var) if target_var > 0 else float('inf')
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
    }