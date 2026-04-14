"""
PyTorch Echo State Network for LOT embeddings (GPU-accelerated).

API mirrors ``rc_computer.ReservoirComputer`` for drop-in use in forecast scripts:
numpy arrays in/out; weights live on ``device`` during compute.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rc_computer import ACTIVATIONS, InitScheme, ReservoirConfig


def _torch_activation(name: str):
    fn, lip = ACTIVATIONS[name]
    if name == "tanh":
        return torch.tanh, lip
    if name == "relu":
        return torch.relu, lip
    if name == "sigmoid":
        return torch.sigmoid, lip
    if name == "identity":
        return lambda x: x, lip
    raise ValueError(f"Unknown activation: {name}")


class ReservoirComputerTorch:
    """ESN with ridge readout; uses PyTorch for matmuls on CPU or CUDA."""

    def __init__(
        self,
        config: ReservoirConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)

        if config.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {config.activation}")
        self._act_fn, self._lipschitz = _torch_activation(config.activation)

        self._initialize_weights()
        self.reset_state()
        self.W_out: Optional[torch.Tensor] = None
        self._check_esp()

    def _initialize_weights(self):
        n = self.config.reservoir_size
        m = self.config.input_size
        dev = self.device
        dt = self.dtype

        if self.config.init_scheme == InitScheme.NORMAL:
            W_in = torch.randn(n, m, device=dev, dtype=dt) * self.config.input_scaling
        elif self.config.init_scheme == InitScheme.SPARSE:
            W_in = torch.randn(n, m, device=dev, dtype=dt)
            mask = torch.rand(n, m, device=dev) < self.config.sparsity
            W_in = W_in * mask.to(dt) * self.config.input_scaling
        elif self.config.init_scheme == InitScheme.SPARSE_UNIFORM:
            W_in = torch.empty(n, m, device=dev, dtype=dt).uniform_(-1, 1)
            mask = torch.rand(n, m, device=dev) < self.config.sparsity
            W_in = W_in * mask.to(dt) * self.config.input_scaling
        else:
            W_in = torch.empty(n, m, device=dev, dtype=dt).uniform_(-1, 1) * self.config.input_scaling

        if self.config.init_scheme == InitScheme.NORMAL:
            W = torch.randn(n, n, device=dev, dtype=dt)
        elif self.config.init_scheme in (InitScheme.SPARSE, InitScheme.SPARSE_UNIFORM):
            W = torch.randn(n, n, device=dev, dtype=dt)
            mask = torch.rand(n, n, device=dev) < self.config.sparsity
            W = W * mask.to(dt)
        else:
            W = torch.empty(n, n, device=dev, dtype=dt).uniform_(-1, 1)

        if self.config.use_operator_norm:
            sigma_max = torch.linalg.svdvals(W)[0]
            if sigma_max > 0:
                W = W * (self.config.spectral_radius / sigma_max)
        else:
            eig = torch.linalg.eigvals(W)
            rho = eig.abs().max()
            if rho > 0:
                W = W * (self.config.spectral_radius / rho)

        bias = torch.empty(n, 1, device=dev, dtype=dt).uniform_(
            -self.config.bias_scale, self.config.bias_scale
        )

        self.W_in = W_in
        self.W = W
        self.bias = bias

    def _check_esp(self):
        if self.config.use_operator_norm:
            operator_norm = self.config.spectral_radius
        else:
            operator_norm = torch.linalg.svdvals(self.W)[0].item()
        esp_bound = self._lipschitz * operator_norm
        if esp_bound >= 1.0:
            print(
                f"[WARNING] ESP condition may be violated: "
                f"L_σ * ||A||_2 = {self._lipschitz} * {operator_norm:.4f} = {esp_bound:.4f} >= 1"
            )

    def reset_state(self):
        n = self.config.reservoir_size
        self.state = torch.zeros(n, 1, device=self.device, dtype=self.dtype)

    def _update(self, u: torch.Tensor) -> torch.Tensor:
        u = u.reshape(-1, 1)
        pre = self.W @ self.state + self.W_in @ u + self.bias
        activated = self._act_fn(pre)
        alpha = self.config.leak_rate
        self.state = (1 - alpha) * self.state + alpha * activated
        return self.state

    def run(self, input_sequence: np.ndarray) -> np.ndarray:
        T = input_sequence.shape[0]
        n = self.config.reservoir_size
        states = torch.empty(T, n, device=self.device, dtype=self.dtype)
        self.reset_state()
        x = torch.as_tensor(input_sequence, device=self.device, dtype=self.dtype)
        for t in range(T):
            self._update(x[t])
            states[t] = self.state.squeeze(-1)
        return states.detach().cpu().numpy()

    def train(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        washout: int = 0,
    ) -> np.ndarray:
        R_np = states[washout:]
        Y_np = targets[washout:]
        R = torch.as_tensor(R_np, device=self.device, dtype=self.dtype)
        Y = torch.as_tensor(Y_np, device=self.device, dtype=self.dtype)
        ones = torch.ones(R.shape[0], 1, device=self.device, dtype=self.dtype)
        R_aug = torch.cat([R, ones], dim=1)
        ridge = self.config.ridge_param * torch.eye(R_aug.shape[1], device=self.device, dtype=self.dtype)
        lhs = R_aug.T @ R_aug + ridge
        rhs = R_aug.T @ Y
        W_out = torch.linalg.solve(lhs, rhs)
        self.W_out = W_out
        return W_out.detach().cpu().numpy()

    def predict(self, states: np.ndarray) -> np.ndarray:
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")
        R = torch.as_tensor(states, device=self.device, dtype=self.dtype)
        ones = torch.ones(R.shape[0], 1, device=self.device, dtype=self.dtype)
        R_aug = torch.cat([R, ones], dim=1)
        out = R_aug @ self.W_out
        return out.detach().cpu().numpy()

    def run_autonomous(
        self,
        warmup_input: np.ndarray,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.W_out is None:
            raise ValueError("Readout weights not trained. Call train() first.")

        self.reset_state()
        wu = torch.as_tensor(warmup_input, device=self.device, dtype=self.dtype)
        warmup_states = []
        for t in range(warmup_input.shape[0]):
            self._update(wu[t])
            warmup_states.append(self.state.squeeze(-1).detach().clone())

        autonomous_states = []
        predictions = []
        for _ in range(n_steps):
            state_flat = torch.cat([self.state.squeeze(-1), self.state.new_ones(1)])
            pred = state_flat @ self.W_out
            predictions.append(pred.detach().clone())
            self._update(pred)
            autonomous_states.append(self.state.squeeze(-1).detach().clone())

        wu_t = torch.stack(warmup_states)
        au_t = torch.stack(autonomous_states)
        all_states = torch.cat([wu_t, au_t], dim=0)
        pred_t = torch.stack(predictions)
        return all_states.detach().cpu().numpy(), pred_t.detach().cpu().numpy()

    def get_config_dict(self) -> Dict[str, Any]:
        return {
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
            "backend": "torch",
            "device": str(self.device),
        }

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "rc_config.json"), "w") as f:
            json.dump(self.get_config_dict(), f, indent=2)
        np.save(os.path.join(folder, "W.npy"), self.W.detach().cpu().numpy())
        np.save(os.path.join(folder, "W_in.npy"), self.W_in.detach().cpu().numpy())
        np.save(os.path.join(folder, "bias.npy"), self.bias.detach().cpu().numpy())
        if self.W_out is not None:
            np.save(os.path.join(folder, "W_out.npy"), self.W_out.detach().cpu().numpy())
        torch.save(
            {
                "W": self.W,
                "W_in": self.W_in,
                "bias": self.bias,
                "W_out": self.W_out,
            },
            os.path.join(folder, "rc_weights.pt"),
        )

    save_configuration = save

    @classmethod
    def load(cls, folder: str, device: Optional[torch.device] = None) -> "ReservoirComputerTorch":
        with open(os.path.join(folder, "rc_config.json"), "r") as f:
            config_dict = json.load(f)
        config_dict.pop("backend", None)
        config_dict.pop("device", None)
        config_dict["init_scheme"] = InitScheme(config_dict["init_scheme"])
        config = ReservoirConfig(**config_dict)
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rc = cls(config, device=dev)
        pt_path = os.path.join(folder, "rc_weights.pt")
        if os.path.isfile(pt_path):
            ckpt = torch.load(pt_path, map_location=dev)
            rc.W = ckpt["W"].to(dev)
            rc.W_in = ckpt["W_in"].to(dev)
            rc.bias = ckpt["bias"].to(dev)
            if ckpt.get("W_out") is not None:
                rc.W_out = ckpt["W_out"].to(dev)
        else:
            rc.W = torch.as_tensor(np.load(os.path.join(folder, "W.npy")), device=dev, dtype=rc.dtype)
            rc.W_in = torch.as_tensor(np.load(os.path.join(folder, "W_in.npy")), device=dev, dtype=rc.dtype)
            rc.bias = torch.as_tensor(np.load(os.path.join(folder, "bias.npy")), device=dev, dtype=rc.dtype)
            w_out_path = os.path.join(folder, "W_out.npy")
            if os.path.exists(w_out_path):
                rc.W_out = torch.as_tensor(np.load(w_out_path), device=dev, dtype=rc.dtype)
        return rc
