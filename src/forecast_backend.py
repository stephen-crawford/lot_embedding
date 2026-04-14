"""
Choose NumPy or PyTorch reservoir implementation for forecasting CLIs.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

from rc_computer import ReservoirComputer, ReservoirConfig

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def resolve_torch_device(
    pref: Literal["auto", "cpu", "cuda"],
) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch is not installed; use --rc-backend numpy")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def create_reservoir(
    rc_config: ReservoirConfig,
    backend: Literal["auto", "numpy", "torch"] = "auto",
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> Tuple[Any, str, str]:
    """
    Returns:
        rc: ReservoirComputer or ReservoirComputerTorch
        backend_used: "numpy" or "torch"
        device_str: e.g. "cpu", "cuda:0"
    """
    if backend == "numpy":
        return ReservoirComputer(rc_config), "numpy", "cpu"

    if backend == "torch" and torch is None:
        raise RuntimeError("PyTorch is not installed; pip install torch or use --rc-backend numpy")

    if backend == "auto":
        if torch is None:
            return ReservoirComputer(rc_config), "numpy", "cpu"
        backend = "torch"

    if backend == "torch":
        from rc_computer_torch import ReservoirComputerTorch

        dev = resolve_torch_device(device)
        rc = ReservoirComputerTorch(rc_config, device=dev)
        return rc, "torch", str(dev)

    raise ValueError(f"Unknown backend: {backend}")
