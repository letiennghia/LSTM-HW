"""Seeds, device selection and recorded runtime information."""

import importlib.metadata
import platform
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(request: str = "auto") -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device=auto or cpu")
    return torch.device("cuda" if request == "auto" and torch.cuda.is_available() else "cpu" if request == "auto" else request)


def environment(device: torch.device) -> dict:
    packages = ["torch", "datasets", "scikit-learn", "numpy", "pandas", "matplotlib", "PyYAML", "joblib"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "versions": {name: importlib.metadata.version(name) for name in packages},
            "determinism_note": "Seeds and deterministic cuDNN settings reduce variation; GPU results are not guaranteed bitwise identical across hardware or versions."}
