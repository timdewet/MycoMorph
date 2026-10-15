"""Compute-device detection — single source of truth for the GUI + trainer.

NOTE: ``CellQualityClassifier.__init__`` keeps its own inlined probe because it
lives inside the drift-guarded shared model block (kept byte-identical with
MycoMorph). Everything *outside* that block routes through here.
"""

from __future__ import annotations


def gpu_available() -> bool:
    """True if a CUDA or Apple-MPS device is usable."""
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        if torch.cuda.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def detect_device() -> str:
    """Return the best available torch device string: 'cuda' | 'mps' | 'cpu'."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def device_label(device: str | None = None) -> str:
    """Human-readable label for a device (auto-detected if not given)."""
    dev = device or detect_device()
    if dev == "cuda":
        try:
            import torch  # type: ignore
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001
            return "CUDA"
    if dev == "mps":
        return "MPS (Apple Silicon)"
    return "CPU"
