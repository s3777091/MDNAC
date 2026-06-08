from __future__ import annotations

from pathlib import Path

import torch


def create_onnx_session(
    onnx_path: str | Path,
    *,
    device_name: str = "auto",
):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Loading ONNX artifacts requires `onnxruntime`. Install it with `uv sync --extra onnx`."
        ) from exc

    requested = str(device_name).lower()
    available = ort.get_available_providers()
    if requested == "mps":
        raise RuntimeError("MPS is not supported for ONNX Runtime inference in this repo.")

    if requested == "cuda":
        if "CUDAExecutionProvider" not in available or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for ONNX inference, but `CUDAExecutionProvider` is unavailable."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device = torch.device("cuda")
    elif requested == "auto":
        if "CUDAExecutionProvider" in available and torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            device = torch.device("cuda")
        else:
            providers = ["CPUExecutionProvider"]
            device = torch.device("cpu")
    else:
        providers = ["CPUExecutionProvider"]
        device = torch.device("cpu")

    session = ort.InferenceSession(str(Path(onnx_path).expanduser().resolve()), providers=providers)
    return session, device
