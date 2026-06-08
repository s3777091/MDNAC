from __future__ import annotations

from pathlib import Path


def create_onnx_session(
    onnx_path: str | Path,
    *,
    device_name: str = "auto",
):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Loading ONNX artifacts requires `onnxruntime`. "
            "Install the api project dependencies from the `api` directory."
        ) from exc

    requested = str(device_name).strip().lower()
    available = set(ort.get_available_providers())
    if requested == "mps":
        raise RuntimeError("MPS is not supported by ONNX Runtime for this API.")

    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested for ONNX inference, but `CUDAExecutionProvider` is unavailable."
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif requested == "auto" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(Path(onnx_path).expanduser().resolve()),
        providers=providers,
    )
    return session, tuple(session.get_providers())
