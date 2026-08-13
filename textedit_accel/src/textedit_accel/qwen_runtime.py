from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit"


@dataclass(frozen=True)
class H800RuntimeInfo:
    device: str
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_gib: float
    dtype: str
    attention_backend: str | None
    tf32_enabled: bool

    def to_dict(self) -> dict:
        return asdict(self)


def configure_h800(
    *,
    device: str = "cuda:0",
    attention_backend: str = "auto",
    strict_h800: bool = True,
) -> H800RuntimeInfo:
    """Configure deterministic single-device execution for an NVIDIA H800."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen-Image-Edit inference requires CUDA; no GPU was detected"
        )
    index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(index)
    if strict_h800 and "H800" not in properties.name.upper():
        raise RuntimeError(
            f"expected a single NVIDIA H800, found {properties.name!r}; "
            "pass strict_h800=False only for compatibility testing"
        )
    if properties.major < 9:
        raise RuntimeError(
            f"{properties.name} has compute capability {properties.major}.{properties.minor}; "
            "Hopper (9.x) is required by the H800 profile"
        )

    torch.cuda.set_device(index)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    resolved_backend = resolve_attention_backend(attention_backend)
    return H800RuntimeInfo(
        device=device,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        total_memory_gib=properties.total_memory / 2**30,
        dtype="bfloat16",
        attention_backend=resolved_backend,
        tf32_enabled=True,
    )


def resolve_attention_backend(requested: str) -> str | None:
    if requested == "native":
        return None
    if requested == "auto":
        return "flash" if importlib.util.find_spec("flash_attn") is not None else None
    if requested not in {"flash", "_flash_3"}:
        raise ValueError("attention backend must be auto, native, flash, or _flash_3")
    if requested == "flash" and importlib.util.find_spec("flash_attn") is None:
        raise RuntimeError("flash attention requested but flash_attn is not installed")
    return requested


def load_qwen_image_edit(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    runtime: H800RuntimeInfo,
):
    """Load the base Qwen-Image-Edit pipeline entirely on one H800."""

    from diffusers import QwenImageEditPipeline

    pipeline = QwenImageEditPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipeline.to(runtime.device)
    # SpotEdit replaces attention processors during generation. Setting the model
    # backend here still configures the draft pass; the bridge separately applies
    # the same backend to SpotEdit's custom processor.
    if runtime.attention_backend is not None:
        pipeline.transformer.set_attention_backend(runtime.attention_backend)
    return pipeline
