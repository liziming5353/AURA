from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from textedit_accel.backends.base import SelectiveEditBackend
from textedit_accel.types import EditRequest

_PATCH_LOCK = threading.RLock()

_FAMILIES = {
    "qwen": (
        "Qwen_image_edit.qwen_spotedit",
        "Qwen_image_edit.qwen_spot_ultis",
    ),
    "qwen_plus": (
        "Qwen_image_edit_plus.qwen_plus_spotedit",
        "Qwen_image_edit_plus.qwen_spot_ultis",
    ),
    "flux": (
        "FLUX_kontext.flux_spotedit",
        "FLUX_kontext.flux_spot_ultis",
    ),
}


class SpotEditBackend(SelectiveEditBackend):
    """Bridge to the official SpotEdit repository with a forced text ROI.

    SpotEdit is loaded as an external dependency because its repository does not
    currently publish a software license. No upstream source is vendored here.
    """

    def __init__(
        self,
        pipeline,
        *,
        spotedit_path: str | Path,
        family: str = "qwen",
        config_overrides: dict | None = None,
    ):
        if family not in _FAMILIES:
            raise ValueError(
                f"unsupported family {family!r}; choose from {tuple(_FAMILIES)}"
            )
        path = Path(spotedit_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"SpotEdit checkout not found: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        generate_module, utils_module = _FAMILIES[family]
        self.generate_module = importlib.import_module(generate_module)
        self.utils_module = importlib.import_module(utils_module)
        self.pipeline = pipeline
        self.family = family
        self.config = self.utils_module.SpotEditConfig(**(config_overrides or {}))

    def token_grid(self, request: EditRequest) -> tuple[int, int]:
        vae_scale = int(self.pipeline.vae_scale_factor)
        if self.family.startswith("qwen"):
            dimensions = self.generate_module.calculate_dimensions(
                1024 * 1024, request.image.width / request.image.height
            )
            width, height = dimensions[:2]
        else:
            width = height = int(self.pipeline.default_sample_size) * vae_scale
        multiple = vae_scale * 2
        width, height = width // multiple * multiple, height // multiple * multiple
        # Qwen and current SpotEdit implementations pack 2x2 latent patches.
        return height // vae_scale // 2, width // vae_scale // 2

    def edit(
        self,
        request: EditRequest,
        forced_edit_mask: torch.Tensor,
    ) -> tuple[object, dict]:
        grid = self.token_grid(request)
        force = _resize_mask(forced_edit_mask, grid)
        aux: dict = {}
        kwargs = {
            "image": request.image,
            "prompt": request.prompt,
            "num_inference_steps": request.num_inference_steps,
            "config": self.config,
            "aux": aux,
        }
        if self.family.startswith("qwen"):
            device = getattr(self.pipeline, "_execution_device", "cpu")
            generator = torch.Generator(device=device).manual_seed(request.seed)
            kwargs["negative_prompt"] = request.negative_prompt
            kwargs["generator"] = generator
        else:
            # The official FLUX bridge does not expose a Generator argument.
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
        with self._force_recompute(force):
            output = self.generate_module.generate(self.pipeline, **kwargs)
        diagnostics = {
            **aux,
            "forced_edit_tokens": int(force.sum()),
            "total_tokens": force.numel(),
            "forced_edit_ratio": float(force.float().mean()),
            "spotedit_family": self.family,
        }
        return output.images[0], diagnostics

    @contextmanager
    def _force_recompute(self, forced_edit_mask: torch.Tensor):
        """Intersect SpotEdit's reuse mask with the inverse mandatory edit ROI."""

        with _PATCH_LOCK:
            name = "select_reuse_mask"
            original_utils = getattr(self.utils_module, name)
            original_generate = getattr(self.generate_module, name)

            def guarded_selector(*args, **kwargs):
                reuse = original_utils(*args, **kwargs)
                force = _resize_mask(
                    forced_edit_mask, _infer_grid(reuse, forced_edit_mask)
                )
                return reuse & ~force.flatten().to(device=reuse.device)

            setattr(self.utils_module, name, guarded_selector)
            setattr(self.generate_module, name, guarded_selector)
            try:
                yield
            finally:
                setattr(self.utils_module, name, original_utils)
                setattr(self.generate_module, name, original_generate)


def _resize_mask(mask: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError("forced edit mask must be [H, W]")
    return F.adaptive_max_pool2d(mask[None, None].float(), grid)[0, 0].bool()


def _infer_grid(reuse: torch.Tensor, preferred: torch.Tensor) -> tuple[int, int]:
    if reuse.numel() == preferred.numel():
        return tuple(preferred.shape)
    height = max(1, round(reuse.numel() ** 0.5))
    while height > 1 and reuse.numel() % height:
        height -= 1
    return height, reuse.numel() // height
