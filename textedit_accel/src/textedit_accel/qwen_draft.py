from __future__ import annotations

import threading
from contextlib import contextmanager

import torch
from PIL import Image

_DIMENSION_PATCH_LOCK = threading.RLock()


class QwenLowResolutionDraftGenerator:
    """Run both Qwen edit and condition-image tokens at draft resolution.

    Passing ``width`` and ``height`` to the stock pipeline only shrinks generated
    latents; the condition image is still resized around 1024². This adapter
    temporarily redirects Qwen's internal dimension calculation so both streams
    use the low-resolution draft grid.
    """

    def __init__(self, pipeline, downsample: int = 4):
        if downsample < 1:
            raise ValueError("downsample must be positive")
        self.pipeline = pipeline
        self.downsample = downsample

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        num_inference_steps: int,
    ) -> Image.Image:
        width = max(64, _round_to_multiple(image.width // self.downsample, 32))
        height = max(64, _round_to_multiple(image.height // self.downsample, 32))
        device = getattr(self.pipeline, "_execution_device", "cpu")
        generator = torch.Generator(device=device).manual_seed(seed)
        with self._draft_dimensions(width * height):
            output = self.pipeline(
                image=image.resize((width, height), Image.Resampling.LANCZOS),
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        return output.images[0]

    @contextmanager
    def _draft_dimensions(self, target_area: int):
        from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit as module

        with _DIMENSION_PATCH_LOCK:
            original = module.calculate_dimensions

            def draft_dimensions(_target_area, ratio):
                return original(target_area, ratio)

            module.calculate_dimensions = draft_dimensions
            try:
                yield
            finally:
                module.calculate_dimensions = original


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)
