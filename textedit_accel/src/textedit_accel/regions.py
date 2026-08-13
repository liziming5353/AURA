from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from textedit_accel.types import Box, TextRegion


def estimate_target_box(region: TextRegion, image_size: tuple[int, int]) -> Box:
    """Conservatively estimate the target layout when string length changes.

    OCR implementations can override this by putting a ``target_box`` Box in
    ``region.metadata``.
    """

    explicit = region.metadata.get("target_box")
    if isinstance(explicit, Box):
        return explicit.clamp(*image_size)

    source_units = max(_visual_units(region.source_text), 1.0)
    target_units = max(_visual_units(region.target_text), 1.0)
    ratio = min(max(target_units / source_units, 0.5), 4.0)
    new_width = max(region.box.width, math.ceil(region.box.width * ratio))
    center_x = (region.box.x0 + region.box.x1) / 2
    proposed = Box(
        math.floor(center_x - new_width / 2),
        region.box.y0,
        math.ceil(center_x + new_width / 2),
        region.box.y1,
    )
    return proposed.clamp(*image_size)


def _visual_units(text: str) -> float:
    # CJK glyphs are approximately square; latin glyphs are usually narrower.
    return sum(1.0 if ord(char) > 0x2E7F else 0.55 for char in text.strip())


def plan_regions(
    regions: Sequence[TextRegion],
    target_text: str | Sequence[str],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[TextRegion, ...]:
    targets = [target_text] if isinstance(target_text, str) else list(target_text)
    planned: list[TextRegion] = []
    for index, region in enumerate(regions):
        target = targets[index] if index < len(targets) else region.target_text
        current = TextRegion(
            box=region.box,
            source_text=region.source_text,
            target_text=target,
            confidence=region.confidence,
            metadata=region.metadata,
        )
        target_box = estimate_target_box(current, image_size)
        union = current.box.union(target_box).expand(padding, *image_size)
        planned.append(
            TextRegion(
                box=union,
                source_text=current.source_text,
                target_text=target,
                confidence=current.confidence,
                metadata={
                    **current.metadata,
                    "source_box": current.box,
                    "target_box": target_box,
                },
            )
        )
    return tuple(planned)


def rasterize_regions(
    regions: Sequence[TextRegion],
    image_size: tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean pixel mask where True means full-resolution recompute."""

    width, height = image_size
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for region in regions:
        box = region.box.clamp(width, height)
        mask[box.y0 : box.y1, box.x0 : box.x1] = True
    return mask


def pixel_mask_to_tokens(
    mask: torch.Tensor, grid_size: tuple[int, int]
) -> torch.Tensor:
    """Conservatively project a pixel mask to a token grid."""

    if mask.ndim != 2:
        raise ValueError("pixel mask must have shape [H, W]")
    token_h, token_w = grid_size
    pooled = torch.nn.functional.adaptive_max_pool2d(
        mask[None, None].float(), (token_h, token_w)
    )
    return pooled[0, 0].bool()


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return (
        torch.nn.functional.max_pool2d(
            mask[None, None].float(), kernel, stride=1, padding=radius
        )[0, 0]
        > 0
    )
