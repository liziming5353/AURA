from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MixedResolutionPlan:
    """A static token routing plan produced by semantic verification."""

    fine_mask: torch.Tensor
    coarse_factor: int

    def __post_init__(self) -> None:
        if self.fine_mask.ndim != 2 or self.fine_mask.dtype != torch.bool:
            raise ValueError("fine_mask must be a two-dimensional boolean tensor")
        if self.coarse_factor < 1:
            raise ValueError("coarse_factor must be positive")

    @property
    def fine_ratio(self) -> float:
        return float(self.fine_mask.float().mean())

    @property
    def estimated_token_count(self) -> int:
        height, width = self.fine_mask.shape
        factor = self.coarse_factor
        blocks = F.max_pool2d(
            self.fine_mask[None, None].float(), factor, factor, ceil_mode=True
        )[0, 0].bool()
        fine_tokens = 0
        for row, col in blocks.nonzero().tolist():
            fine_tokens += min(factor, height - row * factor) * min(
                factor, width - col * factor
            )
        return fine_tokens + blocks.numel() - int(blocks.sum())


@dataclass
class PackedTokens:
    tokens: torch.Tensor
    coordinates: torch.Tensor
    fine_indices: torch.Tensor
    coarse_blocks: torch.Tensor
    original_shape: tuple[int, int]
    coarse_factor: int


def build_plan(
    edit_mask: torch.Tensor,
    *,
    coarse_factor: int = 2,
    uniform_stride: int = 8,
) -> MixedResolutionPlan:
    fine = edit_mask.bool().clone()
    if uniform_stride > 0:
        fine[::uniform_stride, ::uniform_stride] = True
    return MixedResolutionPlan(fine_mask=fine, coarse_factor=coarse_factor)


def pack_tokens(tokens: torch.Tensor, plan: MixedResolutionPlan) -> PackedTokens:
    """Pack selected blocks as fine tokens and other blocks as one mean token.

    ``coordinates`` contains normalized ``(y, x, scale)`` values and is intended
    for a backbone adapter to construct positional embeddings.
    """

    if tokens.ndim != 4:
        raise ValueError("tokens must have shape [B, H, W, C]")
    batch, height, width, channels = tokens.shape
    if tuple(plan.fine_mask.shape) != (height, width):
        raise ValueError("plan and token grid shapes differ")

    factor = plan.coarse_factor
    packed = []
    coordinates = []
    fine_indices = []
    coarse_blocks = []
    flat_index = 0
    for block_y in range(0, height, factor):
        for block_x in range(0, width, factor):
            y1, x1 = min(block_y + factor, height), min(block_x + factor, width)
            block_mask = plan.fine_mask[block_y:y1, block_x:x1]
            block = tokens[:, block_y:y1, block_x:x1]
            if bool(block_mask.any()):
                for local_y in range(y1 - block_y):
                    for local_x in range(x1 - block_x):
                        y, x = block_y + local_y, block_x + local_x
                        packed.append(tokens[:, y, x])
                        coordinates.append((y / height, x / width, 1.0))
                        fine_indices.append((flat_index, y, x))
                        flat_index += 1
            else:
                packed.append(block.mean(dim=(1, 2)))
                coordinates.append(
                    (
                        ((block_y + y1 - 1) / 2) / height,
                        ((block_x + x1 - 1) / 2) / width,
                        factor,
                    )
                )
                coarse_blocks.append((flat_index, block_y, y1, block_x, x1))
                flat_index += 1

    return PackedTokens(
        tokens=torch.stack(packed, dim=1).reshape(batch, -1, channels),
        coordinates=torch.tensor(
            coordinates, device=tokens.device, dtype=torch.float32
        ),
        fine_indices=torch.tensor(
            fine_indices, device=tokens.device, dtype=torch.long
        ).reshape(-1, 3),
        coarse_blocks=torch.tensor(
            coarse_blocks, device=tokens.device, dtype=torch.long
        ).reshape(-1, 5),
        original_shape=(height, width),
        coarse_factor=factor,
    )


def restore_tokens(packed: PackedTokens) -> torch.Tensor:
    """Restore a full token grid; coarse values are broadcast within their block."""

    batch, _, channels = packed.tokens.shape
    height, width = packed.original_shape
    restored = packed.tokens.new_empty((batch, height, width, channels))
    for packed_index, y, x in packed.fine_indices.tolist():
        restored[:, y, x] = packed.tokens[:, packed_index]
    for packed_index, y0, y1, x0, x1 in packed.coarse_blocks.tolist():
        restored[:, y0:y1, x0:x1] = packed.tokens[:, packed_index, None, None]
    return restored
