from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DraftGenerator(Protocol):
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        num_inference_steps: int,
    ) -> Image.Image: ...


@dataclass(frozen=True)
class SemanticLockConfig:
    """Configuration for the SpecEdit-inspired draft-and-verify stage."""

    # 4x per axis corresponds to 16x fewer spatial tokens.
    draft_downsample: int = 4
    draft_steps: int = 12
    discrepancy_quantile: float = 0.82
    minimum_score: float = 0.015
    feature_scales: tuple[int, ...] = (1, 2, 4)
    dilation_radius: int = 1
    uniform_stride: int = 8


class SemanticVerifier:
    """Create a static edit mask from a cheap draft and the source image.

    The paper uses intermediate VAE decoder features. This implementation uses
    normalized multi-scale image features by default and accepts a custom
    ``feature_extractor`` for exact backbone-specific VAE features.
    """

    def __init__(self, config: SemanticLockConfig, feature_extractor=None):
        self.config = config
        self.feature_extractor = feature_extractor

    @torch.no_grad()
    def discrepancy(self, source: Image.Image, draft: Image.Image) -> torch.Tensor:
        source_tensor = self._tensor(source)
        draft_tensor = self._tensor(draft.resize(source.size, Image.Resampling.BICUBIC))
        if self.feature_extractor is not None:
            source_features = self.feature_extractor(source_tensor)
            draft_features = self.feature_extractor(draft_tensor)
        else:
            source_features = self._multiscale(source_tensor)
            draft_features = self._multiscale(draft_tensor)

        maps = []
        target_size = source_tensor.shape[-2:]
        for source_feature, draft_feature in zip(
            source_features, draft_features, strict=True
        ):
            source_feature = F.normalize(source_feature.float(), dim=1)
            draft_feature = F.normalize(draft_feature.float(), dim=1)
            score = (source_feature - draft_feature).square().sum(dim=1, keepdim=True)
            maps.append(
                F.interpolate(score, target_size, mode="bilinear", align_corners=False)
            )
        return torch.stack(maps).mean(dim=0)[0, 0]

    @torch.no_grad()
    def verify(
        self,
        source: Image.Image,
        draft: Image.Image,
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.discrepancy(source, draft)
        token_score = F.adaptive_avg_pool2d(score[None, None], grid_size)[0, 0]
        positive = token_score[token_score >= self.config.minimum_score]
        if positive.numel() == 0:
            edit = torch.zeros_like(token_score, dtype=torch.bool)
        else:
            threshold = torch.quantile(positive, self.config.discrepancy_quantile)
            edit = token_score >= max(float(threshold), self.config.minimum_score)
        edit = self._dilate(edit)
        edit |= self._uniform_mask(grid_size, edit.device)
        return edit, token_score

    def _multiscale(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for scale in self.config.feature_scales:
            if scale == 1:
                features.append(tensor)
            else:
                features.append(F.avg_pool2d(tensor, scale, scale, ceil_mode=True))
        return features

    def _dilate(self, mask: torch.Tensor) -> torch.Tensor:
        radius = self.config.dilation_radius
        if radius <= 0:
            return mask
        return (
            F.max_pool2d(
                mask[None, None].float(),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0]
            > 0
        )

    def _uniform_mask(
        self, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        stride = self.config.uniform_stride
        if stride > 0:
            mask[::stride, ::stride] = True
        return mask

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1).div_(255).unsqueeze(0)
