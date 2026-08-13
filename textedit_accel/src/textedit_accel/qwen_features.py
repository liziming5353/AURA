from __future__ import annotations

import torch


class QwenVAEFeatureExtractor:
    """Perceptual features from the base Qwen-Image-Edit VAE decoder.

    This follows SpotEdit's LPIPS-like feature choice: decoder ``conv_in``,
    ``mid_block`` and the first upsampling block. Inputs are RGB tensors in
    ``[0, 1]`` with shape ``[B, 3, H, W]``.
    """

    def __init__(self, vae):
        self.vae = vae

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> list[torch.Tensor]:
        parameter = next(self.vae.parameters())
        video = image.to(device=parameter.device, dtype=parameter.dtype)
        video = video.mul(2).sub(1).unsqueeze(2)
        encoded = self.vae.encode(video, return_dict=True)
        latent = encoded.latent_dist.mode()

        decoder = self.vae.decoder
        feature_cache = None
        feature_index = [0]
        conv_in = decoder.conv_in(latent)
        mid = decoder.mid_block(conv_in, feature_cache, feature_index)
        up0 = decoder.up_blocks[0](mid, feature_cache, feature_index)
        return [
            self._to_image_feature(conv_in),
            self._to_image_feature(mid),
            self._to_image_feature(up0),
        ]

    @staticmethod
    def _to_image_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 5:
            return feature.mean(dim=2)
        if feature.ndim != 4:
            raise ValueError(f"unexpected Qwen VAE feature shape: {feature.shape}")
        return feature
