from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from textedit_accel.cli import build_parser
from textedit_accel.qwen_draft import QwenLowResolutionDraftGenerator
from textedit_accel.qwen_features import QwenVAEFeatureExtractor
from textedit_accel.qwen_runtime import resolve_attention_backend


class PassBlock(torch.nn.Module):
    def forward(self, value, feature_cache, feature_index):
        del feature_cache, feature_index
        return value


class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.decoder = SimpleNamespace(
            conv_in=torch.nn.Identity(),
            mid_block=PassBlock(),
            up_blocks=[PassBlock()],
        )

    def encode(self, video, return_dict=True):
        del return_dict
        distribution = SimpleNamespace(mode=lambda: video)
        return SimpleNamespace(latent_dist=distribution)


class FakeQwenPipeline:
    _execution_device = "cpu"

    def __init__(self):
        self.condition_size = None

    def __call__(self, *, image, prompt, width, height, **kwargs):
        del prompt, kwargs
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
            calculate_dimensions,
        )

        self.condition_size = calculate_dimensions(
            1024 * 1024, image.width / image.height
        )[:2]
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])


def test_qwen_vae_features_have_three_spatial_levels():
    extractor = QwenVAEFeatureExtractor(FakeVAE())
    features = extractor(torch.rand(1, 3, 8, 8))

    assert len(features) == 3
    assert all(feature.shape == (1, 3, 8, 8) for feature in features)


def test_qwen_draft_shrinks_condition_and_generation_streams():
    pipeline = FakeQwenPipeline()
    draft = QwenLowResolutionDraftGenerator(pipeline, downsample=4).generate(
        Image.new("RGB", (1024, 1024)),
        "replace text",
        seed=42,
        num_inference_steps=2,
    )

    assert draft.size == (256, 256)
    assert pipeline.condition_size == (256, 256)


def test_native_attention_has_no_external_dependency():
    assert resolve_attention_backend("native") is None
    with pytest.raises(ValueError):
        resolve_attention_backend("unknown")


def test_cli_accepts_multiple_text_regions():
    args = build_parser().parse_args(
        [
            "--spotedit-path",
            "third_party/SpotEdit",
            "--image",
            "input.png",
            "--output",
            "output.png",
            "--prompt",
            "replace both labels",
            "--box",
            "[0,0,10,10]",
            "--target-text",
            "first",
            "--box",
            "[20,20,30,30]",
            "--target-text",
            "second",
        ]
    )

    assert len(args.box) == len(args.target_text) == 2
