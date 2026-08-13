import torch

from textedit_accel.regions import pixel_mask_to_tokens, plan_regions, rasterize_regions
from textedit_accel.types import Box, TextRegion


def test_target_box_expands_for_longer_text():
    regions = plan_regions(
        (TextRegion(Box(40, 20, 80, 40), source_text="AI"),),
        "人工智能大会",
        (200, 100),
        padding=4,
    )

    assert regions[0].box.x0 < 40
    assert regions[0].box.x1 > 80
    assert regions[0].target_text == "人工智能大会"


def test_rasterization_and_token_projection_are_conservative():
    region = TextRegion(Box(31, 31, 33, 33))
    pixels = rasterize_regions((region,), (64, 64))
    tokens = pixel_mask_to_tokens(pixels, (4, 4))

    assert pixels.sum() == 4
    assert tokens.dtype == torch.bool
    assert tokens.sum() == 4
