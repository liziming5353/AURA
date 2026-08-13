import torch
from PIL import Image

from textedit_accel.backends.base import SelectiveEditBackend
from textedit_accel.pipeline import HybridTextEditPipeline
from textedit_accel.semantic_lock import SemanticLockConfig, SemanticVerifier
from textedit_accel.types import Box, EditRequest, TextRegion


class FakeBackend(SelectiveEditBackend):
    def token_grid(self, request):
        return 8, 8

    def edit(self, request, forced_edit_mask):
        return request.image.copy(), {"received": forced_edit_mask.clone()}


class FakeDraft:
    def generate(self, image, prompt, *, seed, num_inference_steps):
        draft = image.copy()
        draft.paste((255, 255, 255), (48, 48, 64, 64))
        return draft


def test_pipeline_unions_ocr_and_semantic_masks():
    image = Image.new("RGB", (64, 64), "black")
    request = EditRequest(
        image=image,
        prompt="replace the text",
        target_text="longer target",
        regions=(TextRegion(Box(0, 0, 16, 16), source_text="a"),),
    )
    verifier = SemanticVerifier(
        SemanticLockConfig(
            discrepancy_quantile=0.0,
            minimum_score=0.001,
            dilation_radius=0,
            uniform_stride=0,
        )
    )
    result = HybridTextEditPipeline(
        FakeBackend(),
        draft_generator=FakeDraft(),
        verifier=verifier,
        roi_padding=0,
    )(request)

    received = result.diagnostics["received"]
    assert received.dtype == torch.bool
    assert received[:2].any()  # OCR ROI
    assert received[6:, 6:].any()  # semantic discrepancy


def test_pipeline_requires_regions_until_ocr_is_configured():
    request = EditRequest(
        image=Image.new("RGB", (32, 32)),
        prompt="replace text",
        target_text="new",
    )
    try:
        HybridTextEditPipeline(FakeBackend())(request)
    except ValueError as error:
        assert "OCRProvider" in str(error)
    else:
        raise AssertionError("expected missing OCR regions to fail")
