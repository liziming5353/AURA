from __future__ import annotations

import torch

from textedit_accel.backends.base import SelectiveEditBackend
from textedit_accel.ocr import NullOCRProvider, OCRProvider
from textedit_accel.regions import pixel_mask_to_tokens, plan_regions, rasterize_regions
from textedit_accel.semantic_lock import DraftGenerator, SemanticVerifier
from textedit_accel.types import EditRequest, EditResult


class HybridTextEditPipeline:
    """OCR guardrails + SpecEdit verification + SpotEdit selective compute."""

    def __init__(
        self,
        backend: SelectiveEditBackend,
        *,
        ocr: OCRProvider | None = None,
        draft_generator: DraftGenerator | None = None,
        verifier: SemanticVerifier | None = None,
        roi_padding: int = 24,
    ):
        self.backend = backend
        self.ocr = ocr or NullOCRProvider()
        self.draft_generator = draft_generator
        self.verifier = verifier
        self.roi_padding = roi_padding
        if (draft_generator is None) != (verifier is None):
            raise ValueError("draft_generator and verifier must be configured together")

    @torch.no_grad()
    def __call__(self, request: EditRequest) -> EditResult:
        detected = tuple(request.regions or self.ocr.detect(request.image))
        if not detected:
            raise ValueError(
                "no text region available: pass EditRequest.regions or configure an OCRProvider"
            )

        regions = plan_regions(
            detected,
            request.target_text,
            request.image.size,
            padding=self.roi_padding,
        )
        pixel_mask = rasterize_regions(regions, request.image.size)
        grid = self.backend.token_grid(request)
        ocr_mask = pixel_mask_to_tokens(pixel_mask, grid)
        forced_mask = ocr_mask.clone()
        diagnostics = {
            "token_grid": grid,
            "ocr_edit_tokens": int(ocr_mask.sum()),
            "ocr_edit_ratio": float(ocr_mask.float().mean()),
        }

        if self.draft_generator is not None and self.verifier is not None:
            draft = self.draft_generator.generate(
                request.image,
                request.prompt,
                seed=request.seed,
                num_inference_steps=self.verifier.config.draft_steps,
            )
            semantic_mask, scores = self.verifier.verify(request.image, draft, grid)
            forced_mask |= semantic_mask.to(forced_mask.device)
            diagnostics.update(
                {
                    "semantic_edit_tokens": int(semantic_mask.sum()),
                    "semantic_edit_ratio": float(semantic_mask.float().mean()),
                    "semantic_score_min": float(scores.min()),
                    "semantic_score_max": float(scores.max()),
                    "draft_image": draft,
                }
            )

        image, backend_diagnostics = self.backend.edit(request, forced_mask)
        diagnostics.update(backend_diagnostics)
        return EditResult(
            image=image,
            edit_mask=forced_mask.cpu(),
            regions=regions,
            diagnostics=diagnostics,
        )
