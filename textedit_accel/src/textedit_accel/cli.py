from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from textedit_accel.backends import SpotEditBackend
from textedit_accel.pipeline import HybridTextEditPipeline
from textedit_accel.qwen_draft import QwenLowResolutionDraftGenerator
from textedit_accel.qwen_features import QwenVAEFeatureExtractor
from textedit_accel.qwen_runtime import (
    DEFAULT_MODEL_ID,
    configure_h800,
    load_qwen_image_edit,
)
from textedit_accel.semantic_lock import (
    SemanticLockConfig,
    SemanticVerifier,
)
from textedit_accel.types import Box, EditRequest, TextRegion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit text editing on one NVIDIA H800"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Base Qwen-Image-Edit model"
    )
    parser.add_argument(
        "--spotedit-path", required=True, help="Official SpotEdit checkout"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source-text", action="append", default=[])
    parser.add_argument("--target-text", action="append", required=True)
    parser.add_argument(
        "--box",
        action="append",
        type=json.loads,
        required=True,
        help='Repeatable JSON box, for example: --box "[120,220,600,330]"',
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--draft-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-padding", type=int, default=24)
    parser.add_argument("--disable-draft", action="store_true")
    parser.add_argument("--pixel-verifier", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "native", "flash", "_flash_3"),
        default="auto",
    )
    parser.add_argument("--no-strict-h800", action="store_true")
    parser.add_argument("--spot-threshold", type=float, default=0.15)
    parser.add_argument("--spot-initial-steps", type=int, default=4)
    parser.add_argument("--spot-reset-steps", default="13,22,31")
    parser.add_argument("--semantic-quantile", type=float, default=0.82)
    parser.add_argument("--uniform-stride", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = configure_h800(
        attention_backend=args.attention_backend,
        strict_h800=not args.no_strict_h800,
    )
    model = load_qwen_image_edit(args.model, runtime=runtime)
    reset_steps = [
        int(value) for value in args.spot_reset_steps.split(",") if value.strip()
    ]
    backend = SpotEditBackend(
        model,
        spotedit_path=args.spotedit_path,
        config_overrides={
            "reuse_mode": "velocity",
            "threshold": args.spot_threshold,
            "initial_steps": args.spot_initial_steps,
            "reset_steps": reset_steps,
            "dilation_radius": 1,
        },
        attention_backend=runtime.attention_backend,
    )
    semantic_config = SemanticLockConfig(
        draft_steps=args.draft_steps,
        discrepancy_quantile=args.semantic_quantile,
        uniform_stride=args.uniform_stride,
    )
    editor = HybridTextEditPipeline(
        backend,
        draft_generator=(
            None
            if args.disable_draft
            else QwenLowResolutionDraftGenerator(
                model, semantic_config.draft_downsample
            )
        ),
        verifier=(
            None
            if args.disable_draft
            else SemanticVerifier(
                semantic_config,
                feature_extractor=(
                    None if args.pixel_verifier else QwenVAEFeatureExtractor(model.vae)
                ),
            )
        ),
        roi_padding=args.roi_padding,
    )
    if len(args.target_text) != len(args.box):
        raise SystemExit("the number of --target-text values must match --box values")
    sources = args.source_text + [""] * (len(args.box) - len(args.source_text))
    regions = tuple(
        TextRegion(
            Box(*(int(value) for value in box)),
            source_text=sources[index],
            target_text=args.target_text[index],
        )
        for index, box in enumerate(args.box)
    )
    image = Image.open(args.image).convert("RGB")
    result = editor(
        EditRequest(
            image=image,
            prompt=args.prompt,
            target_text=tuple(args.target_text),
            regions=regions,
            seed=args.seed,
            num_inference_steps=args.steps,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output)
    summary = {
        key: value
        for key, value in result.diagnostics.items()
        if isinstance(value, (str, int, float, bool, tuple))
    }
    summary["runtime"] = runtime.to_dict()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
