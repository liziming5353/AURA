from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from textedit_accel.backends import SpotEditBackend
from textedit_accel.pipeline import HybridTextEditPipeline
from textedit_accel.semantic_lock import (
    DiffusersDraftGenerator,
    SemanticLockConfig,
    SemanticVerifier,
)
from textedit_accel.types import Box, EditRequest, TextRegion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accelerated diffusion text editing")
    parser.add_argument(
        "--model", required=True, help="Diffusers model id or local directory"
    )
    parser.add_argument(
        "--spotedit-path", required=True, help="Official SpotEdit checkout"
    )
    parser.add_argument(
        "--family", choices=("qwen", "qwen_plus", "flux"), default="qwen"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source-text", default="")
    parser.add_argument("--target-text", required=True)
    parser.add_argument(
        "--box",
        type=json.loads,
        required=True,
        help='Text box as JSON, for example: "[120,220,600,330]"',
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--draft-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-padding", type=int, default=24)
    parser.add_argument("--disable-draft", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = _load_pipeline(args.model, args.family)
    backend = SpotEditBackend(
        pipeline,
        spotedit_path=args.spotedit_path,
        family=args.family,
        config_overrides={"reuse_mode": "velocity"},
    )
    semantic_config = SemanticLockConfig(draft_steps=args.draft_steps)
    editor = HybridTextEditPipeline(
        backend,
        draft_generator=(
            None
            if args.disable_draft
            else DiffusersDraftGenerator(pipeline, semantic_config.draft_downsample)
        ),
        verifier=None if args.disable_draft else SemanticVerifier(semantic_config),
        roi_padding=args.roi_padding,
    )
    box = Box(*(int(value) for value in args.box))
    image = Image.open(args.image).convert("RGB")
    request = EditRequest(
        image=image,
        prompt=args.prompt,
        target_text=args.target_text,
        regions=(
            TextRegion(box, source_text=args.source_text, target_text=args.target_text),
        ),
        seed=args.seed,
        num_inference_steps=args.steps,
    )
    result = editor(request)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output)
    summary = {
        key: value
        for key, value in result.diagnostics.items()
        if isinstance(value, (str, int, float, bool, tuple))
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_pipeline(model: str, family: str):
    try:
        from diffusers import (
            FluxKontextPipeline,
            QwenImageEditPipeline,
            QwenImageEditPlusPipeline,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install model dependencies with: pip install -e '.[qwen]'"
        ) from exc

    pipeline_type = {
        "qwen": QwenImageEditPipeline,
        "qwen_plus": QwenImageEditPlusPipeline,
        "flux": FluxKontextPipeline,
    }[family]
    return pipeline_type.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda")


if __name__ == "__main__":
    main()
