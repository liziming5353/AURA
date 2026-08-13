from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-H800 latency benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--spotedit-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source-text", default="")
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--box", type=json.loads, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--draft-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=("auto", "native", "flash", "_flash_3"),
        default="auto",
    )
    args = parser.parse_args()

    runtime = configure_h800(attention_backend=args.attention_backend)
    model = load_qwen_image_edit(args.model, runtime=runtime)
    image = Image.open(args.image).convert("RGB")
    region = TextRegion(
        Box(*(int(value) for value in args.box)),
        source_text=args.source_text,
        target_text=args.target_text,
    )
    request = EditRequest(
        image=image,
        prompt=args.prompt,
        target_text=args.target_text,
        regions=(region,),
        seed=args.seed,
        num_inference_steps=args.steps,
    )
    backend = SpotEditBackend(
        model,
        spotedit_path=args.spotedit_path,
        config_overrides={"reuse_mode": "velocity", "dilation_radius": 1},
        attention_backend=runtime.attention_backend,
    )
    spot_only = HybridTextEditPipeline(backend, roi_padding=24)
    semantic_config = SemanticLockConfig(draft_steps=args.draft_steps)
    hybrid = HybridTextEditPipeline(
        backend,
        draft_generator=QwenLowResolutionDraftGenerator(
            model, downsample=semantic_config.draft_downsample
        ),
        verifier=SemanticVerifier(
            semantic_config,
            feature_extractor=QwenVAEFeatureExtractor(model.vae),
        ),
        roi_padding=24,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "runtime": runtime.to_dict(),
        "steps": args.steps,
        "draft_steps": args.draft_steps,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    results["baseline"] = _measure(
        "baseline",
        lambda: _baseline(model, request, runtime.device),
        output_dir,
        args.repeats,
        args.warmup,
        runtime.device,
    )
    results["spotedit_ocr"] = _measure(
        "spotedit_ocr",
        lambda: spot_only(request).image,
        output_dir,
        args.repeats,
        args.warmup,
        runtime.device,
    )
    results["hybrid"] = _measure(
        "hybrid",
        lambda: hybrid(request).image,
        output_dir,
        args.repeats,
        args.warmup,
        runtime.device,
    )
    baseline = results["baseline"]["median_seconds"]
    for name in ("spotedit_ocr", "hybrid"):
        results[name]["speedup"] = baseline / results[name]["median_seconds"]

    report = output_dir / "benchmark.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


@torch.no_grad()
def _baseline(model, request: EditRequest, device: str) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(request.seed)
    return model(
        image=request.image,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        generator=generator,
    ).images[0]


def _measure(
    name, function, output_dir: Path, repeats: int, warmup: int, device: str
) -> dict:
    for _ in range(warmup):
        function()
        torch.cuda.synchronize(device)
    latencies = []
    peaks = []
    for iteration in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        image = function()
        torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - started)
        peaks.append(torch.cuda.max_memory_allocated(device) / 2**30)
        image.save(output_dir / f"{name}-{iteration}.png")
    return {
        "latencies_seconds": latencies,
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "peak_memory_gib": max(peaks),
    }


if __name__ == "__main__":
    main()
