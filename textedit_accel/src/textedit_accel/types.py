from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Box:
    """Axis-aligned pixel box using half-open coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"invalid box: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def clamp(self, width: int, height: int) -> Box:
        x0 = min(max(self.x0, 0), width - 1)
        y0 = min(max(self.y0, 0), height - 1)
        x1 = min(max(self.x1, x0 + 1), width)
        y1 = min(max(self.y1, y0 + 1), height)
        return Box(x0, y0, x1, y1)

    def expand(self, pixels: int, width: int, height: int) -> Box:
        return Box(
            max(0, self.x0 - pixels),
            max(0, self.y0 - pixels),
            min(width, self.x1 + pixels),
            min(height, self.y1 + pixels),
        )

    def union(self, other: Box) -> Box:
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass(frozen=True)
class TextRegion:
    box: Box
    source_text: str = ""
    target_text: str = ""
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditRequest:
    image: Image.Image
    prompt: str
    target_text: str | Sequence[str] = ""
    regions: Sequence[TextRegion] | None = None
    negative_prompt: str | None = None
    seed: int = 42
    num_inference_steps: int = 50


@dataclass
class EditResult:
    image: Image.Image
    edit_mask: Any
    regions: Sequence[TextRegion]
    diagnostics: dict[str, Any] = field(default_factory=dict)
