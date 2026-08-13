from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from PIL import Image

from textedit_accel.types import TextRegion


class OCRProvider(ABC):
    """Interface for OCR/text detection implementations."""

    @abstractmethod
    def detect(self, image: Image.Image) -> Sequence[TextRegion]:
        """Return text and pixel-space boxes in the input image."""


class NullOCRProvider(OCRProvider):
    """Placeholder used until an OCR implementation is configured."""

    def detect(self, image: Image.Image) -> Sequence[TextRegion]:
        del image
        return ()


class CallableOCRProvider(OCRProvider):
    """Adapter for a user supplied ``fn(PIL.Image) -> Sequence[TextRegion]``."""

    def __init__(self, fn):
        self._fn = fn

    def detect(self, image: Image.Image) -> Sequence[TextRegion]:
        return tuple(self._fn(image))
