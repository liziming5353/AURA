from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from textedit_accel.types import EditRequest


class SelectiveEditBackend(ABC):
    @abstractmethod
    def token_grid(self, request: EditRequest) -> tuple[int, int]:
        """Return the generated latent token grid as ``(height, width)``."""

    @abstractmethod
    def edit(
        self,
        request: EditRequest,
        forced_edit_mask: torch.Tensor,
    ) -> tuple[Any, dict]:
        """Run selective editing; True mask positions must be recomputed."""
