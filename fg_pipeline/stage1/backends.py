"""Backend seam for Stage 1 critique detection.

The ``CritiqueDetectorBackend`` protocol fixes the Stage-1-facing entry point
(:meth:`detect`). The default :class:`ReleasedAnnotationBackend` parses rows
from the released ``hsa_dpo_detection.jsonl`` supervision; a future trained
detector backend can be swapped in without changing the Stage 1 output schema.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fg_pipeline.stage1.parser import ParseResult, parse_detection_row
from fg_pipeline.stage1.schemas import Stage1Record


@runtime_checkable
class CritiqueDetectorBackend(Protocol):
    """Minimal Stage-1-facing contract for any critique detection backend."""

    name: str

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        """Return a :class:`ParseResult` for one source row."""
        ...


class ReleasedAnnotationBackend:
    """Default backend: parse released HSA-DPO detection supervision."""

    name: str = "released_annotations"

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        return parse_detection_row(row, strict=strict)


_REGISTRY: dict[str, type[CritiqueDetectorBackend]] = {
    ReleasedAnnotationBackend.name: ReleasedAnnotationBackend,
}


def get_backend(name: str) -> CritiqueDetectorBackend:
    """Return an instantiated backend by registered name."""

    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown stage1 backend {name!r}; available: {available}"
        )
    return _REGISTRY[key]()


__all__ = [
    "CritiqueDetectorBackend",
    "ReleasedAnnotationBackend",
    "Stage1Record",
    "get_backend",
]
