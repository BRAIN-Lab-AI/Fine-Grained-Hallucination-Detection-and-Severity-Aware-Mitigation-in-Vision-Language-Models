"""Stage 1 — critique detection / critique extraction.

This package turns released fine-grained supervision rows into a normalized
Stage 1 record (``Stage1Record``) that downstream stages can consume directly.

The current default backend (:class:`ReleasedAnnotationBackend`) parses the
released ``hsa_dpo_detection.jsonl`` format and does not require any model
inference. The ``CritiqueDetectorBackend`` protocol is the seam for plugging
in a trained detector later without changing the Stage 1 output schema.
"""

from fg_pipeline.stage1.schemas import (
    CritiqueItem,
    Stage1Record,
    SEVERITY_LABEL_TO_SCORE,
)
from fg_pipeline.stage1.parser import (
    ParseResult,
    parse_detection_row,
)
from fg_pipeline.stage1.backends import (
    CritiqueDetectorBackend,
    ReleasedAnnotationBackend,
    get_backend,
)

__all__ = [
    "CritiqueItem",
    "Stage1Record",
    "SEVERITY_LABEL_TO_SCORE",
    "ParseResult",
    "parse_detection_row",
    "CritiqueDetectorBackend",
    "ReleasedAnnotationBackend",
    "get_backend",
]
