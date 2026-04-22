from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SEVERITY_LABEL_TO_SCORE: dict[str, int] = {
    "minor": 1,
    "moderate": 2,
    "major": 3,
}

VALID_HALLUCINATION_TYPES: tuple[str, ...] = (
    "object",
    "attribute",
    "relationship",
)

UNKNOWN = "unknown"


@dataclass
class CritiqueItem:
    """One critique extracted for a sample.

    ``rationale`` is the parsed tag text describing the hallucination.
    ``evidence_text`` is the short span before the colon in the score line
    (e.g. "Sidewalk", "Umbrella hues") when the score line is well-formed.
    ``source_tag_text`` / ``source_score_text`` preserve the raw lines so
    downstream stages can re-parse if needed.
    """

    index: int
    hallucination_type: str
    severity_label: str
    severity_score: Optional[int]
    rationale: str
    evidence_text: Optional[str] = None
    source_tag_text: Optional[str] = None
    source_score_text: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1Record:
    """Normalized Stage 1 output record.

    Stable contract consumed by Stage 2. Fields that cannot be recovered from
    the source row surface as ``None`` rather than being dropped. No
    confidence / calibration / threshold fields live on this record by design.

    The released Stage 1 source rows do not carry the original user prompt
    separately from the assessed candidate response. For now, ``question`` and
    ``response_text`` may mirror the same assessed sentence, while the raw GPT
    supervision text is preserved under ``metadata["raw_annotation_text"]``.
    """

    id: int | str
    image: Optional[str]
    question: str
    response_text: str
    is_hallucinated: bool
    critiques: list[CritiqueItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict already expands CritiqueItem dataclasses recursively.
        return data
