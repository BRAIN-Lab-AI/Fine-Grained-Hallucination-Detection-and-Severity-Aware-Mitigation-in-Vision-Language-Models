from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from fg_pipeline.stage1.schemas import (
    SEVERITY_LABEL_TO_SCORE,
    CritiqueItem,
    Stage1Record,
)
from fg_pipeline.stage2.schemas import Stage2Record
from fg_pipeline.stage3.schemas import Stage3Record, VoteDecision


@dataclass
class PreferenceCleanRecord:
    """Preference pair compatible with the original HSA-DPO trainer."""

    id: int | str
    question: str
    chosen: str
    rejected: str
    chosen_score: float = 1.0
    rejected_score: float = 1.0
    image: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "PreferenceCleanRecord",
    "Stage1Record",
    "Stage2Record",
    "Stage3Record",
    "VoteDecision",
    "CritiqueItem",
    "SEVERITY_LABEL_TO_SCORE",
]
