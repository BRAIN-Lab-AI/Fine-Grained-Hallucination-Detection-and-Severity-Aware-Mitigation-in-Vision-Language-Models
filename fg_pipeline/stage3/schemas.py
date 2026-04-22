from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class VoteDecision:
    """One verification vote over a Stage 2 rewrite candidate."""

    vote_index: int
    criterion: str
    approved: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage3Record:
    """Stage 3 audit record produced for one hallucinated Stage 2 input."""

    id: int | str
    image: Optional[str]
    question: str
    original_response: str
    rewrite_response: str
    critiques: list[dict[str, Any]] = field(default_factory=list)
    votes: list[VoteDecision] = field(default_factory=list)
    approvals: int = 0
    rejections: int = 0
    passed_majority: bool = False
    response_severity_score: float = 1.0
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
