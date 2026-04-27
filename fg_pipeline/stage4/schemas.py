from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Stage4RepairRecord:
    """Audit record for a Stage 4 LLaVA repair of a Stage 3 rejected row."""

    id: int | str
    image: Optional[str]
    question: str
    original_response: str
    failed_rewrite_response: str
    repair_response: str
    critiques: list[dict[str, Any]] = field(default_factory=list)
    stage3_votes: list[dict[str, Any]] = field(default_factory=list)
    response_severity_score: float = 1.0
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
