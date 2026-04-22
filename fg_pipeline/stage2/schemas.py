from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Stage2Record:
    """Normalized Stage 2 output record.

    One record per hallucinated Stage 1 input. Non-hallucinated rows are
    skipped and do not produce a Stage 2 record.

    No confidence / calibration / threshold fields live on this record.
    Stage 3 consumes this record to produce validated preference pairs.
    """

    id: int | str
    image: Optional[str]
    question: str
    original_response: str
    rewrite_response: str
    critiques: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
