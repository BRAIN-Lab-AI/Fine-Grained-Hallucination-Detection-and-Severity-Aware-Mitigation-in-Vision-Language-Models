from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Direction = Literal["higher_better", "lower_better"]
ModelKind = Literal["base", "lora", "merged"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_path: str
    model_base: str | None
    kind: ModelKind
    conv_mode: str
    temperature: float = 0.0
    num_beams: int = 1
    max_new_tokens: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        return cls(
            model_id=str(payload["model_id"]),
            model_path=str(payload["model_path"]),
            model_base=payload.get("model_base"),
            kind=payload["kind"],
            conv_mode=str(payload.get("conv_mode", "vicuna_v1")),
            temperature=float(payload.get("temperature", 0.0)),
            num_beams=int(payload.get("num_beams", 1)),
            max_new_tokens=int(payload.get("max_new_tokens", 512)),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    enabled: bool = True
    question_file: str | None = None
    annotation_file: str | None = None
    image_root: str | None = None
    dataset_root: str | None = None
    judge_required: bool = False
    split: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionArtifact:
    benchmark: str
    model_id: str
    path: str
    num_examples: int
    decode_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricArtifact:
    benchmark: str
    model_id: str
    metrics: dict[str, float | int | str | None]
    comparable_to_paper: bool
    comparison_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeArtifact:
    benchmark: str
    model_id: str
    judge_name: str
    judge_version: str | None
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonRow:
    benchmark: str
    metric: str
    direction: Direction
    base_value: float | None
    our_value: float | None
    paper_reference_value: float | None
    delta_vs_base: float | None
    relative_delta_vs_base: float | None
    paper_row_name: str | None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
