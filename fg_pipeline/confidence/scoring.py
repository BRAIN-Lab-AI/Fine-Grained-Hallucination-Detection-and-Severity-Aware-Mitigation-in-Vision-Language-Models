from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from fg_pipeline.schemas import SentenceSignal


@runtime_checkable
class ConfidenceScorer(Protocol):
    """Per-signal confidence scorer. Stage 3's locked semantics for c^j."""

    name: str

    def score(
        self, signal_data: dict[str, Any], record_context: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Return (confidence, scorer_metadata) for one parsed signal."""


class BootstrapScorer:
    """Placeholder scorer that passes through teacher labels as c^j = 1.0.

    Every emitted signal is tagged ``is_placeholder=True`` so downstream
    filters can refuse to treat these values as real confidence once a
    trained scorer lands.
    """

    name = "bootstrap"

    def score(
        self, signal_data: dict[str, Any], record_context: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        return 1.0, {
            "scorer": self.name,
            "method": "teacher_label_passthrough",
            "is_placeholder": True,
        }


_SCORER_REGISTRY: dict[str, type] = {"bootstrap": BootstrapScorer}


def get_scorer(name: str) -> ConfidenceScorer:
    try:
        cls = _SCORER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCORER_REGISTRY))
        raise ValueError(f"unknown scorer {name!r}; available: {available}") from exc
    return cls()


def average_confidence(signals: Iterable[SentenceSignal]) -> float:
    signals = list(signals)
    if not signals:
        return 0.0
    return sum(signal.confidence for signal in signals) / len(signals)


def max_severity_weighted_confidence(signals: Iterable[SentenceSignal]) -> float:
    weighted = [
        signal.confidence * float(signal.severity or 0)
        for signal in signals
    ]
    return max(weighted, default=0.0)


def adaptive_severity(signals: Iterable[SentenceSignal], alpha: float = 0.5) -> float:
    """Stable aggregation of confidence and severity for later DPO weighting."""

    signals = list(signals)
    if not signals:
        return 0.0
    mean_term = sum(signal.confidence * float(signal.severity or 0) for signal in signals) / len(signals)
    max_term = max_severity_weighted_confidence(signals)
    return alpha * mean_term + (1.0 - alpha) * max_term
