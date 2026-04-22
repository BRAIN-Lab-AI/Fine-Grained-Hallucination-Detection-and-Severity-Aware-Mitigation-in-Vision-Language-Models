"""Stage 3 verification backends.

The default backend is a deterministic heuristic verifier that produces three
independent votes:

1. hallucination_removal
2. content_preservation
3. overall_preference

This keeps Stage 3 runnable locally without external services while preserving
the 3-vote majority contract. Stronger judge backends can be added later behind
the same protocol.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from fg_pipeline.stage3.schemas import VoteDecision


VOTE_COUNT = 3
APPROVALS_REQUIRED = 2
VOTE_POLICY_VERSION = "heuristic_v1"

_CRITERIA = {
    1: "hallucination_removal",
    2: "content_preservation",
    3: "overall_preference",
}

_WORD_RE = re.compile(r"\w+")


@runtime_checkable
class VerificationBackend(Protocol):
    """Minimal Stage-3-facing contract for any verification backend."""

    name: str

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        """Return one vote over a Stage 2 rewrite candidate."""
        ...


class VerificationError(ValueError):
    """Raised when Stage 3 verification cannot proceed in strict mode."""


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str | None) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _extract_fields(record: Any) -> tuple[str, str, list[dict[str, Any]]]:
    if hasattr(record, "original_response"):
        original = record.original_response or ""
        rewrite = record.rewrite_response or ""
        critiques = list(record.critiques or [])
    else:
        original = record.get("original_response", "") or ""
        rewrite = record.get("rewrite_response", "") or ""
        critiques = list(record.get("critiques") or [])
    return original, rewrite, critiques


def _evidence_terms(critiques: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        evidence = (critique.get("evidence_text") or "").strip()
        if evidence:
            terms.append(evidence)
    return terms


class HeuristicVerificationBackend:
    """Deterministic local verifier for Stage 3.

    This backend is deliberately simple. It is useful for pipeline bring-up and
    tests, not as a final research judge.
    """

    name = "heuristic"
    policy_version = VOTE_POLICY_VERSION

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _CRITERIA:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..{VOTE_COUNT}")

        original, rewrite, critiques = _extract_fields(record)
        original_norm = _normalize_text(original)
        rewrite_norm = _normalize_text(rewrite)
        if not original_norm or not rewrite_norm:
            raise VerificationError("Stage 2 record is missing original_response or rewrite_response")

        changed = original_norm != rewrite_norm
        evidence_terms = _evidence_terms(critiques)
        present_terms = [
            term for term in evidence_terms if term and re.search(r"\b" + re.escape(term) + r"\b", original, flags=re.IGNORECASE)
        ]
        removed_terms = [
            term for term in present_terms if not re.search(r"\b" + re.escape(term) + r"\b", rewrite, flags=re.IGNORECASE)
        ]
        corrected_marker = "[corrected]" in rewrite_norm

        original_tokens = _tokenize(original)
        rewrite_tokens = _tokenize(rewrite)
        overlap = len(set(original_tokens) & set(rewrite_tokens))
        overlap_ratio = overlap / len(set(original_tokens) or {""})
        length_ratio = len(rewrite_tokens) / max(len(original_tokens), 1)

        criterion = _CRITERIA[vote_index]
        approved = False
        reason = ""

        if criterion == "hallucination_removal":
            if present_terms:
                required = max(1, (len(present_terms) + 1) // 2)
                approved = changed and len(removed_terms) >= required
                reason = (
                    f"removed {len(removed_terms)}/{len(present_terms)} evidence span(s); "
                    f"changed={changed}"
                )
            else:
                approved = changed and not corrected_marker
                reason = f"no evidence spans available; changed={changed}; corrected_marker={corrected_marker}"

        elif criterion == "content_preservation":
            approved = (
                changed
                and not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            reason = (
                f"length_ratio={length_ratio:.3f}; overlap_ratio={overlap_ratio:.3f}; "
                f"changed={changed}; corrected_marker={corrected_marker}"
            )

        elif criterion == "overall_preference":
            removal_ok = (
                (len(removed_terms) > 0 if present_terms else changed and not corrected_marker)
            )
            preservation_ok = (
                not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            approved = changed and removal_ok and preservation_ok
            reason = (
                f"removal_ok={removal_ok}; preservation_ok={preservation_ok}; "
                f"changed={changed}"
            )

        return VoteDecision(
            vote_index=vote_index,
            criterion=criterion,
            approved=approved,
            reason=reason,
        )


def get_backend(name: str) -> VerificationBackend:
    key = (name or "").strip().lower()
    if key == HeuristicVerificationBackend.name:
        return HeuristicVerificationBackend()
    raise ValueError(f"unknown stage3 backend {name!r}; available: {HeuristicVerificationBackend.name}")


__all__ = [
    "VerificationBackend",
    "VerificationError",
    "HeuristicVerificationBackend",
    "VOTE_COUNT",
    "APPROVALS_REQUIRED",
    "VOTE_POLICY_VERSION",
    "get_backend",
]
