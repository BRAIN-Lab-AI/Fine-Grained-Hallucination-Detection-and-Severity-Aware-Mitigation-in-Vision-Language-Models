"""Stage 4 repair prompt template."""

from __future__ import annotations

from typing import Any


PROMPT_VERSION = "stage4_repair_v1"


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value or {})


def _format_critiques(critiques: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in critiques:
        critique = _as_dict(item)
        h_type = critique.get("hallucination_type", "unknown")
        severity = critique.get("severity_label", "unknown")
        score = critique.get("severity_score")
        evidence = (critique.get("evidence_text") or "").strip()
        rationale = (critique.get("rationale") or "").strip()
        severity_text = f"{severity}/{score}" if score is not None else str(severity)
        if evidence and rationale:
            lines.append(f"- [{h_type}; {severity_text}] {evidence}: {rationale}")
        elif rationale:
            lines.append(f"- [{h_type}; {severity_text}] {rationale}")
        elif evidence:
            lines.append(f"- [{h_type}; {severity_text}] {evidence}")
    return "\n".join(lines) if lines else "- (no structured critiques)"


def _format_votes(votes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in votes:
        vote = _as_dict(item)
        decision = "approved" if vote.get("approved") else "rejected"
        criterion = vote.get("criterion", "unknown")
        family = vote.get("model_family") or vote.get("backend_name") or "judge"
        reason = (vote.get("reason") or "").strip()
        if reason:
            lines.append(f"- vote {vote.get('vote_index')}: {decision} by {family} on {criterion}: {reason}")
        else:
            lines.append(f"- vote {vote.get('vote_index')}: {decision} by {family} on {criterion}")
    return "\n".join(lines) if lines else "- (no Stage 3 votes available)"


def build_repair_prompt(record: Any) -> str:
    """Build the Stage 4 repair prompt from a rejected Stage 3 audit row."""

    row = _as_dict(record)
    question = row.get("question", "") or ""
    original = row.get("original_response", "") or ""
    failed_rewrite = row.get("rewrite_response", "") or row.get("failed_rewrite_response", "") or ""
    critiques = list(row.get("critiques") or [])
    votes = list(row.get("votes") or row.get("stage3_votes") or [])

    return f"""You are repairing a hallucination-mitigation rewrite for a vision-language model.

Question:
{question}

Original response:
{original}

Previous rewrite that failed validation:
{failed_rewrite}

Stage 1 critiques:
{_format_critiques(critiques)}

Stage 3 validation feedback:
{_format_votes(votes)}

Instructions:
- Produce a corrected final answer that should be preferred over the original response.
- Fix every hallucination identified in the critiques.
- Address the Stage 3 validation reasons, especially rejected votes.
- Preserve correct supported visual content from the original response.
- Do not introduce new unsupported objects, attributes, relations, counts, actions, or identities.
- Output only the repaired answer. Do not include markdown, labels, or explanations.

Repaired response:"""
