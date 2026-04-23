from __future__ import annotations

from typing import Any


PROMPT_VERSION = "judge_v1"

_CRITERION_GUIDANCE = {
    "hallucination_removal": (
        "Decide whether the rewrite removes the hallucinated or unsupported content "
        "flagged by the critiques."
    ),
    "content_preservation": (
        "Decide whether the rewrite preserves correct supported content from the original "
        "response without damaging or deleting valid details."
    ),
    "overall_preference": (
        "Decide whether the rewrite is the better final answer overall, considering both "
        "hallucination removal and preservation of correct content."
    ),
}


def _coerce_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    return dict(record)


def _format_critiques(record: dict[str, Any]) -> str:
    critiques = record.get("critiques") or []
    if not critiques:
        return "- none"
    lines: list[str] = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        evidence = critique.get("evidence_text") or "n/a"
        lines.append(
            "- "
            f"type={critique.get('hallucination_type', 'unknown')}; "
            f"severity={critique.get('severity_label', 'unknown')}; "
            f"evidence={evidence}; "
            f"rationale={critique.get('rationale', '')}"
        )
    return "\n".join(lines)


def build_vote_prompt(record: Any, criterion: str) -> str:
    payload = _coerce_dict(record)
    original = payload.get("original_response", "") or ""
    rewrite = payload.get("rewrite_response", "") or ""
    question = payload.get("question", "") or ""
    guidance = _CRITERION_GUIDANCE.get(
        criterion,
        "Decide whether the rewrite should be preferred over the original response.",
    )
    return (
        "You are verifying a hallucination-mitigation rewrite for a vision-language model.\n"
        "Return strict JSON only with keys: approved (boolean), reason (string).\n"
        "Be conservative. Approve only when the rewrite clearly satisfies the criterion.\n\n"
        f"Criterion: {criterion}\n"
        f"Instruction: {guidance}\n\n"
        f"Question:\n{question}\n\n"
        f"Original response:\n{original}\n\n"
        f"Rewrite response:\n{rewrite}\n\n"
        "Stage 1 critiques:\n"
        f"{_format_critiques(payload)}\n"
    )


__all__ = ["PROMPT_VERSION", "build_vote_prompt"]
