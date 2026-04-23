from __future__ import annotations

from typing import Any

from fg_pipeline.stage1.parser import extract_question


PROMPT_VERSION = "detector_v1"


def build_detector_prompt(*, question: str, response_text: str) -> str:
    prompt_question = question or response_text
    assessed = response_text or question
    return (
        "You are a fine-grained hallucination detector for a vision-language model.\n"
        "Given an image and a candidate response, return exactly one of these formats:\n"
        "1. NO HALLUCINATION\n"
        "2. A Tags/Scores report matching this structure:\n"
        "Tags:\n"
        "<object>\n"
        "1. ...\n"
        "<attribute>\n"
        "1. ...\n"
        "<relationship>\n"
        "1. ...\n"
        "Scores:\n"
        "<object>\n"
        "1. Evidence span: Major (3 points): ...\n"
        "...\n\n"
        "Rules:\n"
        "- Use only object, attribute, and relationship types.\n"
        "- Use severity labels Minor, Moderate, or Major.\n"
        "- If there is no hallucination, return NO HALLUCINATION exactly.\n"
        "- Do not add extra commentary before or after the required output.\n\n"
        f"Prompt or task context:\n{prompt_question}\n\n"
        f"Candidate response to assess:\n{assessed}\n"
    )


def coerce_stage1_inputs(row: dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or "").strip()
    response_text = str(
        row.get("response_text")
        or row.get("candidate_response")
        or row.get("answer")
        or row.get("text")
        or question
        or ""
    ).strip()
    if not response_text:
        conversations = row.get("conversations") or []
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if str(turn.get("from", "")).lower() == "human":
                response_text = extract_question(str(turn.get("value") or ""))
                break
    if not question:
        question = response_text
    return question, response_text


__all__ = ["PROMPT_VERSION", "build_detector_prompt", "coerce_stage1_inputs"]
