from __future__ import annotations

import json
import os
from typing import Any, Iterable

import requests

from fg_pipeline.eval.utils import count_tokens, mean_or_none, safe_div


HSS_RUBRIC = {
    0: "None: No hallucination.",
    1: "Minor: incorrect minor detail that does not significantly affect the scene.",
    2: "Moderate: noticeable incorrect detail while the overall scene remains understandable.",
    3: "Major: significant fabricated or wrong detail that fundamentally alters the scene.",
}


def _openai_request(messages: list[dict[str, str]], model: str) -> tuple[dict[str, Any], str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for judge-based metrics")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": messages,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    return json.loads(text), text


def judge_hss_rows(rows: Iterable[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(rows)
    results: list[dict[str, Any]] = []
    raw_successes = 0
    for row in rows:
        system = (
            "You evaluate hallucination severity in model image responses. "
            "Return strict JSON only."
        )
        user = (
            "Rate the hallucination severity of the response using this rubric:\n"
            + "\n".join(f"{score}: {text}" for score, text in HSS_RUBRIC.items())
            + "\n\nQuestion:\n"
            + str(row.get("question", ""))
            + "\n\nResponse:\n"
            + str(row.get("text", ""))
            + '\n\nReturn JSON with keys "severity_label", "severity_score", and "reason".'
        )
        parsed, raw = _openai_request(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
        )
        score = int(parsed["severity_score"])
        raw_successes += 1
        results.append(
            {
                "id": row.get("id"),
                "severity_label": parsed.get("severity_label"),
                "severity_score": score,
                "reason": parsed.get("reason"),
                "raw_response": raw,
            }
        )
    scores = [int(result["severity_score"]) for result in results]
    lengths = [count_tokens(row.get("text")) for row in rows]
    summary = {
        "avg_hss": mean_or_none(scores),
        "sum_hss": float(sum(scores)),
        "avg_response_length": mean_or_none(lengths),
        "num_judged": raw_successes,
    }
    return results, summary


def judge_mmhal_rows(rows: Iterable[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        system = "You score multimodal hallucination responses. Return strict JSON only."
        user = (
            "Evaluate the answer for hallucination.\n"
            f"Question:\n{row.get('question', '')}\n\n"
            f"Answer:\n{row.get('text', '')}\n\n"
            'Return JSON with keys "score" (1-5), "hallucination" (true/false), and "reason".'
        )
        parsed, raw = _openai_request(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
        )
        results.append(
            {
                "id": row.get("id"),
                "score": float(parsed["score"]),
                "hallucination": bool(parsed["hallucination"]),
                "reason": parsed.get("reason"),
                "raw_response": raw,
            }
        )
    hallucinated = sum(1 for row in results if row["hallucination"])
    summary = {
        "overall": mean_or_none([float(row["score"]) for row in results]),
        "response_hallucination_rate": safe_div(hallucinated, len(results)),
        "num_judged": len(results),
    }
    return results, summary


def judge_llava_bench_rows(
    rows: Iterable[dict[str, Any]],
    model: str,
    *,
    contexts: dict[str, str] | None = None,
    references: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("id"))
        system = "You review visual assistant answers. Return strict JSON only."
        user = (
            f"Question:\n{row.get('question', '')}\n\n"
            f"Context:\n{(contexts or {}).get(row_id, '')}\n\n"
            f"Reference answer:\n{(references or {}).get(row_id, '')}\n\n"
            f"Candidate answer:\n{row.get('text', '')}\n\n"
            'Return JSON with keys "score" (1-10), "reason", and "passes".'
        )
        parsed, raw = _openai_request(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
        )
        results.append(
            {
                "id": row.get("id"),
                "score": float(parsed["score"]),
                "passes": bool(parsed.get("passes", True)),
                "reason": parsed.get("reason"),
                "raw_response": raw,
            }
        )
    summary = {
        "overall": mean_or_none([float(item["score"]) for item in results]),
        "num_reviewed": len(results),
        "judge_model": model,
    }
    return results, summary
