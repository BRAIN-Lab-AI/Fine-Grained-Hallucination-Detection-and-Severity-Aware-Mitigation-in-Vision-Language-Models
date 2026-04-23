from __future__ import annotations

from typing import Any

from fg_pipeline.stage1.prompts import build_detector_prompt, coerce_stage1_inputs


def build_llava_detector_example(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one released supervision row into a LLaVA SFT example.

    The human turn uses the same prompt template as LlavaDetectorBackend at
    inference time so that training and inference inputs are identical.
    """

    row_id = row.get("id")
    image = row.get("image")
    question, response_text = coerce_stage1_inputs(row)
    annotation_text = ""
    conversations = row.get("conversations") or []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).lower() == "gpt":
            annotation_text = str(turn.get("value") or "").strip()
            break
    return {
        "id": row_id,
        "image": image,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + build_detector_prompt(question=question, response_text=response_text),
            },
            {
                "from": "gpt",
                "value": annotation_text,
            },
        ],
    }


def prediction_for_mhalubench(stage1_record: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_row)
    is_hallucinated = bool(stage1_record.get("is_hallucinated"))
    payload["claim_prediction"] = 1 if is_hallucinated else 0
    payload["segment_prediction"] = 1 if is_hallucinated else 0
    payload["detector_source"] = stage1_record.get("metadata", {}).get("source")
    return payload


def prediction_for_mfhallubench(stage1_record: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source_row)
    critiques = list(stage1_record.get("critiques") or [])
    top = critiques[0] if critiques else {}
    if hasattr(top, "to_dict"):
        top = top.to_dict()
    is_hallucinated = bool(stage1_record.get("is_hallucinated"))
    payload["binary_prediction"] = 1 if is_hallucinated else 0
    payload["type_prediction"] = top.get("hallucination_type", "none") if is_hallucinated else "none"
    payload["severity_prediction"] = top.get("severity_label", "none") if is_hallucinated else "none"
    payload["multi_prediction"] = (
        f"{payload['type_prediction']}:{payload['severity_prediction']}"
        if is_hallucinated
        else "none"
    )
    payload["detector_source"] = stage1_record.get("metadata", {}).get("source")
    return payload


__all__ = [
    "build_llava_detector_example",
    "prediction_for_mhalubench",
    "prediction_for_mfhallubench",
]
