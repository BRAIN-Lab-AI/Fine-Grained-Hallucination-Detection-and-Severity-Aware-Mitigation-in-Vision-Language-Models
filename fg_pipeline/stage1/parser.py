"""Permissive parser for the released fine-grained detection format.

Input rows look like::

    {
      "id": ...,
      "image": "vg/images/xxx.jpg",
      "conversations": [
        {"from": "human", "value": "<image>\\nDescription to Assess:\\n<text>"},
        {"from": "gpt",   "value": "NO HALLUCINATION"           # or
                                 "Tags:\\n<object>\\n1 . ...\\n"
                                 "Scores:\\n<object>\\n1 . Span: Major (3 points): ..."}
      ]
    }

The parser turns one such row into a :class:`Stage1Record`. It is intentionally
best-effort: mismatches between ``Tags:`` and ``Scores:`` degrade gracefully
into ``severity_label="unknown"`` / ``severity_score=None`` with warnings
recorded in ``metadata.parse_warnings``. The assessed candidate sentence
(``Description to Assess``) becomes ``response_text`` because that is the text
Stage 2 must rewrite; the raw GPT annotation payload is preserved in
``metadata["raw_annotation_text"]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from fg_pipeline.stage1.schemas import (
    SEVERITY_LABEL_TO_SCORE,
    UNKNOWN,
    VALID_HALLUCINATION_TYPES,
    CritiqueItem,
    Stage1Record,
)


NO_HALLUCINATION_MARKER = "NO HALLUCINATION"
_TAGS_HEADER = "Tags:"
_SCORES_HEADER = "Scores:"

_TYPE_HEADER_RE = re.compile(r"^\s*<\s*([A-Za-z][A-Za-z0-9_-]*)\s*>\s*$")
_NUM_PREFIX_RE = re.compile(r"^\s*\d+\s*\.\s*")
_SEVERITY_RE = re.compile(
    r"(?P<label>Minor|Moderate|Major)\s*\(\s*(?P<score>\d+)\s*points?\s*\)",
    re.IGNORECASE,
)
_IMAGE_HEADER_RE = re.compile(r"^\s*<image>\s*", re.IGNORECASE)
_DESC_PREFIX_RE = re.compile(
    r"^\s*Description\s+to\s+Assess\s*:\s*", re.IGNORECASE
)


class ParseError(ValueError):
    """Raised when a source row cannot be parsed in strict mode."""


@dataclass
class ParseResult:
    """Internal container for parser output before assembling the record."""

    record: Stage1Record
    warnings: list[str]


def extract_question(human_text: str) -> str:
    """Strip the ``<image>`` header and ``Description to Assess:`` prefix."""

    text = human_text or ""
    text = _IMAGE_HEADER_RE.sub("", text, count=1)
    text = text.lstrip("\n\r")
    text = _DESC_PREFIX_RE.sub("", text, count=1)
    return text.strip()


def _normalize_type(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key in VALID_HALLUCINATION_TYPES:
        return key
    return UNKNOWN


def _normalize_severity(score_line: Optional[str]) -> tuple[str, Optional[int], Optional[str]]:
    """Return ``(severity_label, severity_score, evidence_text)``.

    ``evidence_text`` is the text before the first colon on the score line
    when the line is well-formed; it is ``None`` otherwise.
    """

    if not score_line:
        return UNKNOWN, None, None

    body = _NUM_PREFIX_RE.sub("", score_line).strip()
    match = _SEVERITY_RE.search(body)
    if not match:
        return UNKNOWN, None, None

    label_raw = match.group("label").lower()
    severity_score = SEVERITY_LABEL_TO_SCORE.get(label_raw)
    if severity_score is None:
        # Defensive — regex already constrains label, but keep symmetry.
        return UNKNOWN, None, None

    evidence = None
    colon_idx = body.find(":", 0, match.start())
    if colon_idx >= 0:
        candidate = body[:colon_idx].strip()
        if candidate:
            evidence = candidate
    return label_raw, severity_score, evidence


def _split_sections(gpt_text: str) -> tuple[Optional[str], Optional[str]]:
    """Split the GPT payload into the (tags, scores) raw blocks.

    Returns ``(None, None)`` when the payload is not in the hallucinated
    ``Tags:`` / ``Scores:`` format.
    """

    if _TAGS_HEADER not in gpt_text or _SCORES_HEADER not in gpt_text:
        return None, None

    tags_idx = gpt_text.find(_TAGS_HEADER)
    scores_idx = gpt_text.find(_SCORES_HEADER, tags_idx)
    if scores_idx < 0 or scores_idx < tags_idx:
        return None, None

    tags_body = gpt_text[tags_idx + len(_TAGS_HEADER): scores_idx]
    scores_body = gpt_text[scores_idx + len(_SCORES_HEADER):]
    return tags_body, scores_body


def _collect_type_blocks(section_body: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Parse a section (tags or scores) into an ordered list of type blocks.

    Returns ``(blocks, unassigned_lines)`` where each block is
    ``(raw_type_label, [raw_item_lines])``. ``unassigned_lines`` captures any
    non-empty lines that appeared before the first ``<type>`` header and are
    thus not attributable to a hallucination type.
    """

    blocks: list[tuple[str, list[str]]] = []
    unassigned: list[str] = []
    current_items: Optional[list[str]] = None

    for raw_line in section_body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        header_match = _TYPE_HEADER_RE.match(line)
        if header_match:
            current_items = []
            blocks.append((header_match.group(1), current_items))
            continue
        if current_items is None:
            unassigned.append(line.strip())
            continue
        current_items.append(line.strip())

    return blocks, unassigned


def _aggregate_type_blocks(
    blocks: list[tuple[str, list[str]]],
) -> "dict[str, list[str]]":
    """Group items by normalized hallucination type, preserving appearance order."""

    ordered: dict[str, list[str]] = {}
    for raw_type, items in blocks:
        key = _normalize_type(raw_type)
        bucket = ordered.setdefault(key, [])
        bucket.extend(items)
    return ordered


def _strip_item_prefix(raw_item: str) -> str:
    return _NUM_PREFIX_RE.sub("", raw_item).strip()


def parse_detection_row(
    row: dict[str, Any],
    *,
    strict: bool = False,
) -> ParseResult:
    """Parse one released-annotation row into a Stage 1 record.

    When ``strict=True`` any mismatch between ``Tags:`` and ``Scores:`` counts,
    or unparseable malformed hallucinated rows, raise :class:`ParseError`
    instead of being folded into ``metadata.parse_warnings``.
    """

    if not isinstance(row, dict):
        raise ParseError(f"row must be a dict, got {type(row).__name__}")

    warnings: list[str] = []

    row_id = row.get("id")
    image = row.get("image")
    conversations = row.get("conversations") or []

    human_text = ""
    gpt_text = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        src = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        if src == "human" and not human_text:
            human_text = value or ""
        elif src == "gpt" and not gpt_text:
            gpt_text = value or ""

    assessed_text = extract_question(human_text)
    annotation_text = (gpt_text or "").strip()

    metadata: dict[str, Any] = {
        "source": "released_annotations",
        "raw_annotation_text": annotation_text,
    }

    if annotation_text == NO_HALLUCINATION_MARKER:
        record = Stage1Record(
            id=row_id,
            image=image,
            question=assessed_text,
            response_text=assessed_text,
            is_hallucinated=False,
            critiques=[],
            metadata=metadata,
        )
        return ParseResult(record=record, warnings=warnings)

    tags_body, scores_body = _split_sections(annotation_text)
    if tags_body is None or scores_body is None:
        msg = f"row id={row_id!r} is not NO HALLUCINATION and lacks Tags:/Scores: sections"
        if strict:
            raise ParseError(msg)
        warnings.append(msg)
        metadata["parse_warnings"] = list(warnings)
        record = Stage1Record(
            id=row_id,
            image=image,
            question=assessed_text,
            response_text=assessed_text,
            is_hallucinated=True,
            critiques=[],
            metadata=metadata,
        )
        return ParseResult(record=record, warnings=warnings)

    tag_blocks, tag_unassigned = _collect_type_blocks(tags_body)
    score_blocks, score_unassigned = _collect_type_blocks(scores_body)

    if tag_unassigned:
        warnings.append(
            f"row id={row_id!r} has {len(tag_unassigned)} Tags line(s) before any <type> header; dropped"
        )
    if score_unassigned:
        warnings.append(
            f"row id={row_id!r} has {len(score_unassigned)} Scores line(s) before any <type> header; dropped"
        )

    tags_by_type = _aggregate_type_blocks(tag_blocks)
    scores_by_type = _aggregate_type_blocks(score_blocks)

    critiques: list[CritiqueItem] = []
    global_index = 0
    seen_types: set[str] = set()

    # Iterate in the order types first appear in the Tags section.
    tag_type_order: list[str] = []
    for raw_type, _ in tag_blocks:
        key = _normalize_type(raw_type)
        if key not in tag_type_order:
            tag_type_order.append(key)

    for h_type in tag_type_order:
        seen_types.add(h_type)
        tag_items = tags_by_type.get(h_type, [])
        score_items = scores_by_type.get(h_type, [])

        if len(tag_items) != len(score_items):
            warnings.append(
                f"row id={row_id!r} type={h_type} has {len(tag_items)} tag(s) "
                f"but {len(score_items)} score(s); recoverable"
            )

        for idx, tag_line in enumerate(tag_items):
            score_line = score_items[idx] if idx < len(score_items) else None
            rationale = _strip_item_prefix(tag_line)
            label, severity_score, evidence = _normalize_severity(score_line)
            if score_line is None and strict:
                raise ParseError(
                    f"row id={row_id!r} type={h_type} missing score at position {idx}"
                )
            if score_line is not None and label == UNKNOWN and strict:
                raise ParseError(
                    f"row id={row_id!r} type={h_type} score malformed at position {idx}: {score_line!r}"
                )
            global_index += 1
            critiques.append(
                CritiqueItem(
                    index=global_index,
                    hallucination_type=h_type,
                    severity_label=label,
                    severity_score=severity_score,
                    rationale=rationale,
                    evidence_text=evidence,
                    source_tag_text=tag_line,
                    source_score_text=score_line,
                )
            )

    # Any types that appear only in Scores are warnings — no tag to attach them to.
    for raw_type, items in score_blocks:
        key = _normalize_type(raw_type)
        if key not in seen_types and items:
            warnings.append(
                f"row id={row_id!r} Scores section has type={key} with no matching Tags; {len(items)} item(s) dropped"
            )

    if strict and warnings:
        raise ParseError(f"strict parse failed for row id={row_id!r}: {warnings[0]}")

    is_hallucinated = True
    if warnings:
        metadata["parse_warnings"] = list(warnings)

    record = Stage1Record(
        id=row_id,
        image=image,
        question=assessed_text,
        response_text=assessed_text,
        is_hallucinated=is_hallucinated,
        critiques=critiques,
        metadata=metadata,
    )
    return ParseResult(record=record, warnings=warnings)
