from __future__ import annotations

import re
from typing import Optional

# Severity labels -> HS^j integer score used paper-side.
SEVERITY_TO_INT = {"minor": 1, "moderate": 2, "major": 3}
VALID_TYPES = {"object", "attribute", "relationship"}

_TYPE_HEADER_RE = re.compile(r"^\s*<(object|attribute|relationship)>\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*(Tags|Scores)\s*:\s*$", re.IGNORECASE)
_SCORE_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:\d+\s*\.\s*)?                         # optional "N ." prefix
    (?P<span>[^:]+?)\s*:\s*                  # span text up to first colon
    (?P<severity>Minor|Moderate|Major)\s*    # severity label
    \(\s*(?P<points>\d+)\s*points?\s*\)      # "(N points)"
    \s*:?\s*
    (?P<rationale>.*?)\s*$                   # remainder = rationale
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TAG_LINE_RE = re.compile(r"^\s*(?:(?P<idx>\d+)\s*\.\s*)?(?P<text>.+?)\s*$")


def _split_sections(text: str) -> tuple[str, str]:
    """Return (tags_block, scores_block) as raw strings. Either may be empty."""

    tags_lines: list[str] = []
    scores_lines: list[str] = []
    current: Optional[str] = None
    for line in text.splitlines():
        section = _SECTION_RE.match(line)
        if section:
            current = section.group(1).lower()
            continue
        if current == "tags":
            tags_lines.append(line)
        elif current == "scores":
            scores_lines.append(line)
    return "\n".join(tags_lines), "\n".join(scores_lines)


def _iter_type_blocks(block: str):
    """Yield (type_name, [line, ...]) pairs in document order."""

    current_type: Optional[str] = None
    buffer: list[str] = []
    for line in block.splitlines():
        header = _TYPE_HEADER_RE.match(line)
        if header:
            if current_type is not None:
                yield current_type, buffer
            current_type = header.group(1).lower()
            buffer = []
        else:
            if current_type is not None:
                buffer.append(line)
    if current_type is not None:
        yield current_type, buffer


def _collect_tags(tags_block: str) -> dict[str, list[str]]:
    """Map type -> ordered list of tag description strings."""

    collected: dict[str, list[str]] = {}
    for type_name, lines in _iter_type_blocks(tags_block):
        bucket = collected.setdefault(type_name, [])
        for line in lines:
            if not line.strip():
                continue
            match = _TAG_LINE_RE.match(line)
            if match and match.group("text"):
                bucket.append(match.group("text").strip())
    return collected


def parse_detection_response(raw_response: str) -> list[dict]:
    """Parse a teacher-annotated detection response into a list of signal dicts.

    Returns [] for ``NO HALLUCINATION`` or when no scored signals are found.
    Each signal dict has keys: sentence_index, hallucination_type, severity,
    severity_label, rationale, raw_label, tag_text.
    """

    if raw_response is None:
        return []
    text = raw_response.strip()
    if not text or text.upper() == "NO HALLUCINATION":
        return []

    tags_block, scores_block = _split_sections(text)
    tags_by_type = _collect_tags(tags_block)
    tag_cursor: dict[str, int] = {t: 0 for t in tags_by_type}

    signals: list[dict] = []
    for type_name, lines in _iter_type_blocks(scores_block):
        if type_name not in VALID_TYPES:
            continue
        for line in lines:
            if not line.strip():
                continue
            match = _SCORE_LINE_RE.match(line)
            if not match:
                continue
            severity_label = match.group("severity").lower()
            severity_int = SEVERITY_TO_INT.get(severity_label)
            if severity_int is None:
                continue
            rationale = match.group("rationale").strip() or None
            span = match.group("span").strip()

            tag_text: Optional[str] = None
            bucket = tags_by_type.get(type_name, [])
            cursor = tag_cursor.get(type_name, 0)
            if cursor < len(bucket):
                tag_text = bucket[cursor]
                tag_cursor[type_name] = cursor + 1

            signals.append(
                {
                    "sentence_index": len(signals),
                    "hallucination_type": type_name,
                    "severity": severity_int,
                    "severity_label": severity_label,
                    "rationale": rationale,
                    "raw_label": line.strip(),
                    "tag_text": tag_text,
                    "span": span,
                }
            )
    return signals
