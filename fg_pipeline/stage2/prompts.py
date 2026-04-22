"""Stage 2 rewrite prompt template.

``PROMPT_VERSION`` is written into every Stage 2 record's metadata so that
output JSONL is always traceable to a specific prompt revision.

The template is intentionally backend-agnostic: both ``TemplateRewriteBackend``
(smoke) and ``LlavaRewriteBackend`` (real) derive the same conceptual
instruction from :func:`build_rewrite_prompt`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fg_pipeline.stage1.schemas import Stage1Record, CritiqueItem


PROMPT_VERSION = "v1"

_REWRITE_TEMPLATE = """\
You are a precise editor for image descriptions.

Original response:
{original_response}

Identified hallucinations:
{critique_lines}

Instructions:
- Rewrite the original response to correct or remove every identified hallucination.
- Preserve all accurate content.
- Do not introduce new claims that are not supported by the original response.
- Produce a single concise corrected sentence or short paragraph.
- Do not add preamble or explanation — output only the corrected response.

Corrected response:\
"""


def _format_critique_lines(critiques) -> str:
    lines: list[str] = []
    for c in critiques:
        if hasattr(c, "to_dict"):
            c = c.to_dict()
        h_type = c.get("hallucination_type", "unknown")
        severity = c.get("severity_label", "unknown")
        rationale = c.get("rationale", "")
        evidence = c.get("evidence_text")
        if evidence:
            lines.append(f"- [{h_type}/{severity}] {evidence}: {rationale}")
        else:
            lines.append(f"- [{h_type}/{severity}] {rationale}")
    return "\n".join(lines) if lines else "- (no structured critiques)"


def build_rewrite_prompt(record) -> str:
    """Build the Stage 2 rewrite prompt for a hallucinated Stage 1 record.

    ``record`` can be a :class:`~fg_pipeline.stage1.schemas.Stage1Record`
    dataclass or a plain ``dict`` with the same fields.
    """

    if hasattr(record, "response_text"):
        original = record.response_text
        critiques = record.critiques
    else:
        original = record.get("response_text", "")
        critiques = record.get("critiques", [])

    critique_lines = _format_critique_lines(critiques)
    return _REWRITE_TEMPLATE.format(
        original_response=original,
        critique_lines=critique_lines,
    )
