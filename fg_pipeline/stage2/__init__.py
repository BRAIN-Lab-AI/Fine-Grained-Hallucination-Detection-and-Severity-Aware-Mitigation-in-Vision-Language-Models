"""Stage 2 — critique-guided rewrite.

This package consumes Stage 1 JSONL and emits one corrected rewrite
candidate per hallucinated record.  Non-hallucinated rows are skipped.

The ``RewriteBackend`` protocol is the seam for plugging in a real LVLM
backend later.  The default ``TemplateRewriteBackend`` is a deterministic
smoke backend for local development and testing; it is **not**
research-quality.  ``LlavaRewriteBackend`` is the intended research path.
"""

from fg_pipeline.stage2.schemas import Stage2Record
from fg_pipeline.stage2.prompts import PROMPT_VERSION, build_rewrite_prompt
from fg_pipeline.stage2.backends import (
    RewriteBackend,
    RewriteError,
    TemplateRewriteBackend,
    LlavaRewriteBackend,
    get_backend,
)

__all__ = [
    "Stage2Record",
    "PROMPT_VERSION",
    "build_rewrite_prompt",
    "RewriteBackend",
    "RewriteError",
    "TemplateRewriteBackend",
    "LlavaRewriteBackend",
    "get_backend",
]
