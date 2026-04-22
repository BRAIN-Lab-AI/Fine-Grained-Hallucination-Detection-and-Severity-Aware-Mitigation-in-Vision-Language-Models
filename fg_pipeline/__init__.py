"""Extension pipeline for fine-grained hallucination mitigation.

Current state:

- Stage 1 (``fg_pipeline.stage1``) — critique detection / critique extraction
  over the released fine-grained supervision. Default backend is
  :class:`~fg_pipeline.stage1.ReleasedAnnotationBackend`; a real trained
  detector can be plugged in later behind the same
  :class:`~fg_pipeline.stage1.CritiqueDetectorBackend` protocol.
- Stage 2 (``fg_pipeline.stage2``) — critique-guided rewrite. Consumes
  Stage 1 JSONL and emits one corrected rewrite per hallucinated record.
  Default backend is :class:`~fg_pipeline.stage2.TemplateRewriteBackend`
  (smoke, deterministic, non-research); the intended research backend is
  :class:`~fg_pipeline.stage2.LlavaRewriteBackend`.
- Stage 3 (``fg_pipeline.stage3``) — majority-vote preference validation.
  Consumes Stage 2 JSONL, emits a Stage 3 audit JSONL plus clean trainer-
  compatible preference pairs. Default backend is
  :class:`~fg_pipeline.stage3.HeuristicVerificationBackend`.
- Stage 4 (severity-aware DPO) is the released HSA-DPO baseline trainer
  reached via ``hsa_dpo_train.sh`` or the project wrapper
  ``scripts/run_stage4_train.sh``.

Shared utilities live in ``io_utils``, ``paths``, and ``schemas``.
"""

from fg_pipeline.schemas import PreferenceCleanRecord
from fg_pipeline.stage1 import (
    CritiqueDetectorBackend,
    CritiqueItem,
    ReleasedAnnotationBackend,
    Stage1Record,
)
from fg_pipeline.stage2 import (
    LlavaRewriteBackend,
    RewriteBackend,
    Stage2Record,
    TemplateRewriteBackend,
)
from fg_pipeline.stage3 import (
    HeuristicVerificationBackend,
    Stage3Record,
    VerificationBackend,
    VoteDecision,
)

__all__ = [
    "io_utils",
    "paths",
    "schemas",
    "stage1",
    "stage2",
    "stage3",
    "PreferenceCleanRecord",
    "Stage1Record",
    "CritiqueItem",
    "CritiqueDetectorBackend",
    "ReleasedAnnotationBackend",
    "Stage2Record",
    "RewriteBackend",
    "TemplateRewriteBackend",
    "LlavaRewriteBackend",
    "Stage3Record",
    "VoteDecision",
    "VerificationBackend",
    "HeuristicVerificationBackend",
]
