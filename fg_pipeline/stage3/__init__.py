"""Stage 3 — majority-vote preference validation.

This package consumes Stage 2 rewrite JSONL and emits:

- a Stage 3 audit JSONL with 3 verification votes per candidate, and
- a clean preference JSONL compatible with the existing Stage 4 trainer.

The default backend is a deterministic heuristic verifier suitable for local
pipeline validation. The backend seam exists so stronger LLM/VLM judges can be
plugged in later without changing the Stage 3 output schema.
"""

from fg_pipeline.stage3.schemas import Stage3Record, VoteDecision
from fg_pipeline.stage3.backends import (
    HeuristicVerificationBackend,
    QwenLlavaEnsembleBackend,
    VerificationBackend,
    VerificationError,
    VOTE_COUNT,
    APPROVALS_REQUIRED,
    ENSEMBLE_VOTE_POLICY_VERSION,
    VOTE_POLICY_VERSION,
    evaluate_votes,
    get_backend,
)

__all__ = [
    "Stage3Record",
    "VoteDecision",
    "VerificationBackend",
    "VerificationError",
    "HeuristicVerificationBackend",
    "QwenLlavaEnsembleBackend",
    "VOTE_COUNT",
    "APPROVALS_REQUIRED",
    "ENSEMBLE_VOTE_POLICY_VERSION",
    "VOTE_POLICY_VERSION",
    "evaluate_votes",
    "get_backend",
]
