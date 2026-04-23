"""Stage 3 verification backends.

The default backend remains a deterministic heuristic verifier suitable for
local smoke runs. The research backend is a fully local ``Qwen + LLaVA``
ensemble that preserves the fixed 3-vote contract without using any hosted
judge API.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fg_pipeline.stage3.prompts import PROMPT_VERSION, build_vote_prompt
from fg_pipeline.stage3.schemas import VoteDecision


VOTE_COUNT = 3
APPROVALS_REQUIRED = 2
VOTE_POLICY_VERSION = "heuristic_v1"
ENSEMBLE_VOTE_POLICY_VERSION = "qwen_llava_ensemble_v1"

_CRITERIA = {
    1: "hallucination_removal",
    2: "content_preservation",
    3: "overall_preference",
}
_ENSEMBLE_FAMILY_BY_VOTE = {
    1: "qwen",
    2: "llava",
    3: "qwen",
}

_WORD_RE = re.compile(r"\w+")


@runtime_checkable
class VerificationBackend(Protocol):
    """Minimal Stage-3-facing contract for any verification backend."""

    name: str
    policy_version: str
    approval_families_required: tuple[str, ...]

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        """Return one vote over a Stage 2 rewrite candidate."""
        ...


class VerificationError(ValueError):
    """Raised when Stage 3 verification cannot proceed in strict mode."""


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _tokenize(text: str | None) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _extract_fields(record: Any) -> tuple[str, str, list[dict[str, Any]]]:
    if hasattr(record, "original_response"):
        original = record.original_response or ""
        rewrite = record.rewrite_response or ""
        critiques = list(record.critiques or [])
    else:
        original = record.get("original_response", "") or ""
        rewrite = record.get("rewrite_response", "") or ""
        critiques = list(record.get("critiques") or [])
    return original, rewrite, critiques


def _evidence_terms(critiques: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for critique in critiques:
        if hasattr(critique, "to_dict"):
            critique = critique.to_dict()
        evidence = (critique.get("evidence_text") or "").strip()
        if evidence:
            terms.append(evidence)
    return terms


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if not candidate:
        raise ValueError("empty judge response")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON is not an object")
    return payload


def _parse_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "approve", "approved", "pass"}:
            return True
        if lowered in {"false", "no", "reject", "rejected", "fail"}:
            return False
    raise ValueError(f"cannot coerce {value!r} to bool")


def _build_parse_failure_vote(
    *,
    vote_index: int,
    criterion: str,
    backend_name: str,
    model_family: str,
    error: Exception,
) -> VoteDecision:
    return VoteDecision(
        vote_index=vote_index,
        criterion=criterion,
        approved=False,
        reason=f"judge parse failure: {error}",
        model_family=model_family,
        backend_name=backend_name,
    )


class HeuristicVerificationBackend:
    """Deterministic local verifier for Stage 3.

    This backend is deliberately simple. It is useful for pipeline bring-up and
    tests, not as a final research judge.
    """

    name = "heuristic"
    policy_version = VOTE_POLICY_VERSION
    approval_families_required: tuple[str, ...] = ()

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _CRITERIA:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..{VOTE_COUNT}")

        original, rewrite, critiques = _extract_fields(record)
        original_norm = _normalize_text(original)
        rewrite_norm = _normalize_text(rewrite)
        if not original_norm or not rewrite_norm:
            raise VerificationError("Stage 2 record is missing original_response or rewrite_response")

        changed = original_norm != rewrite_norm
        evidence_terms = _evidence_terms(critiques)
        present_terms = [
            term for term in evidence_terms if term and re.search(r"\b" + re.escape(term) + r"\b", original, flags=re.IGNORECASE)
        ]
        removed_terms = [
            term for term in present_terms if not re.search(r"\b" + re.escape(term) + r"\b", rewrite, flags=re.IGNORECASE)
        ]
        corrected_marker = "[corrected]" in rewrite_norm

        original_tokens = _tokenize(original)
        rewrite_tokens = _tokenize(rewrite)
        overlap = len(set(original_tokens) & set(rewrite_tokens))
        overlap_ratio = overlap / len(set(original_tokens) or {""})
        length_ratio = len(rewrite_tokens) / max(len(original_tokens), 1)

        criterion = _CRITERIA[vote_index]
        approved = False
        reason = ""

        if criterion == "hallucination_removal":
            if present_terms:
                required = max(1, (len(present_terms) + 1) // 2)
                approved = changed and len(removed_terms) >= required
                reason = (
                    f"removed {len(removed_terms)}/{len(present_terms)} evidence span(s); "
                    f"changed={changed}"
                )
            else:
                approved = changed and not corrected_marker
                reason = f"no evidence spans available; changed={changed}; corrected_marker={corrected_marker}"

        elif criterion == "content_preservation":
            approved = (
                changed
                and not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            reason = (
                f"length_ratio={length_ratio:.3f}; overlap_ratio={overlap_ratio:.3f}; "
                f"changed={changed}; corrected_marker={corrected_marker}"
            )

        elif criterion == "overall_preference":
            removal_ok = (
                (len(removed_terms) > 0 if present_terms else changed and not corrected_marker)
            )
            preservation_ok = (
                not corrected_marker
                and 0.4 <= length_ratio <= 1.4
                and overlap_ratio >= 0.4
            )
            approved = changed and removal_ok and preservation_ok
            reason = (
                f"removal_ok={removal_ok}; preservation_ok={preservation_ok}; "
                f"changed={changed}"
            )

        return VoteDecision(
            vote_index=vote_index,
            criterion=criterion,
            approved=approved,
            reason=reason,
            model_family="heuristic",
            backend_name=self.name,
        )


class _QwenJudgeRuntime:
    family = "qwen"

    def __init__(
        self,
        model_path: str,
        *,
        image_root: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._bundle: tuple[Any, Any] | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._bundle is not None:
            return self._bundle
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        self._bundle = (tokenizer, model)
        return self._bundle

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def judge(self, record: dict[str, Any], criterion: str) -> str:
        import torch

        tokenizer, model = self._load()
        prompt = build_vote_prompt(record, criterion)
        image_path = self._resolved_image(record)

        if hasattr(tokenizer, "from_list_format") and hasattr(model, "chat"):
            if image_path:
                query = tokenizer.from_list_format(
                    [{"image": str(image_path)}, {"text": prompt}]
                )
            else:
                query = prompt
            response, _ = model.chat(tokenizer, query=query, history=None)
            return str(response).strip()

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._temperature > 0.0,
        }
        if self._temperature > 0.0:
            generate_kwargs["temperature"] = self._temperature
        else:
            generate_kwargs["num_beams"] = 1
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)
        return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class _LlavaJudgeRuntime:
    family = "llava"

    def __init__(
        self,
        model_path: str,
        *,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self._model_path = model_path
        self._model_base = model_base
        self._conv_mode = conv_mode
        self._image_root = Path(image_root) if image_root else Path(".")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._bundle: tuple[Any, Any, Any] | None = None

    def _ensure_llava_on_path(self) -> None:
        llava_root = (
            Path(__file__).resolve().parent.parent.parent
            / "hsa_dpo"
            / "models"
            / "llava-v1_5"
        )
        path_str = str(llava_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    def _load(self) -> tuple[Any, Any, Any]:
        if self._bundle is not None:
            return self._bundle
        self._ensure_llava_on_path()
        from llava.model.builder import load_pretrained_model  # type: ignore[import]
        from llava.mm_utils import get_model_name_from_path  # type: ignore[import]

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=self._model_path,
            model_base=self._model_base,
            model_name=get_model_name_from_path(self._model_path),
        )
        model.eval()
        self._bundle = (tokenizer, model, image_processor)
        return self._bundle

    def _resolved_image(self, record: dict[str, Any]) -> Path | None:
        image_path = record.get("image")
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def judge(self, record: dict[str, Any], criterion: str) -> str:
        import torch
        from PIL import Image as PILImage

        tokenizer, model, image_processor = self._load()
        self._ensure_llava_on_path()
        from llava.constants import (  # type: ignore[import]
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates  # type: ignore[import]
        from llava.mm_utils import process_images, tokenizer_image_token  # type: ignore[import]

        prompt_text = build_vote_prompt(record, criterion)
        image_path = self._resolved_image(record)
        image_tensor = None
        use_image = False
        if image_path:
            pil_image = PILImage.open(image_path).convert("RGB")
            image_tensor = process_images([pil_image], image_processor, model.config).to(
                model.device,
                dtype=torch.float16,
            )
            use_image = True

        if use_image:
            if getattr(model.config, "mm_use_im_start_end", False):
                user_prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt_text
                )
            else:
                user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        else:
            user_prompt = prompt_text

        conv = conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        if use_image:
            input_ids = tokenizer_image_token(
                full_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
        else:
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

        generate_kwargs: dict[str, Any] = {
            "do_sample": self._temperature > 0.0,
            "max_new_tokens": self._max_new_tokens,
            "use_cache": True,
        }
        if self._temperature > 0.0:
            generate_kwargs["temperature"] = self._temperature
        else:
            generate_kwargs["num_beams"] = 1
        if use_image and image_tensor is not None:
            generate_kwargs["images"] = image_tensor

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generate_kwargs)
        return tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()


class QwenLlavaEnsembleBackend:
    """Local research backend with fixed Qwen/LLaVA vote ownership.

    Vote 1: Qwen hallucination removal
    Vote 2: LLaVA content preservation
    Vote 3: Qwen overall preference

    A pair passes only when at least two votes approve *and* there is at least
    one approved vote from each model family.
    """

    name = "qwen_llava_ensemble"
    policy_version = ENSEMBLE_VOTE_POLICY_VERSION
    approval_families_required = ("qwen", "llava")

    def __init__(
        self,
        *,
        qwen_model_path: str,
        llava_model_path: str,
        llava_model_base: str | None = None,
        llava_conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        qwen_max_new_tokens: int = 256,
        llava_max_new_tokens: int = 256,
        qwen_temperature: float = 0.0,
        llava_temperature: float = 0.0,
        qwen_runtime: Any | None = None,
        llava_runtime: Any | None = None,
    ) -> None:
        self._qwen = qwen_runtime or _QwenJudgeRuntime(
            qwen_model_path,
            image_root=image_root,
            max_new_tokens=qwen_max_new_tokens,
            temperature=qwen_temperature,
        )
        self._llava = llava_runtime or _LlavaJudgeRuntime(
            llava_model_path,
            model_base=llava_model_base,
            conv_mode=llava_conv_mode,
            image_root=image_root,
            max_new_tokens=llava_max_new_tokens,
            temperature=llava_temperature,
        )

    def vote(self, record: dict[str, Any], *, vote_index: int, strict: bool = False) -> VoteDecision:
        if vote_index not in _CRITERIA:
            raise VerificationError(f"unsupported vote_index={vote_index}; expected 1..{VOTE_COUNT}")
        criterion = _CRITERIA[vote_index]
        family = _ENSEMBLE_FAMILY_BY_VOTE[vote_index]
        runtime = self._qwen if family == "qwen" else self._llava
        try:
            raw = runtime.judge(record, criterion)
            payload = _extract_json_object(raw)
            approved = _parse_boolean(payload.get("approved"))
            reason = str(payload.get("reason") or "").strip() or "no reason provided"
            return VoteDecision(
                vote_index=vote_index,
                criterion=criterion,
                approved=approved,
                reason=reason,
                model_family=family,
                backend_name=self.name,
            )
        except Exception as exc:
            if strict:
                raise VerificationError(f"{family} judge failed for vote {vote_index}: {exc}") from exc
            return _build_parse_failure_vote(
                vote_index=vote_index,
                criterion=criterion,
                backend_name=self.name,
                model_family=family,
                error=exc,
            )


def evaluate_votes(backend: VerificationBackend, votes: list[VoteDecision]) -> tuple[bool, dict[str, Any]]:
    approvals = sum(1 for vote in votes if vote.approved)
    approved_families = sorted(
        {vote.model_family for vote in votes if vote.approved and vote.model_family}
    )
    required_families = tuple(getattr(backend, "approval_families_required", ()) or ())
    families_ok = all(family in approved_families for family in required_families)
    passed = approvals >= APPROVALS_REQUIRED and families_ok
    return passed, {
        "approved_families": approved_families,
        "approval_families_required": list(required_families),
        "cross_family_approval_ok": families_ok,
        "prompt_version": PROMPT_VERSION,
    }


def get_backend(name: str, **kwargs: Any) -> VerificationBackend:
    key = (name or "").strip().lower()
    if key == HeuristicVerificationBackend.name:
        return HeuristicVerificationBackend()
    if key == QwenLlavaEnsembleBackend.name:
        qwen_model_path = kwargs.get("qwen_model_path") or ""
        llava_model_path = kwargs.get("llava_model_path") or ""
        if not qwen_model_path:
            raise ValueError("qwen_llava_ensemble requires --qwen-model-path")
        if not llava_model_path:
            raise ValueError("qwen_llava_ensemble requires --llava-model-path")
        return QwenLlavaEnsembleBackend(
            qwen_model_path=qwen_model_path,
            llava_model_path=llava_model_path,
            llava_model_base=kwargs.get("llava_model_base"),
            llava_conv_mode=kwargs.get("llava_conv_mode", "vicuna_v1"),
            image_root=kwargs.get("image_root"),
            qwen_max_new_tokens=int(kwargs.get("qwen_max_new_tokens", 256)),
            llava_max_new_tokens=int(kwargs.get("llava_max_new_tokens", 256)),
            qwen_temperature=float(kwargs.get("qwen_temperature", 0.0)),
            llava_temperature=float(kwargs.get("llava_temperature", 0.0)),
            qwen_runtime=kwargs.get("qwen_runtime"),
            llava_runtime=kwargs.get("llava_runtime"),
        )
    available = ", ".join((HeuristicVerificationBackend.name, QwenLlavaEnsembleBackend.name))
    raise ValueError(f"unknown stage3 backend {name!r}; available: {available}")


__all__ = [
    "VerificationBackend",
    "VerificationError",
    "HeuristicVerificationBackend",
    "QwenLlavaEnsembleBackend",
    "VOTE_COUNT",
    "APPROVALS_REQUIRED",
    "VOTE_POLICY_VERSION",
    "ENSEMBLE_VOTE_POLICY_VERSION",
    "evaluate_votes",
    "get_backend",
]
