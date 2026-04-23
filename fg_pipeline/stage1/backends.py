"""Backend seam for Stage 1 critique detection.

The Stage 1 contract is fixed around :class:`Stage1Record`. Two backends are
provided:

* ``released_annotations`` — parser over the released HSA-DPO supervision.
* ``llava_detector`` — local LLaVA-based detector inference backend that emits
  the same normalized Stage 1 contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fg_pipeline.stage1.parser import ParseResult, parse_assessed_annotation, parse_detection_row
from fg_pipeline.stage1.prompts import PROMPT_VERSION, build_detector_prompt, coerce_stage1_inputs
from fg_pipeline.stage1.schemas import Stage1Record


@runtime_checkable
class CritiqueDetectorBackend(Protocol):
    """Minimal Stage-1-facing contract for any critique detection backend."""

    name: str

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        """Return a :class:`ParseResult` for one source row."""
        ...


class ReleasedAnnotationBackend:
    """Default backend: parse released HSA-DPO detection supervision."""

    name: str = "released_annotations"

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        return parse_detection_row(row, strict=strict)


class LlavaDetectorBackend:
    """Local LLaVA detector backend for Stage 1 research runs."""

    name: str = "llava_detector"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        image_root: str | None = None,
        max_new_tokens: int = 384,
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

    def _resolved_image(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = self._image_root / candidate
        return candidate if candidate.exists() else None

    def _generate_annotation(self, row: dict[str, Any]) -> tuple[str, str, str]:
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

        question, response_text = coerce_stage1_inputs(row)
        prompt_text = build_detector_prompt(question=question, response_text=response_text)

        image_path = self._resolved_image(row.get("image"))
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
        annotation_text = tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
        return question, response_text, annotation_text

    def detect(self, row: dict[str, Any], *, strict: bool = False) -> ParseResult:
        question, response_text, annotation_text = self._generate_annotation(row)
        result = parse_assessed_annotation(
            row_id=row.get("id"),
            image=row.get("image"),
            assessed_text=response_text,
            annotation_text=annotation_text,
            source=self.name,
            strict=strict,
        )
        result.record.question = question
        result.record.metadata["prompt_version"] = PROMPT_VERSION
        return result


def get_backend(name: str, **kwargs: Any) -> CritiqueDetectorBackend:
    """Return an instantiated backend by registered name."""

    key = (name or "").strip().lower()
    if key == ReleasedAnnotationBackend.name:
        return ReleasedAnnotationBackend()
    if key == LlavaDetectorBackend.name:
        model_path = kwargs.get("model_path") or ""
        if not model_path:
            raise ValueError("llava_detector requires --model-path / model_path kwarg")
        return LlavaDetectorBackend(
            model_path=model_path,
            model_base=kwargs.get("model_base"),
            conv_mode=kwargs.get("conv_mode", "vicuna_v1"),
            image_root=kwargs.get("image_root"),
            max_new_tokens=int(kwargs.get("max_new_tokens", 384)),
            temperature=float(kwargs.get("temperature", 0.0)),
        )
    available = ", ".join((ReleasedAnnotationBackend.name, LlavaDetectorBackend.name))
    raise ValueError(
        f"unknown stage1 backend {name!r}; available: {available}"
    )


__all__ = [
    "CritiqueDetectorBackend",
    "ReleasedAnnotationBackend",
    "LlavaDetectorBackend",
    "Stage1Record",
    "get_backend",
]
