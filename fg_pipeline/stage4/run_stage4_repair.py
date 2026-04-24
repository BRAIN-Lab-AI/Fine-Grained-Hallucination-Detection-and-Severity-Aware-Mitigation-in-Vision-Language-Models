"""Stage 4 CLI — repair Stage 3 rejected rewrites with LLaVA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Protocol, runtime_checkable

from fg_pipeline.io_utils import ensure_parent_dir, maybe_tqdm, read_jsonl, write_jsonl
from fg_pipeline.paths import (
    DEFAULT_STAGE3_OUTPUT,
    DEFAULT_STAGE3_PREFERENCES,
    DEFAULT_STAGE4_FINAL_PREFERENCES,
    DEFAULT_STAGE4_REPAIR_OUTPUT,
    DEFAULT_STAGE4_REPAIR_PREFERENCES,
    DEFAULT_STAGE4_REPAIR_STATS,
)
from fg_pipeline.schemas import PreferenceCleanRecord
from fg_pipeline.stage2.backends import LlavaRewriteBackend, TemplateRewriteBackend
from fg_pipeline.stage4.prompts import PROMPT_VERSION, build_repair_prompt
from fg_pipeline.stage4.schemas import Stage4RepairRecord


@runtime_checkable
class RepairBackend(Protocol):
    name: str

    def repair(self, record: dict[str, Any], *, strict: bool = False) -> str:
        """Return one repaired response for a rejected Stage 3 audit row."""
        ...


class TemplateRepairBackend:
    """Deterministic smoke backend for Stage 4 repair tests."""

    name = "template"

    def __init__(self) -> None:
        self._stage2_backend = TemplateRewriteBackend()

    def repair(self, record: dict[str, Any], *, strict: bool = False) -> str:
        adapted = {
            "response_text": record.get("original_response", ""),
            "critiques": record.get("critiques") or [],
        }
        return self._stage2_backend.rewrite(adapted, strict=strict)


class LlavaRepairBackend:
    """Stage 4 repair backend using the Stage 2 LLaVA runtime with a repair prompt."""

    name = "llava"

    def __init__(
        self,
        *,
        model_path: str,
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        image_root: str | None = None,
    ) -> None:
        self._runtime = LlavaRewriteBackend(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            image_root=image_root,
            prompt_builder=build_repair_prompt,
        )

    def repair(self, record: dict[str, Any], *, strict: bool = False) -> str:
        return self._runtime.rewrite(record, strict=strict)


class Stage4RepairError(ValueError):
    """Raised when Stage 4 repair cannot proceed."""


class _Stats:
    def __init__(self, backend_name: str, *, input_path: str, stage3_preferences_path: str) -> None:
        self.backend = backend_name
        self.input_path = input_path
        self.stage3_preferences_path = stage3_preferences_path
        self.stage3_approved_pairs = 0
        self.stage3_audit_rows = 0
        self.stage3_rejected_rows = 0
        self.repair_rows_processed = 0
        self.repair_pairs_emitted = 0
        self.repair_rows_skipped = 0
        self.final_preference_pairs = 0
        self.prompt_version = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "input_path": self.input_path,
            "stage3_preferences_path": self.stage3_preferences_path,
            "stage3_approved_pairs": self.stage3_approved_pairs,
            "stage3_audit_rows": self.stage3_audit_rows,
            "stage3_rejected_rows": self.stage3_rejected_rows,
            "repair_rows_processed": self.repair_rows_processed,
            "repair_pairs_emitted": self.repair_pairs_emitted,
            "repair_rows_skipped": self.repair_rows_skipped,
            "final_preference_pairs": self.final_preference_pairs,
            "prompt_version": self.prompt_version,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4: repair Stage 3 rejected rewrites and build final preferences.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STAGE3_OUTPUT, help="Stage 3 audit JSONL.")
    parser.add_argument(
        "--stage3-preferences",
        type=Path,
        default=DEFAULT_STAGE3_PREFERENCES,
        help="Stage 3 approved preference JSONL.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE4_REPAIR_OUTPUT)
    parser.add_argument("--repair-preferences-out", type=Path, default=DEFAULT_STAGE4_REPAIR_PREFERENCES)
    parser.add_argument("--final-preferences-out", type=Path, default=DEFAULT_STAGE4_FINAL_PREFERENCES)
    parser.add_argument("--stats-out", type=Path, default=DEFAULT_STAGE4_REPAIR_STATS)
    parser.add_argument("--backend", type=str, default="template", help="Repair backend: template|llava.")
    parser.add_argument("--model-path", type=str, default=None, help="Local LLaVA model path for --backend llava.")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path if model path is a LoRA adapter.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-root", type=str, default=".")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _aggregate_severity(critiques: list[dict[str, Any]]) -> float:
    scores = []
    for critique in critiques:
        score = critique.get("severity_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return 1.0
    return float(mean(scores))


def _get_backend(name: str, **kwargs: Any) -> RepairBackend:
    key = (name or "").strip().lower()
    if key == TemplateRepairBackend.name:
        return TemplateRepairBackend()
    if key == LlavaRepairBackend.name:
        model_path = kwargs.get("model_path") or ""
        if not model_path:
            raise ValueError("llava repair backend requires --model-path")
        return LlavaRepairBackend(
            model_path=model_path,
            model_base=kwargs.get("model_base"),
            conv_mode=kwargs.get("conv_mode", "vicuna_v1"),
            max_new_tokens=int(kwargs.get("max_new_tokens", 256)),
            temperature=float(kwargs.get("temperature", 0.0)),
            image_root=kwargs.get("image_root"),
        )
    raise ValueError("unknown Stage 4 repair backend {!r}; available: template, llava".format(name))


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _write_stats(path: Path, stats: _Stats) -> None:
    stats_path = ensure_parent_dir(path)
    with stats_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def _reset_file(path: Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8", newline="\n"):
        pass


def _preference_from_repair(row: dict[str, Any], repair_text: str, *, backend_name: str) -> dict[str, Any]:
    critiques = list(row.get("critiques") or [])
    severity_score = row.get("response_severity_score")
    if not isinstance(severity_score, (int, float)):
        severity_score = _aggregate_severity(critiques)
    return PreferenceCleanRecord(
        id=row.get("id"),
        question=row.get("question") or row.get("original_response", ""),
        chosen=repair_text,
        rejected=row.get("original_response", ""),
        chosen_score=1.0,
        rejected_score=float(severity_score),
        image=row.get("image"),
        metadata={
            "source_stage": "stage4_repair_preference",
            "repair_backend": backend_name,
            "prompt_version": PROMPT_VERSION,
            "response_severity_score": float(severity_score),
            "stage3_backend": (row.get("metadata") or {}).get("backend"),
            "stage3_vote_policy_version": (row.get("metadata") or {}).get("vote_policy_version"),
            "stage3_approvals": row.get("approvals"),
            "stage3_rejections": row.get("rejections"),
        },
    ).to_dict()


def _repair_record(row: dict[str, Any], repair_text: str, *, backend_name: str) -> dict[str, Any]:
    critiques = list(row.get("critiques") or [])
    severity_score = row.get("response_severity_score")
    if not isinstance(severity_score, (int, float)):
        severity_score = _aggregate_severity(critiques)
    return Stage4RepairRecord(
        id=row.get("id"),
        image=row.get("image"),
        question=row.get("question", ""),
        original_response=row.get("original_response", ""),
        failed_rewrite_response=row.get("rewrite_response", ""),
        repair_response=repair_text,
        critiques=critiques,
        stage3_votes=list(row.get("votes") or []),
        response_severity_score=float(severity_score),
        chosen=repair_text,
        rejected=row.get("original_response", ""),
        metadata={
            "source_stage": "stage4_repair",
            "repair_backend": backend_name,
            "prompt_version": PROMPT_VERSION,
            "stage3_metadata": row.get("metadata") or {},
        },
    ).to_dict()


def _iter_rejected_rows(rows: Iterable[dict[str, Any]], *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    emitted = 0
    for row in rows:
        if row.get("passed_majority"):
            continue
        yield row
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def _combine_final_preferences(
    *,
    stage3_preferences: list[dict[str, Any]],
    repair_preferences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    seen_ids: set[int | str] = set()
    for row in stage3_preferences + repair_preferences:
        row_id = row.get("id")
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        final_rows.append(row)
    return final_rows


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 4 input not found: {args.input}", file=sys.stderr)
        print("Run Stage 3 first:  bash scripts/run_stage3_validate.sh", file=sys.stderr)
        return 2
    if not args.stage3_preferences.exists():
        print(
            f"Stage 4 Stage 3 preferences not found; treating as zero approved rows: {args.stage3_preferences}",
            file=sys.stderr,
        )

    try:
        backend = _get_backend(
            args.backend,
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            image_root=args.image_root,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Stage 4 backend error: {exc}", file=sys.stderr)
        return 2

    stats = _Stats(
        backend.name,
        input_path=str(args.input.resolve()),
        stage3_preferences_path=str(args.stage3_preferences.resolve()),
    )
    stage3_preferences = _load_jsonl_if_exists(args.stage3_preferences)
    stats.stage3_approved_pairs = len(stage3_preferences)
    stage3_audit_rows = _load_jsonl_if_exists(args.input)
    stats.stage3_audit_rows = len(stage3_audit_rows)
    rejected_rows = list(_iter_rejected_rows(stage3_audit_rows, limit=args.limit))
    stats.stage3_rejected_rows = len(rejected_rows)

    if args.resume:
        existing_repair_rows = _load_jsonl_if_exists(args.output)
        existing_repair_prefs = _load_jsonl_if_exists(args.repair_preferences_out)
        processed_ids = {row.get("id") for row in existing_repair_rows}
        repair_preferences = existing_repair_prefs
        stats.repair_rows_processed = len(existing_repair_rows)
        stats.repair_pairs_emitted = len(existing_repair_prefs)
    else:
        processed_ids = set()
        repair_preferences = []
        _reset_file(args.output)
        _reset_file(args.repair_preferences_out)

    approved_ids = {row.get("id") for row in stage3_preferences}
    try:
        for row in maybe_tqdm(rejected_rows, desc="Stage 4 repair", total=len(rejected_rows)):
            row_id = row.get("id")
            if row_id in processed_ids or row_id in approved_ids:
                continue
            repair_text = backend.repair(row, strict=args.strict).strip()
            if not repair_text:
                if args.strict:
                    raise Stage4RepairError(f"row id={row_id!r} backend returned empty repair")
                stats.repair_rows_skipped += 1
                continue
            repair_row = _repair_record(row, repair_text, backend_name=backend.name)
            pref_row = _preference_from_repair(row, repair_text, backend_name=backend.name)
            _append_jsonl_row(args.output, repair_row)
            _append_jsonl_row(args.repair_preferences_out, pref_row)
            repair_preferences.append(pref_row)
            processed_ids.add(row_id)
            stats.repair_rows_processed += 1
            stats.repair_pairs_emitted += 1
    except Stage4RepairError as exc:
        print(f"Stage 4 repair failed: {exc}", file=sys.stderr)
        return 3

    final_preferences = _combine_final_preferences(
        stage3_preferences=stage3_preferences,
        repair_preferences=repair_preferences,
    )
    stats.final_preference_pairs = len(final_preferences)
    write_jsonl(args.final_preferences_out, final_preferences)
    _write_stats(args.stats_out, stats)

    print(
        f"Stage 4 wrote {stats.repair_pairs_emitted} repair pair(s) "
        f"and {stats.final_preference_pairs} final preference pair(s)"
    )
    print(f"Stage 4 repairs -> {args.output}")
    print(f"Stage 4 final preferences -> {args.final_preferences_out}")
    print(f"Stage 4 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
