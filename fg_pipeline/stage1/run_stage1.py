"""Stage 1 CLI — extract critiques from released detection supervision.

Usage::

    python -m fg_pipeline.stage1.run_stage1 \
        --input fg_pipeline/data/hsa_dpo_detection.jsonl \
        --output output/fghd/stage1/detection_critiques.jsonl \
        --stats-out output/fghd/stage1/stats.json

The CLI writes a normalized Stage 1 JSONL plus a compact stats JSON. It never
calls a model; it parses released supervision via
:class:`ReleasedAnnotationBackend`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.io_utils import ensure_parent_dir, read_jsonl, write_jsonl
from fg_pipeline.paths import (
    DEFAULT_STAGE1_INPUT,
    DEFAULT_STAGE1_OUTPUT,
    DEFAULT_STAGE1_STATS,
)
from fg_pipeline.stage1.backends import get_backend
from fg_pipeline.stage1.parser import ParseError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: extract critiques from released fine-grained supervision.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STAGE1_INPUT,
        help="Path to input detection JSONL (default: mirrored released file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE1_OUTPUT,
        help="Path to output Stage 1 JSONL.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=DEFAULT_STAGE1_STATS,
        help="Path to output Stage 1 stats JSON.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="released_annotations",
        help="Stage 1 backend to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows (smoke runs).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed hallucinated rows instead of recording warnings.",
    )
    return parser


def _iter_records(
    backend,
    rows: Iterable[dict[str, Any]],
    *,
    strict: bool,
    limit: int | None,
) -> Iterable[dict[str, Any]]:
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        result = backend.detect(row, strict=strict)
        yield result.record.to_dict()


def _compute_stats(output_path: Path) -> dict[str, Any]:
    total = 0
    hallucinated = 0
    non_hallucinated = 0
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            if rec.get("is_hallucinated"):
                hallucinated += 1
            else:
                non_hallucinated += 1
            for critique in rec.get("critiques") or []:
                by_type[critique.get("hallucination_type", "unknown")] += 1
                by_severity[critique.get("severity_label", "unknown")] += 1

    return {
        "total_rows": total,
        "hallucinated_rows": hallucinated,
        "non_hallucinated_rows": non_hallucinated,
        "critique_count_by_type": dict(by_type),
        "critique_count_by_severity": dict(by_severity),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Stage 1 input not found: {args.input}", file=sys.stderr)
        return 2

    backend = get_backend(args.backend)

    rows_iter = read_jsonl(args.input)
    ensure_parent_dir(args.output)
    try:
        write_jsonl(
            args.output,
            _iter_records(backend, rows_iter, strict=args.strict, limit=args.limit),
        )
    except ParseError as exc:
        print(f"Stage 1 strict parse failed: {exc}", file=sys.stderr)
        return 3

    stats = _compute_stats(args.output)
    ensure_parent_dir(args.stats_out)
    with args.stats_out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Stage 1 wrote {stats['total_rows']} records "
        f"({stats['hallucinated_rows']} hallucinated, "
        f"{stats['non_hallucinated_rows']} non-hallucinated) -> {args.output}"
    )
    print(f"Stage 1 stats -> {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
