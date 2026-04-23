from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fg_pipeline.io_utils import read_jsonl
from fg_pipeline.paths import DEFAULT_STAGE1_INPUT
from fg_pipeline.stage1.detector_data import build_llava_detector_example


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare released Stage 1 supervision as a LLaVA detector SFT dataset.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_STAGE1_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/fghd/stage1/detector_train.json"),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"Stage 1 detector dataset input not found: {args.input}", file=sys.stderr)
        return 2
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    payload = [build_llava_detector_example(row) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Stage 1 detector train set wrote {len(payload)} example(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
