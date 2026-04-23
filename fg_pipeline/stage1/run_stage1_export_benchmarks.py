from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.stage1.detector_data import prediction_for_mfhallubench, prediction_for_mhalubench


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Stage 1 detector predictions into benchmark prediction JSONL files.",
    )
    parser.add_argument("--benchmark", required=True, choices=("mhalubench", "mfhallubench"))
    parser.add_argument("--stage1-input", type=Path, required=True)
    parser.add_argument("--annotation-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.stage1_input.exists():
        print(f"Missing Stage 1 predictions: {args.stage1_input}", file=sys.stderr)
        return 2
    if not args.annotation_input.exists():
        print(f"Missing benchmark annotations: {args.annotation_input}", file=sys.stderr)
        return 2

    stage1_by_id = {str(row.get("id")): row for row in read_jsonl(args.stage1_input)}
    source_rows = list(read_jsonl(args.annotation_input))
    projector = prediction_for_mhalubench if args.benchmark == "mhalubench" else prediction_for_mfhallubench
    output_rows = []
    missing = 0
    for row in source_rows:
        key = str(row.get("id"))
        if key not in stage1_by_id:
            missing += 1
            continue
        output_rows.append(projector(stage1_by_id[key], row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    print(
        f"Stage 1 benchmark export wrote {len(output_rows)} row(s) for {args.benchmark} "
        f"(missing ids: {missing}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
