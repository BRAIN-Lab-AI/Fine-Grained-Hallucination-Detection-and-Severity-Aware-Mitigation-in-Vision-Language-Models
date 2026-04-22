from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.eval.reference_tables import PAPER_METRIC_DIRECTIONS, paper_base_value, paper_reference_value
from fg_pipeline.eval.schemas import ComparisonRow, MetricArtifact, ModelSpec
from fg_pipeline.eval.utils import dump_json, mkdir, quantize_float, safe_div


def _delta(direction: str, base_value: float | None, our_value: float | None) -> tuple[float | None, float | None]:
    if base_value is None or our_value is None:
        return None, None
    absolute = our_value - base_value
    denominator = abs(base_value)
    relative = None if denominator == 0 else (absolute / denominator)
    return absolute, relative


def build_paper_comparison(
    metric_artifacts: Iterable[MetricArtifact],
    models: list[ModelSpec],
) -> list[ComparisonRow]:
    artifacts = list(metric_artifacts)
    if not models:
        return []
    base_model_id = models[0].model_id
    our_model_id = models[min(1, len(models) - 1)].model_id
    by_key: dict[tuple[str, str], dict[str, MetricArtifact]] = {}
    for artifact in artifacts:
        key = (artifact.benchmark, artifact.model_id)
        by_key[key] = {artifact.benchmark: artifact}

    benchmark_to_artifacts: dict[str, dict[str, MetricArtifact]] = {}
    for artifact in artifacts:
        benchmark_to_artifacts.setdefault(artifact.benchmark, {})[artifact.model_id] = artifact

    rows: list[ComparisonRow] = []
    for benchmark, per_model in sorted(benchmark_to_artifacts.items()):
        metric_names = sorted({key for artifact in per_model.values() for key in artifact.metrics.keys()})
        for metric in metric_names:
            direction = PAPER_METRIC_DIRECTIONS.get(metric)
            if direction is None:
                continue
            base_artifact = per_model.get(base_model_id)
            our_artifact = per_model.get(our_model_id)
            base_value = None if base_artifact is None else _coerce_numeric(base_artifact.metrics.get(metric))
            our_value = None if our_artifact is None else _coerce_numeric(our_artifact.metrics.get(metric))
            paper_row_name, paper_value = paper_reference_value(benchmark, metric)
            if base_value is None:
                _, paper_base = paper_base_value(benchmark, metric)
                if paper_base is not None:
                    base_value = paper_base
            absolute, relative = _delta(direction, base_value, our_value)
            note_parts = []
            if our_artifact and not our_artifact.comparable_to_paper:
                note_parts.append(our_artifact.comparison_note or "not paper-comparable")
            rows.append(
                ComparisonRow(
                    benchmark=benchmark,
                    metric=metric,
                    direction=direction,  # type: ignore[arg-type]
                    base_value=quantize_float(base_value),
                    our_value=quantize_float(our_value),
                    paper_reference_value=quantize_float(paper_value),
                    delta_vs_base=quantize_float(absolute),
                    relative_delta_vs_base=quantize_float(relative),
                    paper_row_name=paper_row_name,
                    note="; ".join(part for part in note_parts if part) or None,
                )
            )
    return rows


def _coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_general_report(stage_metrics: dict[str, Any], benchmark_metrics: Iterable[MetricArtifact]) -> dict[str, Any]:
    return {
        "stage_metrics": stage_metrics,
        "benchmarks": [artifact.to_dict() for artifact in benchmark_metrics],
    }


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def render_paper_markdown(models: list[ModelSpec], rows: list[ComparisonRow], availability: dict[str, Any]) -> str:
    model_lines = "\n".join(f"- `{model.model_id}` -> `{model.model_path}`" for model in models)
    benchmark_rows = [
        [
            benchmark,
            info.get("status"),
            info.get("note"),
        ]
        for benchmark, info in sorted(availability.items())
    ]
    comparison_rows = [
        [
            row.benchmark,
            row.metric,
            row.base_value,
            row.our_value,
            row.paper_reference_value,
            row.delta_vs_base,
            row.relative_delta_vs_base,
            row.paper_row_name,
            row.note,
        ]
        for row in rows
    ]
    sections = [
        "# Paper Core Evaluation",
        "## Evaluated Models",
        model_lines or "- none",
        "## Benchmark Availability",
        _markdown_table(["benchmark", "status", "note"], benchmark_rows or [["none", "none", ""]]),
        "## Paper-Faithful Comparison",
        _markdown_table(
            [
                "benchmark",
                "metric",
                "base reproduced",
                "our reproduced",
                "paper reference",
                "delta vs base",
                "relative delta vs base",
                "paper row",
                "note",
            ],
            comparison_rows or [["none", "none", "", "", "", "", "", "", ""]],
        ),
        "## Judge Usage",
        "- Judge-based benchmarks require `OPENAI_API_KEY` and `--openai-judge-model`.",
    ]
    return "\n\n".join(sections)


def render_general_markdown(report: dict[str, Any]) -> str:
    sections = ["# General Evaluation"]
    stage_metrics = report.get("stage_metrics", {})
    for stage_name in ("stage3", "stage4"):
        sections.append(f"## {stage_name.upper()} Summary")
        payload = stage_metrics.get(stage_name) or {}
        if not payload:
            sections.append("- unavailable")
            continue
        rows = [[key, value] for key, value in sorted(payload.items())]
        sections.append(_markdown_table(["metric", "value"], rows))

    sections.append("## Benchmark Summary")
    benchmark_rows: list[list[Any]] = []
    for artifact in report.get("benchmarks", []):
        benchmark_rows.append(
            [
                artifact["benchmark"],
                artifact["model_id"],
                artifact["comparable_to_paper"],
                artifact["comparison_note"],
            ]
        )
    sections.append(
        _markdown_table(
            ["benchmark", "model", "comparable_to_paper", "note"],
            benchmark_rows or [["none", "none", "", ""]],
        )
    )
    sections.append("## Runtime And Judge Usage Summary")
    sections.append("- General evaluation includes stage-internal metrics and may not be paper-comparable.")
    return "\n\n".join(sections)


def write_comparison_bundle(
    output_root: str | Path,
    *,
    models: list[ModelSpec],
    availability: dict[str, Any],
    paper_rows: list[ComparisonRow],
    general_report: dict[str, Any],
) -> None:
    comparison_dir = mkdir(Path(output_root) / "comparison")
    paper_json = {
        "models": [model.to_dict() for model in models],
        "availability": availability,
        "rows": [row.to_dict() for row in paper_rows],
    }
    dump_json(comparison_dir / "paper_core.json", paper_json)
    (comparison_dir / "paper_core.md").write_text(
        render_paper_markdown(models, paper_rows, availability),
        encoding="utf-8",
    )

    dump_json(comparison_dir / "general_eval.json", general_report)
    (comparison_dir / "general_eval.md").write_text(
        render_general_markdown(general_report),
        encoding="utf-8",
    )

    summary_csv = comparison_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "benchmark",
                "metric",
                "direction",
                "base_value",
                "our_value",
                "paper_reference_value",
                "delta_vs_base",
                "relative_delta_vs_base",
                "paper_row_name",
                "note",
            ]
        )
        for row in paper_rows:
            writer.writerow(
                [
                    row.benchmark,
                    row.metric,
                    row.direction,
                    row.base_value,
                    row.our_value,
                    row.paper_reference_value,
                    row.delta_vs_base,
                    row.relative_delta_vs_base,
                    row.paper_row_name,
                    row.note,
                ]
            )
