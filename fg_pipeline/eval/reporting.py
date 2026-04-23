from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from fg_pipeline.eval.reference_tables import PAPER_METRIC_DIRECTIONS, paper_base_value, paper_reference_value
from fg_pipeline.eval.schemas import ComparisonRow, MetricArtifact, ModelSpec
from fg_pipeline.eval.utils import dump_json, mkdir, quantize_float


def _delta(baseline_value: float | None, our_value: float | None) -> tuple[float | None, float | None]:
    if baseline_value is None or our_value is None:
        return None, None
    absolute = our_value - baseline_value
    denominator = abs(baseline_value)
    relative = None if denominator == 0 else (absolute / denominator)
    return absolute, relative


def _coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_paper_comparison(
    metric_artifacts: Iterable[MetricArtifact],
    models: list[ModelSpec],
) -> list[ComparisonRow]:
    artifacts = list(metric_artifacts)
    if not artifacts:
        return []

    base_model_id = models[0].model_id if models else None
    our_model_id = models[min(1, len(models) - 1)].model_id if models else None

    benchmark_to_artifacts: dict[str, dict[str, MetricArtifact]] = {}
    for artifact in artifacts:
        benchmark_to_artifacts.setdefault(artifact.benchmark, {})[artifact.model_id] = artifact

    rows: list[ComparisonRow] = []
    for benchmark, per_model in sorted(benchmark_to_artifacts.items()):
        metric_names = sorted({key for artifact in per_model.values() for key in artifact.metrics.keys()})
        our_artifact = per_model.get(our_model_id) if our_model_id else None
        if our_artifact is None and len(per_model) == 1:
            our_artifact = next(iter(per_model.values()))
        baseline_artifact = per_model.get(base_model_id) if base_model_id else None

        for metric in metric_names:
            direction = PAPER_METRIC_DIRECTIONS.get(metric)
            if direction is None:
                continue

            baseline_value = None if baseline_artifact is None else _coerce_numeric(baseline_artifact.metrics.get(metric))
            baseline_row_name = base_model_id if baseline_artifact is not None else None
            if baseline_value is None:
                baseline_row_name, baseline_value = paper_base_value(benchmark, metric)

            our_value = None if our_artifact is None else _coerce_numeric(our_artifact.metrics.get(metric))
            paper_row_name, paper_value = paper_reference_value(benchmark, metric)
            delta, relative = _delta(baseline_value, our_value)

            strictly_comparable = bool(
                our_artifact is not None
                and our_artifact.comparable_to_paper
                and paper_value is not None
            )
            note = None
            if our_artifact is not None and not our_artifact.comparable_to_paper:
                note = our_artifact.comparison_note or "supplemental only"
            elif paper_value is None:
                note = "no paper reference row"

            rows.append(
                ComparisonRow(
                    benchmark=benchmark,
                    metric=metric,
                    direction=direction,  # type: ignore[arg-type]
                    baseline_value=quantize_float(baseline_value),
                    our_value=quantize_float(our_value),
                    paper_reference_value=quantize_float(paper_value),
                    delta_vs_baseline=quantize_float(delta),
                    relative_delta_vs_baseline=quantize_float(relative),
                    strictly_comparable=strictly_comparable,
                    paper_row_name=paper_row_name,
                    baseline_row_name=baseline_row_name,
                    comparison_note=note,
                )
            )
    return rows


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
        [benchmark, info.get("status"), info.get("note")]
        for benchmark, info in sorted(availability.items())
    ]
    comparison_rows = [
        [
            row.benchmark,
            row.metric,
            row.baseline_value,
            row.our_value,
            row.paper_reference_value,
            row.delta_vs_baseline,
            row.relative_delta_vs_baseline,
            row.baseline_row_name,
            row.paper_row_name,
            row.comparison_note,
        ]
        for row in rows
    ]
    sections = [
        "# Strict Paper Comparison",
        "## Evaluated Models",
        model_lines or "- none",
        "## Benchmark Availability",
        _markdown_table(["benchmark", "status", "note"], benchmark_rows or [["none", "none", ""]]),
        "## Strictly Comparable Rows",
        _markdown_table(
            [
                "benchmark",
                "metric",
                "baseline reproduced",
                "our reproduced",
                "paper reference",
                "delta vs baseline",
                "relative delta vs baseline",
                "baseline row",
                "paper row",
                "note",
            ],
            comparison_rows or [["none", "none", "", "", "", "", "", "", "", ""]],
        ),
        "## Fairness Contract",
        "- This table includes only rows marked strictly comparable to the referenced paper.",
        "- Supplemental or proxy metrics are reported separately and do not appear in this delta table.",
    ]
    return "\n\n".join(sections)


def render_supplemental_markdown(rows: list[ComparisonRow], availability: dict[str, Any]) -> str:
    benchmark_rows = [
        [benchmark, info.get("status"), info.get("note")]
        for benchmark, info in sorted(availability.items())
    ]
    comparison_rows = [
        [
            row.benchmark,
            row.metric,
            row.our_value,
            row.paper_reference_value,
            row.strictly_comparable,
            row.comparison_note,
        ]
        for row in rows
    ]
    sections = [
        "# Supplemental Local Evaluation",
        "## Benchmark Availability",
        _markdown_table(["benchmark", "status", "note"], benchmark_rows or [["none", "none", ""]]),
        "## Supplemental Rows",
        _markdown_table(
            ["benchmark", "metric", "our value", "paper reference", "strictly comparable", "note"],
            comparison_rows or [["none", "none", "", "", "", ""]],
        ),
        "## Notes",
        "- These rows are excluded from the strict paper comparison table.",
        "- Reasons include proxy evaluators, unmatched protocols, or missing paper reference rows.",
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
    sections.append("## Runtime Summary")
    sections.append("- General evaluation may include local proxy or supplemental metrics.")
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
    strict_rows = [row for row in paper_rows if row.strictly_comparable]
    supplemental_rows = [row for row in paper_rows if not row.strictly_comparable]

    paper_json = {
        "models": [model.to_dict() for model in models],
        "availability": availability,
        "rows": [row.to_dict() for row in strict_rows],
    }
    dump_json(comparison_dir / "paper_core.json", paper_json)
    (comparison_dir / "paper_core.md").write_text(
        render_paper_markdown(models, strict_rows, availability),
        encoding="utf-8",
    )

    supplemental_json = {
        "models": [model.to_dict() for model in models],
        "availability": availability,
        "rows": [row.to_dict() for row in supplemental_rows],
    }
    dump_json(comparison_dir / "supplemental_eval.json", supplemental_json)
    (comparison_dir / "supplemental_eval.md").write_text(
        render_supplemental_markdown(supplemental_rows, availability),
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
                "baseline_value",
                "our_value",
                "paper_reference_value",
                "delta_vs_baseline",
                "relative_delta_vs_baseline",
                "strictly_comparable",
                "baseline_row_name",
                "paper_row_name",
                "comparison_note",
            ]
        )
        for row in paper_rows:
            writer.writerow(
                [
                    row.benchmark,
                    row.metric,
                    row.direction,
                    row.baseline_value,
                    row.our_value,
                    row.paper_reference_value,
                    row.delta_vs_baseline,
                    row.relative_delta_vs_baseline,
                    row.strictly_comparable,
                    row.baseline_row_name,
                    row.paper_row_name,
                    row.comparison_note,
                ]
            )
