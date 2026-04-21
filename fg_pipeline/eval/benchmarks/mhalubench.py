from __future__ import annotations

from pathlib import Path

from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import binary_classification_metrics, default_dataset_root, dump_json, macro_f1_from_confusion, mkdir, read_jsonl


class MHaluBenchBenchmark:
    name = "mhalubench"
    judge_required = False
    requires_model = False

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "mhalubench")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=str(Path(dataset_root) / "predictions.jsonl"),
            annotation_file=str(Path(dataset_root) / "annotations.jsonl"),
            image_root=image_root_override,
            dataset_root=dataset_root,
            judge_required=False,
            split="image_to_text",
        )

    def evaluate(
        self,
        model: ModelSpec | None,
        spec: BenchmarkSpec,
        *,
        run_root: str,
        limit: int | None = None,
        openai_judge_model: str | None = None,
    ) -> tuple[PredictionArtifact | None, MetricArtifact, JudgeArtifact | None]:
        prediction_path = Path(spec.question_file or "")
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing MHaluBench predictions file: {prediction_path}")
        rows = list(read_jsonl(prediction_path))
        if limit is not None:
            rows = rows[:limit]

        def _collect(level: str) -> dict[str, float | None]:
            labels = [int(row[f"{level}_label"]) for row in rows if f"{level}_label" in row and f"{level}_prediction" in row]
            predictions = [int(row[f"{level}_prediction"]) for row in rows if f"{level}_label" in row and f"{level}_prediction" in row]
            return binary_classification_metrics(labels, predictions)

        claim = _collect("claim")
        segment = _collect("segment")
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id="stage3_detector",
            metrics={
                "claim_accuracy": claim["accuracy"],
                "claim_precision": claim["precision"],
                "claim_recall": claim["recall"],
                "claim_macro_f1": claim["f1"],
                "segment_accuracy": segment["accuracy"],
                "segment_precision": segment["precision"],
                "segment_recall": segment["recall"],
                "segment_macro_f1": segment["f1"],
            },
            comparable_to_paper=True,
            comparison_note=None,
        )
        metric_dir = mkdir(Path(run_root) / "models" / "stage3_detector" / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        return None, metric_artifact, None
