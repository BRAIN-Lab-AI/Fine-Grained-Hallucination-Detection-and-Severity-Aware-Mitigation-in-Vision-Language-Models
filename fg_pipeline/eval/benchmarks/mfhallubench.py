from __future__ import annotations

from pathlib import Path

from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import binary_classification_metrics, default_dataset_root, dump_json, mkdir, read_jsonl


class MFHaluBenchBenchmark:
    name = "mfhallubench"
    judge_required = False
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "mfhallubench")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=str(Path(dataset_root) / "predictions.jsonl"),
            annotation_file=str(Path(dataset_root) / "annotations.jsonl"),
            image_root=image_root_override,
            dataset_root=dataset_root,
            judge_required=False,
            split="default",
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
            raise FileNotFoundError(f"Missing MFHaluBench predictions file: {prediction_path}")
        rows = list(read_jsonl(prediction_path))
        if limit is not None:
            rows = rows[:limit]
        labels = [int(row["binary_label"]) for row in rows if "binary_label" in row and "binary_prediction" in row]
        predictions = [int(row["binary_prediction"]) for row in rows if "binary_label" in row and "binary_prediction" in row]
        binary = binary_classification_metrics(labels, predictions)
        multi_pairs = [(str(row["multi_label"]), str(row["multi_prediction"])) for row in rows if "multi_label" in row and "multi_prediction" in row]
        multi_acc = None
        if multi_pairs:
            multi_acc = sum(1 for truth, pred in multi_pairs if truth == pred) / len(multi_pairs)
        per_type_confusion: dict[str, dict[str, int]] = {}
        per_severity_confusion: dict[str, dict[str, int]] = {}
        for row in rows:
            if "type_label" in row and "type_prediction" in row:
                per_type_confusion.setdefault(str(row["type_label"]), {})
                bucket = per_type_confusion[str(row["type_label"])]
                bucket[str(row["type_prediction"])] = bucket.get(str(row["type_prediction"]), 0) + 1
            if "severity_label" in row and "severity_prediction" in row:
                per_severity_confusion.setdefault(str(row["severity_label"]), {})
                bucket = per_severity_confusion[str(row["severity_label"])]
                bucket[str(row["severity_prediction"])] = bucket.get(str(row["severity_prediction"]), 0) + 1
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id if model is not None else "stage1_detector",
            metrics={
                "binary_precision": binary["precision"],
                "binary_recall": binary["recall"],
                "binary_accuracy": binary["accuracy"],
                "binary_f1": binary["f1"],
                "multi_accuracy": multi_acc,
                "type_confusion_labels": list(per_type_confusion.keys()) if per_type_confusion else None,
                "severity_confusion_labels": list(per_severity_confusion.keys()) if per_severity_confusion else None,
            },
            comparable_to_paper=True,
            comparison_note=None,
        )
        model_id = metric_artifact.model_id
        metric_dir = mkdir(Path(run_root) / "models" / model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        dump_json(
            metric_dir / f"{self.name}.meta.json",
            {"type_confusion": per_type_confusion, "severity_confusion": per_severity_confusion},
        )
        return None, metric_artifact, None
