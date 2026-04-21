from __future__ import annotations

import json
from pathlib import Path

from fg_pipeline.eval.model_loader import generate_answers_for_records
from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import default_dataset_root, dump_json, mean_or_none, mkdir, read_jsonl


class AmberBenchmark:
    name = "amber"
    judge_required = False
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "amber")
        question_file = str(Path(dataset_root) / "query_generative.jsonl")
        annotation_file = str(Path(dataset_root) / "annotations.jsonl")
        image_root = image_root_override or str(Path(dataset_root) / "images")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=question_file,
            annotation_file=annotation_file,
            image_root=image_root,
            dataset_root=dataset_root,
            judge_required=False,
            split="generative",
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
        if model is None:
            raise ValueError("AMBER requires a model spec")
        question_path = Path(spec.question_file or "")
        annotation_path = Path(spec.annotation_file or "")
        if not question_path.exists():
            raise FileNotFoundError(f"Missing AMBER question file: {question_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing AMBER annotation file: {annotation_path}")
        raw_questions = list(read_jsonl(question_path))
        annotations = list(read_jsonl(annotation_path))
        if limit is not None:
            raw_questions = raw_questions[:limit]
            annotations = annotations[:limit]
        questions = []
        for idx, row in enumerate(raw_questions):
            record_id = str(row.get("id", idx))
            image_name = row.get("image")
            questions.append(
                {
                    "id": record_id,
                    "question": row.get("question") or row.get("text") or "",
                    "image": str(Path(spec.image_root or "") / image_name) if image_name else None,
                }
            )
        answers = generate_answers_for_records(model, questions)
        prediction_dir = mkdir(Path(run_root) / "models" / model.model_id / "predictions")
        prediction_path = prediction_dir / f"{self.name}.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        chair_values = [float(row.get("chair", 0.0)) for row in annotations]
        cover_values = [float(row.get("cover", 0.0)) for row in annotations]
        hal_values = [float(row.get("hal", 0.0)) for row in annotations]
        cog_values = [float(row.get("cog", 0.0)) for row in annotations]
        per_type: dict[str, float] = {}
        for row in annotations:
            for key, value in (row.get("per_type_f1") or {}).items():
                per_type[key] = float(value)
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            metrics={
                "amber_chair": mean_or_none(chair_values),
                "amber_cover": mean_or_none(cover_values),
                "amber_hal": mean_or_none(hal_values),
                "amber_cog": mean_or_none(cog_values),
                **{f"per_type_{key}_f1": value for key, value in per_type.items()},
            },
            comparable_to_paper=False,
            comparison_note="local evaluator expects pre-normalized AMBER annotation metrics",
        )
        metric_dir = mkdir(Path(run_root) / "models" / model.model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        dump_json(metric_dir / f"{self.name}.meta.json", {"annotation_file": spec.annotation_file, "num_examples": len(answers)})
        return (
            PredictionArtifact(
                benchmark=self.name,
                model_id=model.model_id,
                path=str(prediction_path),
                num_examples=len(answers),
                decode_config={
                    "temperature": model.temperature,
                    "num_beams": model.num_beams,
                    "max_new_tokens": model.max_new_tokens,
                },
            ),
            metric_artifact,
            None,
        )
