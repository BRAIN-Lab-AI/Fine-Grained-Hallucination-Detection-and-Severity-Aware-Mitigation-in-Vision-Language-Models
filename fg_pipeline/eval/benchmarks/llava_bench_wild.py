from __future__ import annotations

import json
from pathlib import Path

from fg_pipeline.eval.judges import judge_llava_bench_rows
from fg_pipeline.eval.model_loader import generate_answers_for_records
from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact
from fg_pipeline.eval.utils import default_dataset_root, dump_json, mkdir, read_jsonl


class LlavaBenchWildBenchmark:
    name = "llava_bench_wild"
    judge_required = True
    requires_model = True

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec:
        dataset_root = default_dataset_root(dataset_root_override, "llava-bench-in-the-wild")
        question_file = str(Path(dataset_root) / "questions.jsonl")
        annotation_file = str(Path(dataset_root) / "context.jsonl")
        image_root = image_root_override or str(Path(dataset_root) / "images")
        return BenchmarkSpec(
            name=self.name,
            enabled=True,
            question_file=question_file,
            annotation_file=annotation_file,
            image_root=image_root,
            dataset_root=dataset_root,
            judge_required=True,
            split="wild",
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
            raise ValueError("LLaVA-Bench requires a model spec")
        if not openai_judge_model:
            raise RuntimeError("LLaVA-Bench requires --openai-judge-model")
        question_path = Path(spec.question_file or "")
        if not question_path.exists():
            raise FileNotFoundError(f"Missing LLaVA-Bench question file: {question_path}")
        raw_questions = list(read_jsonl(question_path))
        if limit is not None:
            raw_questions = raw_questions[:limit]
        questions = []
        for idx, row in enumerate(raw_questions):
            record_id = str(row.get("question_id", row.get("id", idx)))
            image_name = row.get("image")
            image_path = Path(spec.image_root or "") / image_name if image_name else None
            questions.append(
                {
                    "id": record_id,
                    "question": row.get("text") or row.get("question") or "",
                    "image": str(image_path) if image_path else None,
                }
            )
        answers = generate_answers_for_records(model, questions)
        prediction_dir = mkdir(Path(run_root) / "models" / model.model_id / "predictions")
        prediction_path = prediction_dir / f"{self.name}.jsonl"
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in answers:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        contexts: dict[str, str] = {}
        if spec.annotation_file and Path(spec.annotation_file).exists():
            for row in read_jsonl(spec.annotation_file):
                contexts[str(row.get("question_id", row.get("id")))] = row.get("text") or row.get("context") or ""
        reference_answers: dict[str, str] = {}
        reference_path = Path(spec.dataset_root or "") / "answers_gpt4.jsonl"
        if reference_path.exists():
            for row in read_jsonl(reference_path):
                reference_answers[str(row.get("question_id", row.get("id")))] = row.get("text") or row.get("answer") or ""
        judged_rows, summary = judge_llava_bench_rows(
            answers,
            openai_judge_model,
            contexts=contexts,
            references=reference_answers,
        )
        judge_dir = mkdir(Path(run_root) / "models" / model.model_id / "judges")
        judge_path = judge_dir / f"{self.name}.json"
        dump_json(judge_path, {"rows": judged_rows, "summary": summary})
        metric_artifact = MetricArtifact(
            benchmark=self.name,
            model_id=model.model_id,
            metrics={
                "llava_bench_overall": summary["overall"],
                "num_reviewed": summary["num_reviewed"],
                "judge_model": summary["judge_model"],
            },
            comparable_to_paper=bool(reference_answers and contexts and model.temperature == 0.0 and model.num_beams == 1),
            comparison_note=None if (reference_answers and contexts) else "project-owned judge path without vendored review assets",
        )
        metric_dir = mkdir(Path(run_root) / "models" / model.model_id / "metrics")
        dump_json(metric_dir / f"{self.name}.json", metric_artifact.to_dict())
        dump_json(
            metric_dir / f"{self.name}.meta.json",
            {
                "question_file": spec.question_file,
                "annotation_file": spec.annotation_file,
                "image_root": spec.image_root,
                "num_examples": len(answers),
            },
        )
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
            JudgeArtifact(
                benchmark=self.name,
                model_id=model.model_id,
                judge_name="openai_chat_completions",
                judge_version=openai_judge_model,
                path=str(judge_path),
            ),
        )
