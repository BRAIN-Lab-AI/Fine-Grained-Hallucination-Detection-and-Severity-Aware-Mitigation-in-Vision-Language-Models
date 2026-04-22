from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fg_pipeline.eval.benchmarks import BENCHMARK_REGISTRY
from fg_pipeline.eval.judges import judge_hss_rows
from fg_pipeline.eval.reporting import build_general_report, build_paper_comparison, write_comparison_bundle
from fg_pipeline.eval.schemas import BenchmarkSpec, MetricArtifact, ModelSpec
from fg_pipeline.eval.utils import (
    discover_stage_paths,
    dump_json,
    load_model_specs,
    mkdir,
    summarize_stage3,
    summarize_stage4,
)


_PAPER_CORE_DEFAULT = "mhalubench,mfhallubench,object_halbench,amber,mmhal_bench,pope_adv,llava_bench_wild,hss"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project-owned paper/general evaluation.")
    parser.add_argument("--run-name", required=True, help="Stable output run name under output/eval.")
    parser.add_argument("--models-json", required=True, help="Path to a JSON manifest of ModelSpec rows.")
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark names. Defaults to the paper-core set when --paper-core is used.",
    )
    parser.add_argument("--paper-core", action="store_true", help="Run the paper-core comparison suite.")
    parser.add_argument("--general", action="store_true", help="Run stage-internal general evaluation.")
    parser.add_argument("--output-root", default="output/eval", help="Evaluation output root.")
    parser.add_argument("--openai-judge-model", default=None, help="Judge model for GPT-style evaluations.")
    parser.add_argument("--image-root-override", default=None, help="Override benchmark image roots.")
    parser.add_argument("--dataset-root-override", default=None, help="Override benchmark dataset roots.")
    parser.add_argument("--skip-missing-datasets", action="store_true", help="Skip missing benchmark assets instead of failing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def _selected_benchmarks(args: argparse.Namespace) -> list[str]:
    raw = [item.strip() for item in (args.benchmarks or "").split(",") if item.strip()]
    if args.paper_core and not raw:
        raw = _PAPER_CORE_DEFAULT.split(",")
    return raw


def _load_models(path: str) -> list[ModelSpec]:
    return [ModelSpec.from_dict(item) for item in load_model_specs(path)]


def _benchmark_spec(name: str, args: argparse.Namespace) -> BenchmarkSpec:
    if name == "hss":
        return BenchmarkSpec(name="hss", enabled=True, judge_required=True, split="default")
    try:
        adapter = BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted({*BENCHMARK_REGISTRY.keys(), "hss"}))
        raise SystemExit(f"Unknown benchmark {name!r}; available: {available}") from exc
    return adapter.build_spec(
        dataset_root_override=args.dataset_root_override,
        image_root_override=args.image_root_override,
    )


def _stage_metrics() -> dict[str, dict]:
    discovered = discover_stage_paths()
    stage_metrics: dict[str, dict] = {}
    if "stage3_dir" in discovered:
        stage_metrics["stage3"] = summarize_stage3(discovered["stage3_dir"])
    if "stage4_dir" in discovered:
        stage_metrics["stage4"] = summarize_stage4(discovered["stage4_dir"])
    return stage_metrics


def _run_hss(
    models: list[ModelSpec],
    output_root: Path,
    judge_model: str | None,
) -> list[MetricArtifact]:
    if not judge_model:
        raise RuntimeError("HSS requires --openai-judge-model")
    discovered = discover_stage_paths()
    pref_path = discovered.get("preference_pairs")
    if not pref_path:
        raise FileNotFoundError(
            "HSS requires a preference-pairs JSONL (e.g. hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl)"
        )
    from fg_pipeline.io_utils import read_jsonl

    pref_rows = list(read_jsonl(pref_path))
    target_model = models[min(1, len(models) - 1)]
    rows = [
        {
            "id": row.get("id"),
            "question": row.get("question"),
            "text": row.get("chosen"),
        }
        for row in pref_rows
    ]
    judged_rows, summary = judge_hss_rows(rows, judge_model)
    judge_dir = mkdir(output_root / "models" / target_model.model_id / "judges")
    dump_json(judge_dir / "hss.json", {"rows": judged_rows, "summary": summary})
    artifact = MetricArtifact(
        benchmark="hss",
        model_id=target_model.model_id,
        metrics=summary,
        comparable_to_paper=False,
        comparison_note="same rubric, local sample/eval set may differ from paper",
    )
    metric_dir = mkdir(output_root / "models" / target_model.model_id / "metrics")
    dump_json(metric_dir / "hss.json", artifact.to_dict())
    return [artifact]


def main() -> int:
    args = parse_args()
    models = _load_models(args.models_json)
    selected = _selected_benchmarks(args)
    output_root = mkdir(Path(args.output_root) / args.run_name)

    if not args.paper_core and not args.general:
        args.paper_core = True

    metric_artifacts: list[MetricArtifact] = []
    availability: dict[str, dict] = {}

    if args.paper_core:
        for benchmark_name in selected:
            spec = _benchmark_spec(benchmark_name, args)
            if benchmark_name == "hss":
                try:
                    hss_artifacts = _run_hss(models, output_root, args.openai_judge_model)
                    metric_artifacts.extend(hss_artifacts)
                    availability[benchmark_name] = {"status": "ok", "note": "judge-based"}
                except Exception as exc:
                    if args.skip_missing_datasets:
                        availability[benchmark_name] = {"status": "skipped", "note": str(exc)}
                        continue
                    raise
                continue

            adapter = BENCHMARK_REGISTRY[benchmark_name]
            if spec.judge_required and not args.openai_judge_model:
                raise SystemExit(
                    f"Benchmark {benchmark_name!r} requires --openai-judge-model or OPENAI_JUDGE_MODEL"
                )
            try:
                if adapter.requires_model:
                    for model in models:
                        _, metric_artifact, _ = adapter.evaluate(
                            model,
                            spec,
                            run_root=str(output_root),
                            limit=args.limit,
                            openai_judge_model=args.openai_judge_model,
                        )
                        metric_artifacts.append(metric_artifact)
                else:
                    _, metric_artifact, _ = adapter.evaluate(
                        None,
                        spec,
                        run_root=str(output_root),
                        limit=args.limit,
                        openai_judge_model=args.openai_judge_model,
                    )
                    metric_artifacts.append(metric_artifact)
                availability[benchmark_name] = {"status": "ok", "note": None}
            except Exception as exc:
                if args.skip_missing_datasets:
                    availability[benchmark_name] = {"status": "skipped", "note": str(exc)}
                    continue
                raise

    stage_metrics = _stage_metrics() if args.general else {}
    general_report = build_general_report(stage_metrics, metric_artifacts)
    paper_rows = build_paper_comparison(metric_artifacts, models) if args.paper_core else []
    write_comparison_bundle(
        output_root,
        models=models,
        availability=availability,
        paper_rows=paper_rows,
        general_report=general_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
