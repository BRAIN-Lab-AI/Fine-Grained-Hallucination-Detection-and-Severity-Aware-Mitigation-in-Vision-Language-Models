from __future__ import annotations

from typing import Any, Protocol

from fg_pipeline.eval.schemas import BenchmarkSpec, JudgeArtifact, MetricArtifact, ModelSpec, PredictionArtifact


class BenchmarkAdapter(Protocol):
    name: str
    judge_required: bool
    requires_model: bool

    def build_spec(
        self,
        *,
        dataset_root_override: str | None = None,
        image_root_override: str | None = None,
    ) -> BenchmarkSpec: ...

    def evaluate(
        self,
        model: ModelSpec | None,
        spec: BenchmarkSpec,
        *,
        run_root: str,
        limit: int | None = None,
        openai_judge_model: str | None = None,
    ) -> tuple[PredictionArtifact | None, MetricArtifact, JudgeArtifact | None]: ...


from fg_pipeline.eval.benchmarks.amber import AmberBenchmark
from fg_pipeline.eval.benchmarks.llava_bench_wild import LlavaBenchWildBenchmark
from fg_pipeline.eval.benchmarks.mfhallubench import MFHaluBenchBenchmark
from fg_pipeline.eval.benchmarks.mhalubench import MHaluBenchBenchmark
from fg_pipeline.eval.benchmarks.mmhal_bench import MMHalBenchmark
from fg_pipeline.eval.benchmarks.object_halbench import ObjectHalBenchBenchmark
from fg_pipeline.eval.benchmarks.pope import PopeBenchmark

BENCHMARK_REGISTRY: dict[str, BenchmarkAdapter] = {
    "amber": AmberBenchmark(),
    "llava_bench_wild": LlavaBenchWildBenchmark(),
    "mfhallubench": MFHaluBenchBenchmark(),
    "mhalubench": MHaluBenchBenchmark(),
    "mmhal_bench": MMHalBenchmark(),
    "object_halbench": ObjectHalBenchBenchmark(),
    "pope_adv": PopeBenchmark(),
}

__all__ = ["BENCHMARK_REGISTRY", "BenchmarkAdapter"]
