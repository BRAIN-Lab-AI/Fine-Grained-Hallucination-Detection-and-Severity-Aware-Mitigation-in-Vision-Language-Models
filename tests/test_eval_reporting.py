from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fg_pipeline.eval.reporting import (
    build_paper_comparison,
    render_general_markdown,
    render_paper_markdown,
    render_supplemental_markdown,
)
from fg_pipeline.eval.schemas import MetricArtifact, ModelSpec


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = [
            ModelSpec(
                model_id="base",
                model_path="models/base",
                model_base=None,
                kind="base",
                conv_mode="vicuna_v1",
            ),
            ModelSpec(
                model_id="ours",
                model_path="models/ours",
                model_base="models/base",
                kind="lora",
                conv_mode="vicuna_v1",
            ),
        ]

    def test_build_paper_comparison_computes_deltas(self) -> None:
        rows = build_paper_comparison(
            [
                MetricArtifact(
                    benchmark="pope_adv",
                    model_id="base",
                    metrics={"f1": 80.0},
                    comparable_to_paper=True,
                    comparison_note=None,
                ),
                MetricArtifact(
                    benchmark="pope_adv",
                    model_id="ours",
                    metrics={"f1": 84.0},
                    comparable_to_paper=True,
                    comparison_note=None,
                ),
            ],
            self.models,
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].delta_vs_baseline or 0.0, 4.0)
        self.assertAlmostEqual(rows[0].paper_reference_value or 0.0, 84.9)
        self.assertTrue(rows[0].strictly_comparable)

    def test_markdown_renderers_handle_missing_values(self) -> None:
        paper = render_paper_markdown(self.models, [], {"pope_adv": {"status": "skipped", "note": "missing"}})
        self.assertIn("Benchmark Availability", paper)
        supplemental = render_supplemental_markdown([], {"amber": {"status": "skipped", "note": "proxy"}})
        self.assertIn("Supplemental Local Evaluation", supplemental)
        general = render_general_markdown({"stage_metrics": {}, "benchmarks": []})
        self.assertIn("General Evaluation", general)


if __name__ == "__main__":
    unittest.main()
