from __future__ import annotations

import unittest

from fg_pipeline.eval.schemas import ComparisonRow, ModelSpec


class EvalSchemasTests(unittest.TestCase):
    def test_model_spec_round_trip(self) -> None:
        payload = {
            "model_id": "base",
            "model_path": "models/llava-v1.5-13b",
            "model_base": None,
            "kind": "base",
            "conv_mode": "vicuna_v1",
            "temperature": 0.0,
            "num_beams": 1,
            "max_new_tokens": 512,
        }
        spec = ModelSpec.from_dict(payload)
        self.assertEqual(spec.to_dict(), payload)

    def test_comparison_row_serializes_direction(self) -> None:
        row = ComparisonRow(
            benchmark="pope_adv",
            metric="f1",
            direction="higher_better",
            base_value=0.8,
            our_value=0.9,
            paper_reference_value=0.85,
            delta_vs_base=0.1,
            relative_delta_vs_base=0.125,
            paper_row_name="HSA-DPO w/ LLaVA-1.5",
            note=None,
        )
        payload = row.to_dict()
        self.assertEqual(payload["direction"], "higher_better")
        self.assertEqual(payload["benchmark"], "pope_adv")


if __name__ == "__main__":
    unittest.main()
