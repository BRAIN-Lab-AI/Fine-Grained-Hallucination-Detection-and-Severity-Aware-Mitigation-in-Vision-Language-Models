"""Backend and CLI smoke tests for Stage 1."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import read_jsonl, write_jsonl
from fg_pipeline.paths import DEFAULT_SMOKE_DETECTION_INPUT
from fg_pipeline.stage1 import (
    CritiqueDetectorBackend,
    LlavaDetectorBackend,
    ReleasedAnnotationBackend,
    get_backend,
    parse_detection_row,
)
from fg_pipeline.stage1.run_stage1_detector_dataset import main as detector_dataset_main
from fg_pipeline.stage1.run_stage1_export_benchmarks import main as export_benchmarks_main
from fg_pipeline.stage1.run_stage1 import main as stage1_main


_CONFIDENCE_KEY_FRAGMENTS = (
    "confidence",
    "calibration",
    "threshold",
    "tau",
    "crc",
    "cv_crc",
    "probability",
)


def _assert_no_confidence_keys(testcase: unittest.TestCase, obj) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = key.lower()
            for fragment in _CONFIDENCE_KEY_FRAGMENTS:
                testcase.assertNotIn(
                    fragment, lowered, f"unexpected key {key!r} in stage1 record"
                )
            _assert_no_confidence_keys(testcase, value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_confidence_keys(testcase, item)


class BackendContractTests(unittest.TestCase):
    def test_get_backend_returns_released_annotations_default(self) -> None:
        backend = get_backend("released_annotations")
        self.assertIsInstance(backend, ReleasedAnnotationBackend)
        self.assertIsInstance(backend, CritiqueDetectorBackend)

    def test_get_backend_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("not_a_real_backend")

    def test_get_backend_constructs_llava_detector(self) -> None:
        backend = get_backend("llava_detector", model_path="models/llava-v1.5-7b")
        self.assertIsInstance(backend, LlavaDetectorBackend)
        self.assertIsInstance(backend, CritiqueDetectorBackend)

    def test_backend_matches_direct_parser_path(self) -> None:
        row = {
            "id": 99,
            "image": "vg/images/x.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescription to Assess:\nA sentence.",
                },
                {
                    "from": "gpt",
                    "value": (
                        "Tags:\n"
                        "<object>\n"
                        "1 . Something off\n"
                        "Scores:\n"
                        "<object>\n"
                        "1 . Span: Major (3 points): detail"
                    ),
                },
            ],
        }
        backend = ReleasedAnnotationBackend()
        backend_record = backend.detect(row).record.to_dict()
        direct_record = parse_detection_row(row).record.to_dict()
        self.assertEqual(backend_record, direct_record)
        self.assertEqual(backend_record["response_text"], "A sentence.")
        self.assertEqual(
            backend_record["metadata"]["raw_annotation_text"],
            (
                "Tags:\n"
                "<object>\n"
                "1 . Something off\n"
                "Scores:\n"
                "<object>\n"
                "1 . Span: Major (3 points): detail"
            ),
        )


class CLISmokeTests(unittest.TestCase):
    def test_cli_runs_on_smoke_fixture(self) -> None:
        self.assertTrue(
            DEFAULT_SMOKE_DETECTION_INPUT.exists(),
            f"missing smoke fixture: {DEFAULT_SMOKE_DETECTION_INPUT}",
        )
        expected_count = sum(1 for _ in read_jsonl(DEFAULT_SMOKE_DETECTION_INPUT))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "detection_critiques.jsonl"
            stats_path = tmp_dir / "stats.json"
            rc = stage1_main(
                [
                    "--input",
                    str(DEFAULT_SMOKE_DETECTION_INPUT),
                    "--output",
                    str(out_path),
                    "--stats-out",
                    str(stats_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            self.assertTrue(stats_path.exists())

            records = list(read_jsonl(out_path))
            self.assertEqual(len(records), expected_count)

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["total_rows"], expected_count)
            self.assertEqual(
                stats["total_rows"],
                stats["hallucinated_rows"] + stats["non_hallucinated_rows"],
            )

            # Stage 1 records must not leak any confidence-related keys.
            for rec in records:
                _assert_no_confidence_keys(self, rec)

            # At least one hallucinated row is present in the smoke fixture.
            self.assertGreater(stats["hallucinated_rows"], 0)
            self.assertGreater(stats["non_hallucinated_rows"], 0)
            # Critique count buckets should also be confidence-free.
            _assert_no_confidence_keys(self, stats["critique_count_by_type"])
            _assert_no_confidence_keys(self, stats["critique_count_by_severity"])

    def test_question_extraction_strips_wrappers(self) -> None:
        row = {
            "id": 0,
            "image": "vg/images/x.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescription to Assess:\nThe clock tower is adorned with a gold cross",
                },
                {"from": "gpt", "value": "NO HALLUCINATION"},
            ],
        }
        record = ReleasedAnnotationBackend().detect(row).record
        self.assertEqual(
            record.question,
            "The clock tower is adorned with a gold cross",
        )
        self.assertEqual(
            record.response_text,
            "The clock tower is adorned with a gold cross",
        )
        self.assertFalse(record.is_hallucinated)

    def test_detector_dataset_cli_writes_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_path = tmp_dir / "detector_train.json"
            rc = detector_dataset_main(
                [
                    "--input",
                    str(DEFAULT_SMOKE_DETECTION_INPUT),
                    "--output",
                    str(out_path),
                    "--limit",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertIn("conversations", payload[0])

    def test_detector_dataset_human_turn_matches_inference_prompt(self) -> None:
        """Training human turn must use the same template as LlavaDetectorBackend at inference."""
        from fg_pipeline.stage1.prompts import build_detector_prompt

        row = {
            "id": 0,
            "image": "vg/images/x.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescription to Assess:\nThe clock shows noon.",
                },
                {"from": "gpt", "value": "NO HALLUCINATION"},
            ],
        }
        from fg_pipeline.stage1.detector_data import build_llava_detector_example
        example = build_llava_detector_example(row)
        human_turn = example["conversations"][0]["value"]
        # Must start with the image marker then the full instruction preamble.
        self.assertTrue(human_turn.startswith("<image>\n"))
        expected_prompt = build_detector_prompt(
            question="The clock shows noon.", response_text="The clock shows noon."
        )
        self.assertIn(expected_prompt, human_turn)
        # Must include the format instructions so the trained model knows the output format.
        self.assertIn("Tags:", human_turn)
        self.assertIn("Scores:", human_turn)
        self.assertIn("NO HALLUCINATION", human_turn)

    def test_benchmark_export_cli_projects_stage1_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            stage1_path = tmp_dir / "stage1.jsonl"
            annotation_path = tmp_dir / "annotations.jsonl"
            out_path = tmp_dir / "predictions.jsonl"
            write_jsonl(
                stage1_path,
                [
                    {
                        "id": "1",
                        "image": "vg/images/x.jpg",
                        "question": "Q",
                        "response_text": "R",
                        "is_hallucinated": True,
                        "critiques": [
                            {
                                "index": 1,
                                "hallucination_type": "object",
                                "severity_label": "major",
                                "severity_score": 3,
                                "rationale": "wrong object",
                                "evidence_text": "cross",
                            }
                        ],
                        "metadata": {"source": "llava_detector"},
                    }
                ],
            )
            write_jsonl(
                annotation_path,
                [
                    {"id": "1", "claim_label": 1, "segment_label": 1},
                ],
            )
            rc = export_benchmarks_main(
                [
                    "--benchmark",
                    "mhalubench",
                    "--stage1-input",
                    str(stage1_path),
                    "--annotation-input",
                    str(annotation_path),
                    "--output",
                    str(out_path),
                ]
            )
            self.assertEqual(rc, 0)
            exported = list(read_jsonl(out_path))
            self.assertEqual(exported[0]["claim_prediction"], 1)
            self.assertEqual(exported[0]["segment_prediction"], 1)


if __name__ == "__main__":
    unittest.main()
