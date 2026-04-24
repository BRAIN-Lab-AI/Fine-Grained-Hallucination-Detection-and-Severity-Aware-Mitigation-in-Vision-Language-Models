"""Stage 4 repair tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.stage4 import build_repair_prompt
from fg_pipeline.stage4.run_stage4_repair import main as stage4_main


def _critique() -> dict:
    return {
        "hallucination_type": "object",
        "severity_label": "major",
        "severity_score": 3,
        "rationale": "The gold cross is not visible.",
        "evidence_text": "gold cross",
    }


def _stage3_row(*, row_id: int, passed: bool, rewrite: str = "The image shows a gold cross.") -> dict:
    return {
        "id": row_id,
        "image": "vg/images/test.jpg",
        "question": "What is in the image?",
        "original_response": "The image shows a gold cross on the tower.",
        "rewrite_response": rewrite,
        "critiques": [_critique()],
        "votes": [
            {
                "vote_index": 1,
                "criterion": "hallucination_removal",
                "approved": False,
                "reason": "Unsupported object remains.",
                "model_family": "gemini",
                "backend_name": "gemini_two_vote",
            }
        ],
        "approvals": 0,
        "rejections": 1,
        "passed_majority": passed,
        "response_severity_score": 3.0,
        "chosen": rewrite if passed else None,
        "rejected": "The image shows a gold cross on the tower." if passed else None,
        "metadata": {"backend": "gemini_two_vote", "vote_policy_version": "gemini_two_vote_v1"},
    }


def _stage3_pref(row_id: int) -> dict:
    return {
        "id": row_id,
        "question": "What is in the image?",
        "chosen": "The image shows the tower.",
        "rejected": "The image shows a gold cross on the tower.",
        "chosen_score": 1.0,
        "rejected_score": 3.0,
        "image": "vg/images/test.jpg",
        "metadata": {"source_stage": "stage3_preference"},
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Stage4PromptTests(unittest.TestCase):
    def test_prompt_includes_failed_rewrite_and_vote_reason(self) -> None:
        prompt = build_repair_prompt(_stage3_row(row_id=1, passed=False))
        self.assertIn("Previous rewrite that failed validation", prompt)
        self.assertIn("The image shows a gold cross.", prompt)
        self.assertIn("Unsupported object remains.", prompt)
        self.assertIn("The gold cross is not visible.", prompt)


class Stage4CLITests(unittest.TestCase):
    def test_repairs_only_rejected_rows_and_combines_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            stage3_audit = tmp_dir / "stage3.jsonl"
            stage3_prefs = tmp_dir / "stage3_prefs.jsonl"
            repair_records = tmp_dir / "repair.jsonl"
            repair_prefs = tmp_dir / "repair_prefs.jsonl"
            final_prefs = tmp_dir / "final_prefs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(stage3_audit, [_stage3_row(row_id=1, passed=True), _stage3_row(row_id=2, passed=False)])
            write_jsonl(stage3_prefs, [_stage3_pref(1)])

            rc = stage4_main(
                [
                    "--input", str(stage3_audit),
                    "--stage3-preferences", str(stage3_prefs),
                    "--output", str(repair_records),
                    "--repair-preferences-out", str(repair_prefs),
                    "--final-preferences-out", str(final_prefs),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                ]
            )

            self.assertEqual(rc, 0)
            repairs = _read_jsonl(repair_records)
            repair_pref_rows = _read_jsonl(repair_prefs)
            final_pref_rows = _read_jsonl(final_prefs)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

            self.assertEqual([row["id"] for row in repairs], [2])
            self.assertEqual([row["id"] for row in repair_pref_rows], [2])
            self.assertEqual([row["id"] for row in final_pref_rows], [1, 2])
            self.assertEqual(stats["stage3_approved_pairs"], 1)
            self.assertEqual(stats["repair_pairs_emitted"], 1)
            self.assertEqual(stats["final_preference_pairs"], 2)

    def test_resume_does_not_duplicate_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            stage3_audit = tmp_dir / "stage3.jsonl"
            stage3_prefs = tmp_dir / "stage3_prefs.jsonl"
            repair_records = tmp_dir / "repair.jsonl"
            repair_prefs = tmp_dir / "repair_prefs.jsonl"
            final_prefs = tmp_dir / "final_prefs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(stage3_audit, [_stage3_row(row_id=2, passed=False)])
            write_jsonl(stage3_prefs, [])

            args = [
                "--input", str(stage3_audit),
                "--stage3-preferences", str(stage3_prefs),
                "--output", str(repair_records),
                "--repair-preferences-out", str(repair_prefs),
                "--final-preferences-out", str(final_prefs),
                "--stats-out", str(stats_path),
                "--backend", "template",
            ]
            self.assertEqual(stage4_main(args), 0)
            self.assertEqual(stage4_main(args + ["--resume"]), 0)

            self.assertEqual(len(_read_jsonl(repair_records)), 1)
            self.assertEqual(len(_read_jsonl(repair_prefs)), 1)
            self.assertEqual(len(_read_jsonl(final_prefs)), 1)

    def test_missing_stage3_preferences_means_zero_approved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            stage3_audit = tmp_dir / "stage3.jsonl"
            missing_stage3_prefs = tmp_dir / "missing_stage3_prefs.jsonl"
            repair_records = tmp_dir / "repair.jsonl"
            repair_prefs = tmp_dir / "repair_prefs.jsonl"
            final_prefs = tmp_dir / "final_prefs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(stage3_audit, [_stage3_row(row_id=2, passed=False)])

            rc = stage4_main(
                [
                    "--input", str(stage3_audit),
                    "--stage3-preferences", str(missing_stage3_prefs),
                    "--output", str(repair_records),
                    "--repair-preferences-out", str(repair_prefs),
                    "--final-preferences-out", str(final_prefs),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertEqual([row["id"] for row in _read_jsonl(repair_records)], [2])
            self.assertEqual([row["id"] for row in _read_jsonl(final_prefs)], [2])
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["stage3_approved_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
