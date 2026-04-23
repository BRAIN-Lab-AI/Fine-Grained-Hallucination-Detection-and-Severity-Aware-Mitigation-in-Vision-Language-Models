"""Stage 3 unit and smoke tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.stage1.schemas import CritiqueItem
from fg_pipeline.stage2.schemas import Stage2Record
from fg_pipeline.stage3 import (
    APPROVALS_REQUIRED,
    ENSEMBLE_VOTE_POLICY_VERSION,
    HeuristicVerificationBackend,
    QwenLlavaEnsembleBackend,
    Stage3Record,
    VOTE_COUNT,
    VoteDecision,
    evaluate_votes,
    get_backend,
)
from fg_pipeline.stage3.backends import VerificationError
from fg_pipeline.stage3.run_stage3 import (
    _aggregate_severity,
    _process_rows,
    main as stage3_main,
)


_CONFIDENCE_FRAGMENTS = (
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
            for fragment in _CONFIDENCE_FRAGMENTS:
                testcase.assertNotIn(fragment, lowered, f"unexpected key {key!r}")
            _assert_no_confidence_keys(testcase, value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_confidence_keys(testcase, item)


def _make_critique(
    *,
    idx: int = 1,
    evidence: str | None = "gold cross",
    severity_score: int | None = 3,
) -> dict:
    return CritiqueItem(
        index=idx,
        hallucination_type="object",
        severity_label="major" if severity_score else "unknown",
        severity_score=severity_score,
        rationale="The claim introduces unsupported content",
        evidence_text=evidence,
        source_tag_text=None,
        source_score_text=None,
    ).to_dict()


def _make_stage2_record(
    *,
    row_id: int = 1,
    original: str = "The image shows a gold cross on the clock tower.",
    rewrite: str = "The image shows the clock tower.",
    critiques: list[dict] | None = None,
) -> Stage2Record:
    return Stage2Record(
        id=row_id,
        image="vg/images/test.jpg",
        question="What is in the image?",
        original_response=original,
        rewrite_response=rewrite,
        critiques=critiques or [_make_critique()],
        metadata={"source_stage": "stage2_rewrite", "backend": "template", "prompt_version": "v1"},
    )


class SeverityAggregationTests(unittest.TestCase):
    def test_mean_of_known_scores(self) -> None:
        critiques = [
            _make_critique(idx=1, severity_score=3),
            _make_critique(idx=2, severity_score=2),
            _make_critique(idx=3, severity_score=1),
        ]
        self.assertAlmostEqual(_aggregate_severity(critiques), 2.0)

    def test_unknown_only_defaults_to_one(self) -> None:
        critiques = [_make_critique(severity_score=None)]
        self.assertAlmostEqual(_aggregate_severity(critiques), 1.0)


class HeuristicBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_vote_one_approves_when_evidence_removed(self) -> None:
        record = _make_stage2_record().to_dict()
        vote = self.backend.vote(record, vote_index=1)
        self.assertTrue(vote.approved)
        self.assertEqual(vote.criterion, "hallucination_removal")

    def test_vote_one_rejects_when_evidence_remains(self) -> None:
        record = _make_stage2_record(
            rewrite="The image shows a gold cross on the clock tower. [corrected]"
        ).to_dict()
        vote = self.backend.vote(record, vote_index=1)
        self.assertFalse(vote.approved)

    def test_vote_two_rejects_corrected_marker(self) -> None:
        record = _make_stage2_record(
            rewrite="The image shows the clock tower. [corrected]"
        ).to_dict()
        vote = self.backend.vote(record, vote_index=2)
        self.assertFalse(vote.approved)

    def test_vote_three_approves_good_rewrite(self) -> None:
        record = _make_stage2_record().to_dict()
        vote = self.backend.vote(record, vote_index=3)
        self.assertTrue(vote.approved)


class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = HeuristicVerificationBackend()

    def test_two_of_three_votes_keeps_pair(self) -> None:
        good = _make_stage2_record().to_dict()
        audit_rows, pref_rows, stats = _process_rows(self.backend, [good], strict=False, limit=None)
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(len(pref_rows), 1)
        self.assertTrue(audit_rows[0]["passed_majority"])
        self.assertGreaterEqual(audit_rows[0]["approvals"], APPROVALS_REQUIRED)
        self.assertEqual(stats.preference_pairs_emitted, 1)

    def test_one_of_three_votes_drops_pair(self) -> None:
        bad = _make_stage2_record(
            row_id=2,
            rewrite="The image shows a gold cross on the clock tower. [corrected]",
            critiques=[_make_critique(evidence="gold cross")],
        ).to_dict()
        audit_rows, pref_rows, stats = _process_rows(self.backend, [bad], strict=False, limit=None)
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(pref_rows, [])
        self.assertFalse(audit_rows[0]["passed_majority"])
        self.assertEqual(stats.dropped_rows, 1)

    def test_preference_row_is_trainer_compatible(self) -> None:
        good = _make_stage2_record().to_dict()
        _, pref_rows, _ = _process_rows(self.backend, [good], strict=False, limit=None)
        self.assertEqual(len(pref_rows), 1)
        pref = pref_rows[0]
        self.assertEqual(pref["id"], 1)
        self.assertEqual(pref["question"], "What is in the image?")
        self.assertEqual(pref["chosen"], "The image shows the clock tower.")
        self.assertEqual(pref["rejected"], "The image shows a gold cross on the clock tower.")
        self.assertAlmostEqual(pref["rejected_score"], 3.0)
        self.assertEqual(pref["image"], "vg/images/test.jpg")

    def test_no_confidence_fields_in_audit_or_preferences(self) -> None:
        good = _make_stage2_record().to_dict()
        audit_rows, pref_rows, _ = _process_rows(self.backend, [good], strict=False, limit=None)
        for row in audit_rows + pref_rows:
            _assert_no_confidence_keys(self, row)


class BackendRegistryTests(unittest.TestCase):
    def test_get_backend_returns_heuristic(self) -> None:
        backend = get_backend("heuristic")
        self.assertIsInstance(backend, HeuristicVerificationBackend)

    def test_get_backend_ensemble_requires_model_paths(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("qwen_llava_ensemble")

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("not_a_backend")


class EnsembleBackendTests(unittest.TestCase):
    class _FakeRuntime:
        def __init__(self, responses: dict[str, str]) -> None:
            self.responses = responses

        def judge(self, record, criterion: str) -> str:
            return self.responses[criterion]

    def test_ensemble_vote_assigns_expected_families(self) -> None:
        backend = QwenLlavaEnsembleBackend(
            qwen_model_path="models/qwen",
            llava_model_path="models/llava",
            qwen_runtime=self._FakeRuntime(
                {
                    "hallucination_removal": '{"approved": true, "reason": "removed"}',
                    "overall_preference": '{"approved": true, "reason": "better"}',
                }
            ),
            llava_runtime=self._FakeRuntime(
                {"content_preservation": '{"approved": true, "reason": "preserved"}'}
            ),
        )
        record = _make_stage2_record().to_dict()
        vote1 = backend.vote(record, vote_index=1)
        vote2 = backend.vote(record, vote_index=2)
        vote3 = backend.vote(record, vote_index=3)
        self.assertEqual(vote1.model_family, "qwen")
        self.assertEqual(vote2.model_family, "llava")
        self.assertEqual(vote3.model_family, "qwen")

    def test_cross_family_requirement_blocks_two_qwen_approvals(self) -> None:
        backend = QwenLlavaEnsembleBackend(
            qwen_model_path="models/qwen",
            llava_model_path="models/llava",
            qwen_runtime=self._FakeRuntime(
                {
                    "hallucination_removal": '{"approved": true, "reason": "removed"}',
                    "overall_preference": '{"approved": true, "reason": "better"}',
                }
            ),
            llava_runtime=self._FakeRuntime(
                {"content_preservation": '{"approved": false, "reason": "not preserved"}'}
            ),
        )
        votes = [
            backend.vote(_make_stage2_record().to_dict(), vote_index=1),
            backend.vote(_make_stage2_record().to_dict(), vote_index=2),
            backend.vote(_make_stage2_record().to_dict(), vote_index=3),
        ]
        passed, meta = evaluate_votes(backend, votes)
        self.assertFalse(passed)
        self.assertFalse(meta["cross_family_approval_ok"])
        self.assertEqual(meta["approved_families"], ["qwen"])

    def test_cross_family_requirement_passes_with_qwen_and_llava(self) -> None:
        backend = QwenLlavaEnsembleBackend(
            qwen_model_path="models/qwen",
            llava_model_path="models/llava",
            qwen_runtime=self._FakeRuntime(
                {
                    "hallucination_removal": '{"approved": true, "reason": "removed"}',
                    "overall_preference": '{"approved": false, "reason": "too weak"}',
                }
            ),
            llava_runtime=self._FakeRuntime(
                {"content_preservation": '{"approved": true, "reason": "preserved"}'}
            ),
        )
        good = _make_stage2_record().to_dict()
        audit_rows, pref_rows, stats = _process_rows(backend, [good], strict=False, limit=None)
        self.assertEqual(len(audit_rows), 1)
        self.assertEqual(len(pref_rows), 1)
        self.assertEqual(stats.vote_policy_version, ENSEMBLE_VOTE_POLICY_VERSION)
        self.assertEqual(stats.approval_families_required, ["qwen", "llava"])
        self.assertTrue(audit_rows[0]["metadata"]["cross_family_approval_ok"])
        self.assertEqual(audit_rows[0]["metadata"]["approved_families"], ["llava", "qwen"])
        self.assertEqual(pref_rows[0]["metadata"]["approved_families"], ["llava", "qwen"])


class CLISmokeTests(unittest.TestCase):
    def test_cli_writes_audit_and_preferences(self) -> None:
        rows = [
            _make_stage2_record(row_id=1).to_dict(),
            _make_stage2_record(
                row_id=2,
                rewrite="The image shows a gold cross on the clock tower. [corrected]",
            ).to_dict(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "stage2.jsonl"
            output_path = tmp_dir / "stage3.jsonl"
            prefs_path = tmp_dir / "prefs.jsonl"
            stats_path = tmp_dir / "stats.json"
            write_jsonl(input_path, rows)

            rc = stage3_main(
                [
                    "--input", str(input_path),
                    "--output", str(output_path),
                    "--preferences-out", str(prefs_path),
                    "--stats-out", str(stats_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(prefs_path.exists())
            self.assertTrue(stats_path.exists())

            audit_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            pref_rows = [json.loads(line) for line in prefs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

            self.assertEqual(len(audit_rows), 2)
            self.assertEqual(len(pref_rows), 1)
            self.assertEqual(stats["total_input_rows"], 2)
            self.assertEqual(stats["vote_rows_processed"], 2)
            self.assertEqual(stats["preference_pairs_emitted"], 1)
            self.assertEqual(stats["dropped_rows"], 1)
            self.assertEqual(stats["backend"], "heuristic")
            self.assertEqual(stats["vote_count"], VOTE_COUNT)
            self.assertEqual(stats["approvals_required"], APPROVALS_REQUIRED)

    def test_missing_input_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rc = stage3_main(
                [
                    "--input", str(tmp_dir / "missing.jsonl"),
                    "--output", str(tmp_dir / "out.jsonl"),
                    "--preferences-out", str(tmp_dir / "prefs.jsonl"),
                    "--stats-out", str(tmp_dir / "stats.json"),
                ]
            )
            self.assertEqual(rc, 2)


class StrictModeTests(unittest.TestCase):
    def test_invalid_row_raises_in_strict_mode(self) -> None:
        invalid = _make_stage2_record(rewrite="").to_dict()
        with self.assertRaises(VerificationError):
            _process_rows(HeuristicVerificationBackend(), [invalid], strict=True, limit=None)


class SchemaRoundTripTests(unittest.TestCase):
    def test_vote_decision_round_trip(self) -> None:
        vote = VoteDecision(vote_index=1, criterion="overall_preference", approved=True, reason="good")
        self.assertEqual(vote.to_dict()["criterion"], "overall_preference")

    def test_stage3_record_round_trip(self) -> None:
        record = Stage3Record(
            id=7,
            image="vg/images/x.jpg",
            question="What is in the image?",
            original_response="The sky is green.",
            rewrite_response="The sky is blue.",
            critiques=[_make_critique()],
            votes=[VoteDecision(vote_index=1, criterion="overall_preference", approved=True, reason="good")],
            approvals=3,
            rejections=0,
            passed_majority=True,
            response_severity_score=2.0,
            chosen="The sky is blue.",
            rejected="The sky is green.",
            metadata={"source_stage": "stage3_verification"},
        )
        data = record.to_dict()
        self.assertEqual(data["id"], 7)
        self.assertEqual(data["approvals"], 3)
        self.assertTrue(data["passed_majority"])
        _assert_no_confidence_keys(self, data)


if __name__ == "__main__":
    unittest.main()
