"""Stage 2 unit and smoke tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.io_utils import write_jsonl
from fg_pipeline.stage1.schemas import CritiqueItem, Stage1Record
from fg_pipeline.stage2 import (
    PROMPT_VERSION,
    Stage2Record,
    TemplateRewriteBackend,
    get_backend,
)
from fg_pipeline.stage2.backends import RewriteError
from fg_pipeline.stage2.prompts import build_rewrite_prompt
from fg_pipeline.stage2.run_stage2 import main as stage2_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            for frag in _CONFIDENCE_FRAGMENTS:
                testcase.assertNotIn(
                    frag,
                    key.lower(),
                    f"unexpected key {key!r} in Stage 2 record",
                )
            _assert_no_confidence_keys(testcase, value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_confidence_keys(testcase, item)


def _make_critique(
    idx: int = 1,
    h_type: str = "object",
    severity_label: str = "major",
    severity_score: int = 3,
    rationale: str = "The claim is wrong",
    evidence: str | None = "gold cross",
) -> CritiqueItem:
    return CritiqueItem(
        index=idx,
        hallucination_type=h_type,
        severity_label=severity_label,
        severity_score=severity_score,
        rationale=rationale,
        evidence_text=evidence,
        source_tag_text=None,
        source_score_text=None,
    )


def _make_stage1_hallucinated(
    row_id: int = 1,
    response: str = "The clock tower is adorned with a gold cross, making it prominent.",
    critiques: list[CritiqueItem] | None = None,
) -> Stage1Record:
    if critiques is None:
        critiques = [_make_critique()]
    return Stage1Record(
        id=row_id,
        image="vg/images/test.jpg",
        question=response,
        response_text=response,
        is_hallucinated=True,
        critiques=critiques,
        metadata={"source": "released_annotations", "raw_annotation_text": "..."},
    )


def _make_stage1_clean(row_id: int = 2) -> Stage1Record:
    text = "The clock tower is tall."
    return Stage1Record(
        id=row_id,
        image="vg/images/test.jpg",
        question=text,
        response_text=text,
        is_hallucinated=False,
        critiques=[],
        metadata={"source": "released_annotations", "raw_annotation_text": "NO HALLUCINATION"},
    )


def _stage1_records_as_dicts(*records: Stage1Record) -> list[dict]:
    return [r.to_dict() for r in records]


# ---------------------------------------------------------------------------
# Pipeline behavior
# ---------------------------------------------------------------------------

class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TemplateRewriteBackend()

    def test_non_hallucinated_row_is_skipped(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        clean = _make_stage1_clean().to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(self.backend, [clean], stats, strict=False, limit=None))
        self.assertEqual(rows, [])
        self.assertEqual(stats.non_hallucinated_skipped, 1)
        self.assertEqual(stats.rewrites_emitted, 0)
        self.assertEqual(stats.total_input, 1)

    def test_hallucinated_row_produces_one_rewrite(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        hal = _make_stage1_hallucinated().to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(self.backend, [hal], stats, strict=False, limit=None))
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats.rewrites_emitted, 1)
        self.assertEqual(stats.hallucinated, 1)

    def test_stage2_record_preserves_required_fields(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        record = _make_stage1_hallucinated(row_id=99, response="The dog is red.").to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(self.backend, [record], stats, strict=False, limit=None))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], 99)
        self.assertEqual(row["image"], "vg/images/test.jpg")
        self.assertEqual(row["question"], "The dog is red.")
        self.assertEqual(row["original_response"], "The dog is red.")
        self.assertIsNotNone(row["rewrite_response"])
        self.assertIsInstance(row["critiques"], list)
        self.assertGreater(len(row["critiques"]), 0)


# ---------------------------------------------------------------------------
# Template backend
# ---------------------------------------------------------------------------

class TemplateBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = TemplateRewriteBackend()

    def test_deterministic_output(self) -> None:
        record = _make_stage1_hallucinated()
        out1 = self.backend.rewrite(record)
        out2 = self.backend.rewrite(record)
        self.assertEqual(out1, out2)

    def test_removes_evidence_span(self) -> None:
        critique = _make_critique(evidence="gold cross")
        record = _make_stage1_hallucinated(
            response="The clock tower is adorned with a gold cross, making it prominent.",
            critiques=[critique],
        )
        result = self.backend.rewrite(record)
        self.assertNotIn("gold cross", result.lower())

    def test_falls_back_when_no_evidence_spans(self) -> None:
        critique = _make_critique(evidence=None)
        record = _make_stage1_hallucinated(
            response="The image shows a park.",
            critiques=[critique],
        )
        result = self.backend.rewrite(record)
        self.assertTrue(bool(result.strip()), "rewrite must be non-empty")

    def test_handles_dict_record(self) -> None:
        record_dict = _make_stage1_hallucinated().to_dict()
        result = self.backend.rewrite(record_dict)
        self.assertTrue(bool(result.strip()))

    def test_multiple_critiques_all_applied(self) -> None:
        critiques = [
            _make_critique(idx=1, evidence="gold cross"),
            _make_critique(idx=2, evidence="prominent"),
        ]
        record = _make_stage1_hallucinated(
            response="The clock tower is adorned with a gold cross, making it prominent.",
            critiques=critiques,
        )
        result = self.backend.rewrite(record)
        self.assertNotIn("gold cross", result.lower())
        self.assertNotIn("prominent", result.lower())


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

class PromptTests(unittest.TestCase):
    def test_prompt_contains_original_response(self) -> None:
        record = _make_stage1_hallucinated(response="A bright red car is parked outside.")
        prompt = build_rewrite_prompt(record)
        self.assertIn("A bright red car is parked outside.", prompt)

    def test_prompt_contains_critique_evidence(self) -> None:
        record = _make_stage1_hallucinated(critiques=[_make_critique(evidence="gold cross")])
        prompt = build_rewrite_prompt(record)
        self.assertIn("gold cross", prompt)

    def test_prompt_version_constant_is_stable(self) -> None:
        self.assertIsInstance(PROMPT_VERSION, str)
        self.assertTrue(PROMPT_VERSION.startswith("v"))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class MetadataTests(unittest.TestCase):
    def test_prompt_version_in_metadata(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        backend = TemplateRewriteBackend()
        hal = _make_stage1_hallucinated().to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(backend, [hal], stats, strict=False, limit=None))
        self.assertEqual(rows[0]["metadata"]["prompt_version"], PROMPT_VERSION)

    def test_backend_name_in_metadata(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        backend = TemplateRewriteBackend()
        hal = _make_stage1_hallucinated().to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(backend, [hal], stats, strict=False, limit=None))
        self.assertEqual(rows[0]["metadata"]["backend"], "template")

    def test_source_stage_in_metadata(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        backend = TemplateRewriteBackend()
        hal = _make_stage1_hallucinated().to_dict()
        stats = _Stats("template")
        rows = list(_run_pipeline(backend, [hal], stats, strict=False, limit=None))
        self.assertEqual(rows[0]["metadata"]["source_stage"], "stage2_rewrite")

    def test_no_confidence_keys_in_output(self) -> None:
        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        backend = TemplateRewriteBackend()
        records = _stage1_records_as_dicts(
            _make_stage1_hallucinated(), _make_stage1_clean()
        )
        stats = _Stats("template")
        rows = list(_run_pipeline(backend, records, stats, strict=False, limit=None))
        for row in rows:
            _assert_no_confidence_keys(self, row)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

class BackendRegistryTests(unittest.TestCase):
    def test_get_template_backend(self) -> None:
        backend = get_backend("template")
        self.assertIsInstance(backend, TemplateRewriteBackend)

    def test_get_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("not_a_backend")

    def test_get_llava_without_model_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("llava")


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class CLISmokeTests(unittest.TestCase):
    def _write_stage1_fixture(self, path: Path, records: list[Stage1Record]) -> None:
        write_jsonl(path, (r.to_dict() for r in records))

    def test_cli_skips_non_hallucinated_and_rewrites_hallucinated(self) -> None:
        records = [
            _make_stage1_clean(row_id=1),
            _make_stage1_hallucinated(row_id=2),
            _make_stage1_hallucinated(row_id=3),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            in_path = tmp_dir / "stage1.jsonl"
            out_path = tmp_dir / "stage2.jsonl"
            stats_path = tmp_dir / "stats.json"
            self._write_stage1_fixture(in_path, records)

            rc = stage2_main(
                [
                    "--input", str(in_path),
                    "--output", str(out_path),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            self.assertTrue(stats_path.exists())

            # Read output records
            out_records = []
            with out_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out_records.append(json.loads(line))

            self.assertEqual(len(out_records), 2, "two hallucinated rows → two rewrites")

            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertEqual(stats["total_input_rows"], 3)
            self.assertEqual(stats["hallucinated_rows"], 2)
            self.assertEqual(stats["non_hallucinated_skipped"], 1)
            self.assertEqual(stats["rewrites_emitted"], 2)
            self.assertEqual(stats["backend"], "template")

            for rec in out_records:
                _assert_no_confidence_keys(self, rec)

    def test_cli_fails_on_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = stage2_main(
                [
                    "--input", str(Path(tmp) / "nonexistent.jsonl"),
                    "--output", str(Path(tmp) / "out.jsonl"),
                    "--stats-out", str(Path(tmp) / "stats.json"),
                ]
            )
            self.assertEqual(rc, 2)

    def test_cli_stats_emitted_field(self) -> None:
        records = [_make_stage1_hallucinated(row_id=10)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            in_path = tmp_dir / "s1.jsonl"
            out_path = tmp_dir / "s2.jsonl"
            stats_path = tmp_dir / "stats.json"
            self._write_stage1_fixture(in_path, records)

            rc = stage2_main(
                [
                    "--input", str(in_path),
                    "--output", str(out_path),
                    "--stats-out", str(stats_path),
                    "--backend", "template",
                ]
            )
            self.assertEqual(rc, 0)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            self.assertIn("rewrites_emitted", stats)
            self.assertIn("backend", stats)
            self.assertNotIn("confidence", str(stats).lower())


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

class StrictModeTests(unittest.TestCase):
    def test_empty_rewrite_raises_in_strict_mode(self) -> None:
        class _EmptyBackend:
            name = "empty"

            def rewrite(self, record, *, strict=False):
                if strict:
                    raise RewriteError("intentionally empty")
                return ""

        from fg_pipeline.stage2.run_stage2 import _run_pipeline, _Stats
        hal = _make_stage1_hallucinated().to_dict()
        stats = _Stats("empty")
        with self.assertRaises(RewriteError):
            list(_run_pipeline(_EmptyBackend(), [hal], stats, strict=True, limit=None))


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

class SchemaRoundTripTests(unittest.TestCase):
    def test_stage2_record_round_trip(self) -> None:
        rec = Stage2Record(
            id=42,
            image="vg/images/x.jpg",
            question="A test question.",
            original_response="The sky is green.",
            rewrite_response="The sky is blue.",
            critiques=[{"index": 1, "hallucination_type": "attribute"}],
            metadata={
                "source_stage": "stage2_rewrite",
                "backend": "template",
                "prompt_version": "v1",
            },
        )
        data = rec.to_dict()
        self.assertEqual(data["id"], 42)
        self.assertEqual(data["original_response"], "The sky is green.")
        self.assertEqual(data["rewrite_response"], "The sky is blue.")
        self.assertEqual(data["metadata"]["source_stage"], "stage2_rewrite")
        _assert_no_confidence_keys(self, data)


if __name__ == "__main__":
    unittest.main()
