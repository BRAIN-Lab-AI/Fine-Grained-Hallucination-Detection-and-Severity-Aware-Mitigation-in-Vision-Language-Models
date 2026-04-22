"""Parser-level tests for Stage 1 critique extraction."""

from __future__ import annotations

import unittest

from fg_pipeline.stage1.parser import (
    ParseError,
    extract_question,
    parse_detection_row,
)


def _make_row(gpt_value: str, *, human: str | None = None, row_id: int = 1) -> dict:
    human_text = (
        human
        if human is not None
        else "<image>\nDescription to Assess:\nA test description."
    )
    return {
        "id": row_id,
        "image": "vg/images/test.jpg",
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": gpt_value},
        ],
    }


class ExtractQuestionTests(unittest.TestCase):
    def test_strips_image_header_and_description_prefix(self) -> None:
        raw = "<image>\nDescription to Assess:\nThe clock tower is tall"
        self.assertEqual(extract_question(raw), "The clock tower is tall")

    def test_preserves_sentence_when_wrappers_missing(self) -> None:
        raw = "The clock tower is tall"
        self.assertEqual(extract_question(raw), "The clock tower is tall")

    def test_handles_case_insensitive_description_prefix(self) -> None:
        raw = "<image>\ndescription to assess:\nHello"
        self.assertEqual(extract_question(raw), "Hello")


class NoHallucinationTests(unittest.TestCase):
    def test_exact_marker_produces_non_hallucinated_record(self) -> None:
        row = _make_row("NO HALLUCINATION", row_id=42)
        result = parse_detection_row(row)
        self.assertFalse(result.record.is_hallucinated)
        self.assertEqual(result.record.critiques, [])
        self.assertEqual(result.record.id, 42)
        self.assertEqual(result.record.question, "A test description.")
        self.assertEqual(result.record.response_text, "A test description.")
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.record.metadata["source"], "released_annotations")
        self.assertEqual(result.record.metadata["raw_annotation_text"], "NO HALLUCINATION")
        self.assertNotIn("parse_warnings", result.record.metadata)


class SingleCritiqueTests(unittest.TestCase):
    def test_single_object_critique_parses_all_fields(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . The women are not standing on a sidewalk, they are performing a dance routine\n"
            "Scores:\n"
            "<object>\n"
            "1 .  Sidewalk: Major (3 points): The location of the women is a significant detail"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertTrue(result.record.is_hallucinated)
        self.assertEqual(len(result.record.critiques), 1)
        self.assertEqual(result.record.response_text, "A test description.")
        self.assertEqual(result.record.metadata["raw_annotation_text"], gpt)
        critique = result.record.critiques[0]
        self.assertEqual(critique.index, 1)
        self.assertEqual(critique.hallucination_type, "object")
        self.assertEqual(critique.severity_label, "major")
        self.assertEqual(critique.severity_score, 3)
        self.assertIn("dance routine", critique.rationale)
        self.assertEqual(critique.evidence_text, "Sidewalk")
        self.assertIn("Sidewalk", critique.source_score_text)
        self.assertIn("dance routine", critique.source_tag_text)
        self.assertEqual(result.warnings, [])

    def test_minor_one_point_singular(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . Something minor\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Minor (1 point): nuance"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(result.record.critiques[0].severity_label, "minor")
        self.assertEqual(result.record.critiques[0].severity_score, 1)
        self.assertEqual(result.record.critiques[0].evidence_text, "Span")


class MultiTypeTests(unittest.TestCase):
    def test_multi_type_preserves_order_and_type_matching(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . Wrong number of umbrellas\n"
            "<attribute>\n"
            "1 . Wrong umbrella colors\n"
            "Scores:\n"
            "<object>\n"
            "1 . Umbrellas: Major (3 points): significant detail\n"
            "<attribute>\n"
            "1 . Umbrella hues: Moderate (2 points): noticeable detail"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(len(result.record.critiques), 2)
        obj, attr = result.record.critiques
        self.assertEqual(obj.hallucination_type, "object")
        self.assertEqual(obj.severity_label, "major")
        self.assertEqual(obj.severity_score, 3)
        self.assertEqual(obj.index, 1)
        self.assertEqual(attr.hallucination_type, "attribute")
        self.assertEqual(attr.severity_label, "moderate")
        self.assertEqual(attr.severity_score, 2)
        self.assertEqual(attr.index, 2)
        self.assertEqual(attr.evidence_text, "Umbrella hues")
        self.assertEqual(result.warnings, [])

    def test_items_without_numeric_prefix_are_accepted(self) -> None:
        # Mirrors id=8517 in the released file: <relationship> block with no "1 ." prefix.
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . The Lego creation is not described as a machine\n"
            "<relationship>\n"
            "The description suggests a relationship that is not supported\n"
            "Scores:\n"
            "<object>\n"
            "1 . Machine: Moderate (2 points): noticeable detail\n"
            "<relationship>\n"
            " Part of Lego creation: Moderate (2 points): noticeable detail"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(len(result.record.critiques), 2)
        rel = result.record.critiques[1]
        self.assertEqual(rel.hallucination_type, "relationship")
        self.assertEqual(rel.severity_label, "moderate")
        self.assertEqual(rel.severity_score, 2)
        self.assertEqual(rel.evidence_text, "Part of Lego creation")


class MismatchedSectionsTests(unittest.TestCase):
    def test_more_tags_than_scores_keeps_tags_with_unknown_severity(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . First claim\n"
            "2 . Second claim\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Major (3 points): detail"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(len(result.record.critiques), 2)
        self.assertEqual(result.record.critiques[0].severity_label, "major")
        self.assertEqual(result.record.critiques[1].severity_label, "unknown")
        self.assertIsNone(result.record.critiques[1].severity_score)
        self.assertTrue(
            any("recoverable" in w for w in result.record.metadata["parse_warnings"])
        )

    def test_score_only_type_logs_warning(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . Only object claim\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Minor (1 point): small\n"
            "<attribute>\n"
            "1 . Orphan: Major (3 points): detail"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(len(result.record.critiques), 1)
        self.assertTrue(
            any(
                "no matching Tags" in w
                for w in result.record.metadata["parse_warnings"]
            )
        )

    def test_hallucinated_without_sections_is_recoverable(self) -> None:
        gpt = "This is a malformed response without Tags or Scores."
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertTrue(result.record.is_hallucinated)
        self.assertEqual(result.record.critiques, [])
        self.assertEqual(result.record.response_text, "A test description.")
        self.assertEqual(result.record.metadata["raw_annotation_text"], gpt)
        self.assertIn("parse_warnings", result.record.metadata)

    def test_strict_mode_raises_on_mismatch(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . First\n"
            "2 . Second\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Major (3 points): detail"
        )
        row = _make_row(gpt)
        with self.assertRaises(ParseError):
            parse_detection_row(row, strict=True)


class MalformedSeverityTests(unittest.TestCase):
    def test_unknown_severity_text_yields_unknown(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . Some issue\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Something weird: explanation"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        self.assertEqual(len(result.record.critiques), 1)
        critique = result.record.critiques[0]
        self.assertEqual(critique.severity_label, "unknown")
        self.assertIsNone(critique.severity_score)

    def test_partially_formatted_severity_text_yields_unknown(self) -> None:
        gpt = (
            "Tags:\n"
            "<object>\n"
            "1 . Some issue\n"
            "Scores:\n"
            "<object>\n"
            "1 . Span: Majorish (4 points): explanation"
        )
        row = _make_row(gpt)
        result = parse_detection_row(row)
        critique = result.record.critiques[0]
        self.assertEqual(critique.severity_label, "unknown")
        self.assertIsNone(critique.severity_score)


if __name__ == "__main__":
    unittest.main()
