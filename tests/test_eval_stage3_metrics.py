from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fg_pipeline.eval.utils import summarize_stage3
from fg_pipeline.io_utils import write_jsonl


class Stage3MetricsTests(unittest.TestCase):
    def test_summarize_stage3_reports_calibration_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            det_path = tmp / "D_det.jsonl"
            report_path = tmp / "D_det_calibration.json"
            write_jsonl(
                det_path,
                [
                    {
                        "raw_detection": "NO HALLUCINATION",
                        "signals": [],
                    },
                    {
                        "raw_detection": "Tags:\n<object>\ncat",
                        "signals": [
                            {
                                "hallucination_type": "object",
                                "severity": 3,
                                "confidence": 0.9,
                                "metadata": {"is_placeholder": False},
                            }
                        ],
                    },
                ],
            )
            report_path.write_text(
                json.dumps(
                    {
                        "group_threshold_policy": {
                            "global_threshold": 0.4,
                            "by_group": {
                                "object|3": {
                                    "insufficient_support": False,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_stage3(det_path, report_path)
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["real_signals"], 1)
            self.assertIn("brier", summary)
            self.assertIn("threshold_policy", summary)
            self.assertEqual(summary["threshold_policy"]["group_count"], 1)


if __name__ == "__main__":
    unittest.main()
