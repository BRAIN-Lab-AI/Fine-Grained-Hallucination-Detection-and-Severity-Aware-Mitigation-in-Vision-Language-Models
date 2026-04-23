from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fg_pipeline.eval.run_eval import main


class PopeSmokeTests(unittest.TestCase):
    def test_run_eval_smoke_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pope_dir = tmp / "pope"
            image_dir = pope_dir / "val2014"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "0.jpg").write_bytes(b"fake")
            question_file = pope_dir / "llava_pope_test.jsonl"
            question_file.write_text(
                json.dumps(
                    {
                        "id": "0",
                        "question": "Is there a cat?",
                        "image": "0.jpg",
                        "label": "yes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = tmp / "models.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "base",
                            "model_path": "models/base",
                            "model_base": None,
                            "kind": "base",
                            "conv_mode": "vicuna_v1",
                            "temperature": 0.0,
                            "num_beams": 1,
                            "max_new_tokens": 32,
                        },
                        {
                            "model_id": "ours",
                            "model_path": "models/ours",
                            "model_base": "models/base",
                            "kind": "lora",
                            "conv_mode": "vicuna_v1",
                            "temperature": 0.0,
                            "num_beams": 1,
                            "max_new_tokens": 32,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            def fake_generate(model, records):
                return [
                    {
                        "id": record["id"],
                        "question": record["question"],
                        "image": record["image"],
                        "text": "yes",
                    }
                    for record in records
                ]

            argv = [
                "run_eval.py",
                "--run-name",
                "smoke",
                "--models-json",
                str(manifest),
                "--benchmarks",
                "pope_adv",
                "--paper-core",
                "--output-root",
                str(tmp / "output"),
                "--dataset-root-override",
                str(tmp),
            ]
            with patch("sys.argv", argv):
                with patch("fg_pipeline.eval.benchmarks.pope.generate_answers_for_records", side_effect=fake_generate):
                    rc = main()
            self.assertEqual(rc, 0)
            comparison = tmp / "output" / "smoke" / "comparison"
            self.assertTrue((comparison / "paper_core.json").exists())
            self.assertTrue((comparison / "paper_core.md").exists())
            self.assertTrue((comparison / "supplemental_eval.json").exists())
            self.assertTrue((comparison / "summary.csv").exists())

    def test_strict_paper_eval_rejects_unfair_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "models.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "base",
                            "model_path": "models/base",
                            "model_base": None,
                            "kind": "base",
                            "conv_mode": "vicuna_v1",
                            "temperature": 0.2,
                            "num_beams": 1,
                            "max_new_tokens": 32,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            argv = [
                "run_eval.py",
                "--run-name",
                "strict",
                "--models-json",
                str(manifest),
                "--paper-core",
                "--output-root",
                str(tmp / "output"),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertIn("temperature=0.0", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
