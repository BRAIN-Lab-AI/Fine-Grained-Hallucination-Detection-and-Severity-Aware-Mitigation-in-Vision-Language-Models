from __future__ import annotations

from pathlib import Path


FG_PIPELINE_ROOT = Path(__file__).resolve().parent
FG_DATA_DIR = FG_PIPELINE_ROOT / "data"

# Stage-3-owned mirror of the released detection annotations.
DEFAULT_DETECTION_INPUT = FG_DATA_DIR / "hsa_dpo_detection.jsonl"

