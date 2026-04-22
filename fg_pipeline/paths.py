from __future__ import annotations

from pathlib import Path


FG_PIPELINE_ROOT = Path(__file__).resolve().parent
FG_DATA_DIR = FG_PIPELINE_ROOT / "data"
REPO_ROOT = FG_PIPELINE_ROOT.parent

# Stage-1-owned mirror of the released detection annotations (input).
DEFAULT_DETECTION_INPUT = FG_DATA_DIR / "hsa_dpo_detection.jsonl"
DEFAULT_SMOKE_DETECTION_INPUT = FG_DATA_DIR / "smoke_detection.jsonl"

# Stage 1 default output layout.
STAGE1_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage1"
DEFAULT_STAGE1_OUTPUT = STAGE1_OUTPUT_DIR / "detection_critiques.jsonl"
DEFAULT_STAGE1_STATS = STAGE1_OUTPUT_DIR / "stats.json"

# Stage 1 reads the same mirrored file by default.
DEFAULT_STAGE1_INPUT = DEFAULT_DETECTION_INPUT

# Stage 2 default output layout.
STAGE2_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage2"
DEFAULT_STAGE2_OUTPUT = STAGE2_OUTPUT_DIR / "rewrites.jsonl"
DEFAULT_STAGE2_STATS = STAGE2_OUTPUT_DIR / "stats.json"

# Stage 2 reads Stage 1 output by default.
DEFAULT_STAGE2_INPUT = DEFAULT_STAGE1_OUTPUT

# Stage 3 default output layout.
STAGE3_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage3"
DEFAULT_STAGE3_OUTPUT = STAGE3_OUTPUT_DIR / "vote_records.jsonl"
DEFAULT_STAGE3_PREFERENCES = STAGE3_OUTPUT_DIR / "preference_pairs.jsonl"
DEFAULT_STAGE3_STATS = STAGE3_OUTPUT_DIR / "stats.json"

# Stage 3 reads Stage 2 output by default.
DEFAULT_STAGE3_INPUT = DEFAULT_STAGE2_OUTPUT

# Stage 4 project pipeline defaults (wrapper over the existing baseline trainer).
STAGE4_OUTPUT_DIR = REPO_ROOT / "output" / "fghd" / "stage4_llava"
DEFAULT_STAGE4_DATA = DEFAULT_STAGE3_PREFERENCES
