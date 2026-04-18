# Project Workflow

This file is the short team-facing map of the repo.
It explains:

- what belongs to the original baseline
- what belongs to our new pipeline
- what we are doing next

For the full research method, see [README.md](README.md).

## Last Update

- Stage 3 Batch 1 scaffolding was added under `fg_pipeline/confidence/`.
- We now have a parser, scorer interface, smoke fixture, and Stage 3 tests.
- This was a structure-first update only: real `c^j` scoring is still not implemented.
- One semantic issue is still open: Stage 3 currently treats the teacher annotation text as `candidate_response`, and that must be corrected before Stage 4/5/6 work continues.

## Current Architecture

The repo has two layers.

- `hsa_dpo/`
  Original HSA-DPO baseline. Keep this layer stable.
- `fg_pipeline/`
  Our new project layer for Stages 3-6.

## Folder Layout

```text
.
├── hsa_dpo/
│   ├── data/
│   │   ├── hsa_dpo_preference_llava1dot5.jsonl   ← baseline preference data
│   │   ├── hsa_dpo_detection.jsonl               ← original released detection copy
│   │   └── images/                               ← baseline training images
│   ├── models/llava-v1_5/                        ← original training entrypoint
│   └── trainer/                                  ← original DPO trainer
│
├── fg_pipeline/
│   ├── data/
│   │   └── hsa_dpo_detection.jsonl               ← Stage 3 working mirror
│   ├── confidence/                               ← Stage 3
│   ├── rewrite/                                  ← Stage 4
│   ├── verification/                             ← Stage 5
│   ├── adaptive_dpo/                             ← Stage 6
│   ├── paths.py                                  ← fg_pipeline-owned default paths
│   ├── schemas.py                                ← shared row formats
│   └── io_utils.py                               ← JSONL helpers
│
├── scripts/
│   ├── run_stage3_confidence.sh
│   ├── run_stage4_rewrite.sh
│   ├── run_stage5_verify.sh
│   └── run_stage6_train.sh
│
└── vg/images/                                    ← Visual Genome images for detection
```

## Ownership Rules

- Do not remove or rename files under `hsa_dpo/` unless we explicitly decide to patch the baseline.
- Stage 3 now defaults to `fg_pipeline/data/hsa_dpo_detection.jsonl`.
- `fg_pipeline/data/hsa_dpo_detection.jsonl` is a mirror of `hsa_dpo/data/hsa_dpo_detection.jsonl`.
- `vg/images/` is only for detection-stage data.
- `hsa_dpo/data/images/` is only for baseline preference training.
- Do not mix the two image stores.

## What Exists Today

- Stage 3 has a bootstrap scaffold in `fg_pipeline/confidence/`.
- Stage 4 is still placeholder rewrite logic.
- Stage 5 is still heuristic filtering.
- Stage 6 still reuses the original trainer.

This means the project structure is ready, but the new research method is not fully implemented yet.

## What We Will Do Next

We start with Stage 3.

1. Clean Stage 3 parsing of the released detection annotations into per-signal labels.
2. Define `c^j` clearly and make Stage 3 emit confidence with stable semantics.
3. Keep Stage 3 output compatible with Stage 4, Stage 5, and Stage 6.
4. After Stage 3 is stable, replace Stage 4 placeholder rewrite logic.
5. Then fix the Stage 5 to Stage 6 bridge so clean pairs match trainer expectations.
6. Finally wire confidence-weighted severity into Stage 6 training.

## Immediate Team Focus

- Read from `fg_pipeline/data/hsa_dpo_detection.jsonl` for Stage 3 work.
- Edit inside `fg_pipeline/` first.
- Treat `hsa_dpo/` as the baseline layer.
- Keep changes small and traceable by stage.
