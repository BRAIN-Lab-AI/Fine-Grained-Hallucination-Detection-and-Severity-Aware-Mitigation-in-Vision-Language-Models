# FG Pipeline Data

This folder holds data files owned by the `fg_pipeline/` extension layer.

Current contents:

- `hsa_dpo_detection.jsonl` — Stage 1 input mirror of the released
  fine-grained supervision.
- `smoke_detection.jsonl` — small Stage 1 smoke fixture covering both
  `NO HALLUCINATION` and hallucinated rows.

Design rule:

- keep the original baseline copy at `hsa_dpo/data/hsa_dpo_detection.jsonl`
- use the mirrored file here as the default Stage 1 input path
- do not move or rename `vg/images/`; the JSONL still references that image store
- if the baseline detection file is refreshed, refresh this mirror as well

Reason:

- Stage 1 (critique detection / extraction) belongs to `fg_pipeline/`
- the original `hsa_dpo/` tree should stay usable as the baseline layer
