## FG Pipeline Data

This folder holds data files owned by the `fg_pipeline/` extension layer.

Current contents:
- `hsa_dpo_detection.jsonl`: Stage 3 detection input mirror

Design rule:
- keep the original baseline copy at `hsa_dpo/data/hsa_dpo_detection.jsonl`
- use the mirrored file here as the default Stage 3 input path
- do not move or rename `vg/images/`; the JSONL still references that image store
- if the baseline detection file is refreshed, refresh this mirror as well

Reason:
- Stage 3 belongs to `fg_pipeline/`
- the original `hsa_dpo/` tree should stay usable as the baseline layer
