# Project Workflow

This file is the short team-facing map of the repo.
It explains:

- what belongs to the original baseline
- what belongs to our new pipeline
- what we are doing next

For the full research method, see [README.md](README.md).

## Last Update

- **Stages 3-6 are now implemented at the repo level.** Stage 3 detection, Stage 4 rewrite, Stage 5 verification/filtering, and Stage 6 adaptive DPO wiring are all in place; `python -m pytest` now passes `95/95`.
- **Stage 3 calibration is stronger now.** [fg_pipeline/confidence/run_calibrate.py](fg_pipeline/confidence/run_calibrate.py) fits post-hoc temperature scaling from the stored severity triplet log-probs, writes `D_det_calibrated.jsonl`, and emits a group-conditional `tau_{type,severity}` policy in `D_det_calibration.json`.
- **Stage 4 and Stage 5 can now consume threshold reports directly.** [fg_pipeline/rewrite/run_rewrite.py](fg_pipeline/rewrite/run_rewrite.py) accepts `--threshold-report` for group-conditional signal filtering, and [fg_pipeline/verification/run_verify.py](fg_pipeline/verification/run_verify.py) accepts `--threshold-report` for `tau_c`.
- **Stage 5 now has explicit `tau_c` selection.** [fg_pipeline/verification/run_select_threshold.py](fg_pipeline/verification/run_select_threshold.py) implements CRC / CV-CRC threshold selection relative to the current verifier target, and [scripts/run_stage5_select_threshold.sh](scripts/run_stage5_select_threshold.sh) is the shell entrypoint.
- **There is now a one-shot calibrated pipeline entrypoint.** [scripts/run_calibrated_pipeline.sh](scripts/run_calibrated_pipeline.sh) runs Stage 3 calibration -> Stage 4 grouped rewrite -> Stage 5 `tau_c` selection -> Stage 5 verification, and starts Stage 6 automatically only when the available GPU count meets the configured minimum.
- **Stage 4** ([fg_pipeline/rewrite/](fg_pipeline/rewrite/)): `D_det -> D_rewrite` is implemented with strict `c^j > τ` filtering, backend registry, smoke-only `template` rewrite backend, and real `llava` backend path. `D_rewrite` now carries `sample_id`, `image`, `prompt`, `source_response`, `rewritten_response`, `filtered_signals`, and rewrite metadata.
- **Stage 5** ([fg_pipeline/verification/](fg_pipeline/verification/)): `D_rewrite -> D_pref_clean` is implemented as filter-and-validate, not confidence redefinition. It uses the carried Stage 3 `c^j` values, applies strict `pair_confidence > τ_c`, verifies `rewritten_response != source_response`, and emits `id`, `question`, `chosen`, `rejected`, `image`, `pair_confidence`, `severity_weight`, `adaptive_weight`, plus metadata.
- **Stage 6 bridge** ([hsa_dpo/models/llava-v1_5/train_dpo.py](hsa_dpo/models/llava-v1_5/train_dpo.py), [fg_pipeline/adaptive_dpo/](fg_pipeline/adaptive_dpo/)): training now prefers the Stage 5 `image` field instead of reconstructing paths from `id`, threads `pair_confidence`, `severity_weight`, and `adaptive_weight` through dataset + collator, and uses `AdaptiveLlavaDPOTrainer` to put `severity_weight` inside the rejected DPO term by default. [scripts/run_stage6_train.sh](scripts/run_stage6_train.sh) now defaults `IMAGE_FOLDER` to repo root, `USE_REJECTED_SCORE=True`, and keeps outer `USE_ADAPTIVE_EXAMPLE_WEIGHT=False` unless you intentionally want an extra weighting variant.
- **Repo-owned evaluation is now in place.** [fg_pipeline/eval/](fg_pipeline/eval/) adds a project-owned comparison layer for the paper-core benchmarks (`mhalubench`, `mfhallubench`, `object_halbench`, `amber`, `mmhal_bench`, `pope_adv`, `llava_bench_wild`, `hss`) plus Stage 3-6 general metrics. The shell entrypoints are [scripts/run_paper_eval.sh](scripts/run_paper_eval.sh) and [scripts/run_general_eval.sh](scripts/run_general_eval.sh).
- **Stage compatibility review**: Stage `3 -> 4 -> 5 -> 6` was re-checked after the Stage 6 changes. The shared schema handoff is now compatible end to end: Stage 3 preserves `image/prompt/candidate_response/signals`, Stage 4 preserves `image/prompt/source_response/rewritten_response/filtered_signals`, Stage 5 preserves `image/question/chosen/rejected/pair_confidence/severity_weight/adaptive_weight`, and Stage 6 now consumes those fields directly.
- **Remaining execution work**: no GPU/DeepSpeed training run was executed in this environment. The next practical step is an end-to-end GPU run with calibrated `τ` / `τ_c` and, for real research data, a non-`template` Stage 4 rewrite backend. The `template` backend remains smoke-only and should not be treated as final Stage 4 data.

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
│   │   ├── hsa_dpo_detection.jsonl               ← Stage 3 working mirror
│   │   └── smoke_detection.jsonl                 ← 4-row Stage 3 fixture
│   ├── confidence/                               ← Stage 3 (parser, scorer, run_detect)
│   ├── rewrite/                                  ← Stage 4
│   ├── verification/                             ← Stage 5
│   ├── adaptive_dpo/                             ← Stage 6
│   ├── eval/                                     ← paper/general evaluation layer
│   ├── paths.py                                  ← fg_pipeline-owned default paths
│   ├── schemas.py                                ← shared row formats
│   └── io_utils.py                               ← JSONL helpers
│
├── scripts/
│   ├── run_stage3_confidence.sh
│   ├── run_stage4_rewrite.sh
│   ├── run_stage5_verify.sh
│   ├── run_stage6_train.sh
│   ├── run_paper_eval.sh
│   └── run_general_eval.sh
│
├── tests/
│   └── test_stage3_parser.py                     ← Stage 3 unit tests
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

## Full Pipeline (Stages 1–6, short form)

Notation: `x_i` = image + instruction, `yhat_i` = candidate LVLM response, `y_i` = rewritten response, `h_type^j ∈ {object, attribute, relationship}`, `HS^j ∈ {1,2,3}` (Minor/Moderate/Major), `c^j` ∈ [0,1] = per-signal confidence, `T` = number of sentence-level signals in `yhat_i`.

### Stage 1 — Hallucinatory Response Generation *(prior work)*

- Purpose: produce candidate responses that may hallucinate.
- Input: image + instruction, target LVLM `M`.
- Process: `yhat_i = M(x_i)`.
- Output: `D_hal = {(x_i, yhat_i)}`.

### Stage 2 — Fine-Grained Annotation via GPT-4 / GPT-4V *(prior work)*

- Purpose: teacher-label every sentence-level hallucination.
- Input: `(x_i, yhat_i) ∈ D_hal`.
- Process: split `yhat_i` into sentences `{yhat_i^j}`; GPT-4/GPT-4V emits `(h_type^j, HS^j, reason^j)` per sentence.
- Output: `D_faif = {(x_i, yhat_i^j, h_type^j, HS^j, reason^j)}`.
- In this repo: released as `hsa_dpo/data/hsa_dpo_detection.jsonl`; mirrored at `fg_pipeline/data/hsa_dpo_detection.jsonl` for Stage 3.

---

🚀 **Our contribution starts here (Stages 3–6).**

---

### Stage 3 — Confidence-Aware Hallucination Detection *(ours)*

- Purpose: replace GPT-4 at inference with a scalable, calibrated detector that **also emits confidence**.
- Input: `D_faif`.
- Process: train/run detector `M_det(x_i, yhat_i^j) → (h_type^j, HS^j, c^j)`.
- Output: `H_i = {(h_type^j, HS^j, c^j)}_{j=1..T}`; implicit dataset `D_det = {(x_i, yhat_i, H_i)}`.
- Novelty: introduces `c^j` — the load-bearing symbol that threads through Stages 4–6.
- Locked decision: `c^j` = token log-probability (exp-normalized over the emitted type/severity span), followed by optional post-hoc temperature scaling for threshold selection.
- Current repo state: implemented in `fg_pipeline/confidence/` with parser, real `LogProbScorer`, calibration utilities, `run_detect.py`, and `run_calibrate.py`. Calibration writes both `D_det_calibrated.jsonl` and `D_det_calibration.json` with group-conditional `tau_{type,severity}`.

### Stage 4 — Confidence-Guided Detect-then-Rewrite *(ours)*

- Purpose: rewrite `yhat_i` into `y_i` using only reliable hallucination signals.
- Input: `(x_i, yhat_i, H_i)`.
- Process:
  1. Filter signals by confidence threshold `τ`: `H_i^filtered = {h^j | c^j > τ}`.
  2. Rewrite: `y_i = M_wri(yhat_i, H_i^filtered)`.
- Output: `D_rewrite = {(x_i, yhat_i, y_i, H_i^filtered)}`.
- Novelty: rewrites are conditioned only on high-confidence hallucinations — noisy signals are dropped before they can poison the rewrite.
- Current repo state: implemented in `fg_pipeline/rewrite/` with a backend registry, smoke-only `template` backend, and real `llava` rewrite backend. `run_rewrite.py` accepts either a global `τ` or `--threshold-report` from Stage 3 calibration.

### Stage 5 — Verification & Filtering *(ours)*

- Purpose: keep only clean, trustworthy preference pairs.
- Input: `D_rewrite`.
- Process:
  1. Validate that `y_i` is actually better than `yhat_i`.
  2. Compute pair-level mean confidence `c̄_i = (1/T) Σ c^j` over signals in `H_i^filtered`.
  3. Keep if `c̄_i > τ_c`.
- Output: `D_pref^clean = {(x_i, yhat_i, y_i, H_i)}`.
- Novelty: pair-level confidence gate `τ_c` is our second confidence threshold; `τ` (Stage 4) and `τ_c` (Stage 5) operate at different granularities.
- Current repo state: implemented in `fg_pipeline/verification/` with heuristic verification, CRC / CV-CRC threshold selection, `run_select_threshold.py`, and `run_verify.py`. Stage 5 writes Stage 6-compatible rows including `image`, `pair_confidence`, `severity_weight`, and `adaptive_weight`.

### Stage 6 — Adaptive Severity-Aware DPO *(ours)*

- Purpose: train the final LVLM so that each pair is weighted by how severe AND how confident the hallucination is.
- Input: `D_pref^clean`.
- Process:
  1. Compute adaptive severity per example: `S_i^adaptive = (1/T) Σ c^j · HS^j`.
     (Optional stronger form: `S_i^adaptive = α · mean(c^j · HS^j) + (1 − α) · max(c^j · HS^j)`.)
  2. Train with DPO:
     `L = −log σ( β [ log π_θ(y_i|x_i) / π_ref(y_i|x_i) − S_i^adaptive · log π_θ(yhat_i|x_i) / π_ref(yhat_i|x_i) ] )`.
- Output: hallucination-mitigated LVLM `π_θ*`.
- Novelty vs baseline HSA-DPO: baseline weights by severity alone (`HS^j`). Ours weights by **`c^j · HS^j`** — confidence × severity — threaded from Stage 3.
- Current repo state: the Stage 6 bridge is implemented. `train_dpo.py` now prefers the Stage 5 `image` field, threads `pair_confidence`, `severity_weight`, and `adaptive_weight`, and uses `AdaptiveLlavaDPOTrainer` to apply `severity_weight` in the inner DPO objective by default. Optional outer example weighting remains available as an explicit variant. The known operational constraint is hardware: the current paper-like setup should run on a real 2-GPU box.

## Compact Pipeline View

```text
Stage 1:  (x, yhat)            → D_hal                      [prior work]
Stage 2:  D_hal  (GPT-4/GPT-4V) → D_faif                    [prior work]
─────────────────────────────────────────────────────────── ours below
Stage 3:  D_faif (M_det)        → H  (+ c^j)  → D_det
Stage 4:  (x, yhat, H)          → rewrite (τ filter) → D_rewrite
Stage 5:  D_rewrite             → verify + τ_c filter → D_pref_clean
Stage 6:  D_pref_clean          → adaptive DPO (c^j · HS^j) → π_θ*
```

Data evolution: `D_hal → D_faif → D_det → D_rewrite → D_pref_clean → π_θ*`.

One-line summary: **confidence-aware detection → filtered rewriting → verified preference learning → adaptive severity training.**

## What Exists Today

- Stage 3 detection, calibration, and grouped threshold reporting are implemented.
- Stage 4 real rewrite is implemented through the `llava` backend; the `template` backend remains smoke-only.
- Stage 5 verification and `tau_c` selection are implemented; the current verifier is heuristic, so threshold guarantees are relative to that verifier.
- Stage 6 adaptive DPO wiring is implemented and tested, but the paper-like configuration still needs a suitable multi-GPU machine.

This means the research pipeline is implemented at the repo level; remaining work is primarily GPU execution, threshold tuning, and stronger verification if you want tighter paper-level guarantees.

## What We Will Do Next

1. Run Stage 3 with the real `log_prob` scorer on the target GPU box and persist `D_det.jsonl`.
2. Use the calibrated/grouped Stage 4 -> Stage 5 path to regenerate `D_pref_clean_grouped.jsonl`.
3. Run Stage 6 on a real 2-GPU machine for the paper-like configuration.
4. Strengthen Stage 5 threshold selection with either a better verifier or a manually audited subset if you want stronger empirical guarantees.

## Execution Guide

Run from the repo root on the Linux GPU machine after the environment is bootstrapped and the base LLaVA model is available at `models/llava-v1.5-13b`.

### Step-by-step Stage 3 -> Stage 6

```bash
cd /workspace/Fine-Grained-Hallucination-Detection-and-Severity-Aware-Mitigation-in-Vision-Language-Models
source .venv/bin/activate
```

1. Stage 3 detect with the real `log_prob` scorer:

```bash
rm -rf output/fghd
INPUT=hsa_dpo/data/hsa_dpo_detection.jsonl \
SCORER=log_prob \
MODEL_PATH=models/llava-v1.5-13b \
IMAGE_ROOT="$(pwd)" \
bash scripts/run_stage3_confidence.sh
```

2. Stage 3 calibration with temperature scaling + group-conditional thresholds:

```bash
INPUT=output/fghd/D_det.jsonl \
OUTPUT_CALIBRATED=output/fghd/D_det_calibrated.jsonl \
REPORT=output/fghd/D_det_calibration.json \
bash scripts/run_stage3_calibrate.sh
```

3. Stage 4 grouped rewrite using the calibration report:

```bash
bash scripts/run_stage4_rewrite.sh \
  output/fghd/D_det_calibrated.jsonl \
  output/fghd/D_rewrite_grouped.jsonl \
  llava \
  --model-path models/llava-v1.5-13b \
  --image-root "$(pwd)" \
  --threshold-report output/fghd/D_det_calibration.json
```

4. Stage 5 `tau_c` selection with CRC / CV-CRC:

```bash
bash scripts/run_stage5_select_threshold.sh \
  output/fghd/D_rewrite_grouped.jsonl \
  output/fghd/D_tau_c_report_grouped.json \
  heuristic \
  --method cv_crc \
  --alpha 0.10 \
  --folds 5 \
  --min-accepted 100
```

5. Stage 5 final verification / filtering:

```bash
bash scripts/run_stage5_verify.sh \
  output/fghd/D_rewrite_grouped.jsonl \
  output/fghd/D_pref_clean_grouped.jsonl \
  heuristic \
  --threshold-report output/fghd/D_tau_c_report_grouped.json
```

6. Stage 6 adaptive DPO training on a real 2-GPU machine:

```bash
DATA_PATH=output/fghd/D_pref_clean_grouped.jsonl \
IMAGE_FOLDER="$(pwd)" \
MODEL_PATH=models/llava-v1.5-13b \
bash scripts/run_stage6_train.sh
```

### One-shot calibrated pipeline

This reruns Stage 3 calibration, Stage 4 grouped rewrite, Stage 5 `tau_c` selection, and Stage 5 verification automatically. It starts Stage 6 only if the machine has enough GPUs for the current configuration.

```bash
cd /workspace/Fine-Grained-Hallucination-Detection-and-Severity-Aware-Mitigation-in-Vision-Language-Models
source .venv/bin/activate
RUN_STAGE6=auto \
MODEL_PATH=models/llava-v1.5-13b \
IMAGE_ROOT="$(pwd)" \
bash scripts/run_calibrated_pipeline.sh
```

If you are already on a suitable 2-GPU machine and want Stage 6 to be required instead of skipped automatically:

```bash
RUN_STAGE6=true \
MODEL_PATH=models/llava-v1.5-13b \
IMAGE_ROOT="$(pwd)" \
bash scripts/run_calibrated_pipeline.sh
```

### Operational Notes

- Stage 6 is expected to OOM on a `1x RTX 4080 SUPER 32 GB` box with the current paper-like configuration.
- The current Stage 5 verifier is heuristic, so CRC / CV-CRC threshold guarantees are relative to that verifier.
- The `template` Stage 4 backend remains smoke-only; use `llava` for real pipeline runs.

## Evaluation Guide

The repo now has two evaluation entrypoints.

- `bash scripts/run_paper_eval.sh`
  Runs the paper-core wrapper and writes:
  `output/eval/<run_name>/comparison/paper_core.{json,md}` plus `summary.csv`.
- `bash scripts/run_general_eval.sh`
  Summarizes Stage 3-6 local metrics and any selected public benchmark subset.

Evaluation is manifest-driven. The runner expects:

- one base `LLaVA-1.5-13B` row
- one local improved model row
- `model_base` when the model kind is `lora`

Judge-based metrics currently include:

- `llava_bench_wild`
- `mmhal_bench`
- `hss`

These require `OPENAI_API_KEY` and `OPENAI_JUDGE_MODEL`.

The generated reports explicitly separate:

- paper reference values
- locally reproduced values
- local proxy values that are not yet strictly paper-comparable

## Immediate Team Focus

- Treat this local repo as the canonical development copy; Vast was only a run environment.
- Keep `hsa_dpo/` as the baseline layer and place new logic in `fg_pipeline/` or the stage scripts unless a compatibility patch is required.
- Use `scripts/run_calibrated_pipeline.sh` for the calibrated Stage 3-5 path.
- Use a 2-GPU box for real Stage 6 runs; the 1x 32 GB setup is expected to OOM.
