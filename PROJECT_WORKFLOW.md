# Project Workflow

This file is the short team-facing map of the repo.
It explains:

- what belongs to the original baseline
- what belongs to our new pipeline
- what we are doing next

For the full research method, see [README.md](README.md).

## Last Update

- **Stage 3 is finished (code + tests).** Parser, real `c^j` scorer, calibration tooling, CLIs, and wrapper scripts are all in place; 39 CPU tests pass (16 parser + 14 LogProbScorer + 9 calibrate).
- **Parser / schema** ([fg_pipeline/confidence/parser.py](fg_pipeline/confidence/parser.py)): `candidate_response = yhat` (Stage-1 description under assessment), `raw_detection` = GPT-4/GPT-4V annotation, `prompt` = canonical `Describe this image in detail.`; `D_det` signals carry `(h_type, severity, span, rationale, c^j, is_placeholder)`.
- **LogProbScorer** ([fg_pipeline/confidence/scorers_logprob.py](fg_pipeline/confidence/scorers_logprob.py)): computes `c^j` as the teacher-forced next-token log-probability of the severity label (Minor / Moderate / Major) under base LLaVA-v1.5. Lazy torch / transformers / vendored-LLaVA imports keep the module CPU-importable. Scorer factory in [fg_pipeline/confidence/scoring.py](fg_pipeline/confidence/scoring.py) resolves scorers by `module:attr` string; [fg_pipeline/confidence/run_detect.py](fg_pipeline/confidence/run_detect.py) exposes `--model-path / --device / --temperature / --image-root`; [scripts/run_stage3_confidence.sh](scripts/run_stage3_confidence.sh) is env-var driven (`SCORER`, `MODEL_PATH`, `DEVICE`, `TEMPERATURE`, `IMAGE_ROOT`).
- **Calibration** ([fg_pipeline/confidence/run_calibrate.py](fg_pipeline/confidence/run_calibrate.py), [scripts/run_stage3_calibrate.sh](scripts/run_stage3_calibrate.sh)): summarizes `D_det.jsonl` confidences (overall / deciles / per-type / per-severity) and proposes τ / τ_c candidates at quantiles 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.50. Placeholder and error signals are excluded; all-placeholder input prints a warning and exits without suggestions.
- **GPU requirement**: the `log_prob` scorer runs a teacher-forced forward pass through LLaVA-v1.5 (~14 GB VRAM fp16 for 7B, ~26 GB for 13B) — a CUDA GPU is required to generate the real `D_det.jsonl`. The `bootstrap` scorer and calibration CLI remain CPU-only.
- **Next action (execution, not code)**: on a GPU box run `SCORER=log_prob MODEL_PATH=models/llava-v1.5-13b bash scripts/run_stage3_confidence.sh` over `fg_pipeline/data/hsa_dpo_detection.jsonl`, then `bash scripts/run_stage3_calibrate.sh` and record the chosen τ (Stage 4 signal filter) and τ_c (Stage 5 pair filter) here before Stage 4 work begins.

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
- Locked decision: `c^j` = token log-probability (exp-normalized over the emitted type/severity span).
- Current repo state: Batch 1 landed parser + `ConfidenceScorer` protocol + `BootstrapScorer` placeholder (`c^j = 1.0`, `is_placeholder = True`). Real `LogProbScorer` lands in Batch 2.

### Stage 4 — Confidence-Guided Detect-then-Rewrite *(ours)*

- Purpose: rewrite `yhat_i` into `y_i` using only reliable hallucination signals.
- Input: `(x_i, yhat_i, H_i)`.
- Process:
  1. Filter signals by confidence threshold `τ`: `H_i^filtered = {h^j | c^j > τ}`.
  2. Rewrite: `y_i = M_wri(yhat_i, H_i^filtered)`.
- Output: `D_rewrite = {(x_i, yhat_i, y_i, H_i^filtered)}`.
- Novelty: rewrites are conditioned only on high-confidence hallucinations — noisy signals are dropped before they can poison the rewrite.
- Current repo state: placeholder passthrough in [fg_pipeline/rewrite/run_rewrite.py](fg_pipeline/rewrite/run_rewrite.py). Real `M_wri` not yet wired.

### Stage 5 — Verification & Filtering *(ours)*

- Purpose: keep only clean, trustworthy preference pairs.
- Input: `D_rewrite`.
- Process:
  1. Validate that `y_i` is actually better than `yhat_i`.
  2. Compute pair-level mean confidence `c̄_i = (1/T) Σ c^j` over signals in `H_i^filtered`.
  3. Keep if `c̄_i > τ_c`.
- Output: `D_pref^clean = {(x_i, yhat_i, y_i, H_i)}`.
- Novelty: pair-level confidence gate `τ_c` is our second confidence threshold; `τ` (Stage 4) and `τ_c` (Stage 5) operate at different granularities.
- Current repo state: heuristic filter in [fg_pipeline/verification/run_verify.py](fg_pipeline/verification/run_verify.py); real verifier not yet wired; Stage 5→6 image bridge still unresolved.

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
- Current repo state: adaptive-loss stub in [fg_pipeline/adaptive_dpo/](fg_pipeline/adaptive_dpo/); Stage 6 still reuses the original trainer path.

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

- Stage 3 has a Batch 1 scaffold in `fg_pipeline/confidence/` (parser + scorer interface + bootstrap scorer + tests + diagnostics).
- Stage 4 is still placeholder rewrite logic.
- Stage 5 is still heuristic filtering.
- Stage 6 still reuses the original trainer.
- Open decisions: `τ` (Stage 4 signal filter), `τ_c` (Stage 5 pair filter), Stage 5→6 image bridge.

This means the project structure is ready, but the new research method is not fully implemented yet.

## What We Will Do Next

We start with Stage 3.

1. Fix the `candidate_response` / `raw_detection` semantics (Batch 2 cleanup).
2. Implement real `c^j` via `LogProbScorer` registered against the existing `ConfidenceScorer` protocol.
3. Keep Stage 3 output compatible with Stage 4, Stage 5, and Stage 6.
4. After Stage 3 is stable, replace Stage 4 placeholder rewrite logic.
5. Then fix the Stage 5 to Stage 6 bridge so clean pairs match trainer expectations.
6. Finally wire confidence-weighted severity into Stage 6 training.

## Immediate Team Focus

- Read from `fg_pipeline/data/hsa_dpo_detection.jsonl` for Stage 3 work.
- Edit inside `fg_pipeline/` first.
- Treat `hsa_dpo/` as the baseline layer.
- Keep changes small and traceable by stage.
